# train_efficientnetb0_plants.py
# EfficientNetB0 image classification (5 plant classes) using PyTorch + GPU
#
# Dataset folder layout (ImageFolder):
# C:\Users\AERO\Downloads\Plant Thingy\dataset\
#   SPIDER PLANT\*.jpg
#   TI PLANT\*.jpg
#   JADE PLANT\*.jpg
#   SNAKE PLANT\*.jpg
#   PANDAKAKI\*.jpg
#
# Run example:
#   python train_efficientnetb0_plants.py --data "C:\Users\AERO\Downloads\dataset" --epochs 25 --batch 32

import argparse
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import features as pil_features


@dataclass
class TrainConfig:
    data_dir: str
    out_dir: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    val_split: float
    img_size: int
    num_workers: int
    seed: int
    amp: bool


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", dest="data_dir", default=r"C:\Users\AERO\Downloads\Plant Thingy\dataset", help="Dataset root folder")
    p.add_argument("--out", dest="out_dir", default="runs_plants_efficientnetb0", help="Output folder")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", dest="batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", dest="weight_decay", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--workers", dest="num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    args = p.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        img_size=args.img_size,
        num_workers=args.num_workers,
        seed=args.seed,
        amp=not args.no_amp,
    )

    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    # Use ImageNet normalization from official weights
    weights = EfficientNet_B0_Weights.DEFAULT
    mean = weights.transforms().mean
    std = weights.transforms().std

    train_tfms = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Load full dataset once, then create separate datasets with different transforms
    IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    full_for_split = datasets.ImageFolder(
    root=cfg.data_dir,
    is_valid_file=lambda p: p.lower().endswith(IMG_EXTENSIONS),
    )

    num_classes = len(full_for_split.classes)
    print(f"Found {len(full_for_split)} images in {num_classes} classes:")
    for i, c in enumerate(full_for_split.classes):
        print(f"  {i}: {c}")

    val_len = int(len(full_for_split) * cfg.val_split)
    train_len = len(full_for_split) - val_len
    gen = torch.Generator().manual_seed(cfg.seed)
    train_subset, val_subset = random_split(full_for_split, [train_len, val_len], generator=gen)

    # Assign transforms by wrapping subsets with a custom dataset view
    class SubsetWithTransform(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
            self.classes = subset.dataset.classes  # for reference

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            x, y = self.subset[idx]
            # subset returns PIL image only if underlying dataset has no transform
            # but ImageFolder returns PIL image by default, so this is safe
            if self.transform:
                x = self.transform(x)
            return x, y

    train_ds = SubsetWithTransform(train_subset, train_tfms)
    val_ds = SubsetWithTransform(val_subset, val_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )

    # Model
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    # Loss + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    scaler = torch.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    best_val_acc = 0.0
    best_path = os.path.join(cfg.out_dir, "best_efficientnetb0_plants.pt")
    last_path = os.path.join(cfg.out_dir, "last_efficientnetb0_plants.pt")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        # ---- train ----
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        n_train = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(cfg.amp and device.type == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = xb.size(0)
            train_loss += loss.item() * bs
            train_acc += accuracy_top1(logits.detach(), yb) * bs
            n_train += bs

        train_loss /= max(1, n_train)
        train_acc /= max(1, n_train)

        # ---- val ----
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        n_val = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(cfg.amp and device.type == "cuda")):
                    logits = model(xb)
                    loss = criterion(logits, yb)

                bs = xb.size(0)
                val_loss += loss.item() * bs
                val_acc += accuracy_top1(logits, yb) * bs
                n_val += bs

        val_loss /= max(1, n_val)
        val_acc /= max(1, n_val)

        scheduler.step()

        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"lr {lr_now:.2e} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"{dt:.1f}s"
        )

        # Save last every epoch
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "classes": full_for_split.classes,
                "img_size": cfg.img_size,
                "mean": list(mean),
                "std": list(std),
            },
            last_path,
        )

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "classes": full_for_split.classes,
                    "img_size": cfg.img_size,
                    "mean": list(mean),
                    "std": list(std),
                },
                best_path,
            )
            print(f"  Saved new best: {best_path} (val acc {best_val_acc:.4f})")

    print(f"Done. Best val acc: {best_val_acc:.4f}")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
