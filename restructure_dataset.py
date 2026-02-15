import os, shutil, random

RAW_ROOT = "raw_dataset"      # your messy folders
OUT_ROOT = "dataset"          # clean output
VAL_RATIO = 0.2
SEED = 42

random.seed(SEED)

CLASS_RULES = {
    "JADE PLANT": ["jade"],
    "PANDAKAKI": ["pandakaki"],
    "SNAKE PLANT": ["snake", "snek"],
    "SPIDER PLANT": ["spider"],
    "TI PLANT": ["ti plant", "ti_plant", "ti-plant", "ti "],
}

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")

def detect_class(folder_name: str):
    name = folder_name.lower()
    for cls, keys in CLASS_RULES.items():
        for k in keys:
            if k in name:
                return cls
    return None

def collect_images(folder_path):
    out = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith(IMG_EXT):
                out.append(os.path.join(root, f))
    return out

def safe_copy(src, dst_dir, prefix):
    os.makedirs(dst_dir, exist_ok=True)
    base = os.path.basename(src)
    dst = os.path.join(dst_dir, f"{prefix}_{base}")
    i = 1
    while os.path.exists(dst):
        dst = os.path.join(dst_dir, f"{prefix}_{i}_{base}")
        i += 1
    shutil.copy2(src, dst)

# Create output folders
classes = list(CLASS_RULES.keys())
for split in ["train", "val"]:
    for cls in classes + ["UNSORTED"]:
        os.makedirs(os.path.join(OUT_ROOT, split, cls), exist_ok=True)

folders = [d for d in os.listdir(RAW_ROOT) if os.path.isdir(os.path.join(RAW_ROOT, d))]

total = 0
for folder in folders:
    cls = detect_class(folder)
    src_folder = os.path.join(RAW_ROOT, folder)
    imgs = collect_images(src_folder)
    if not imgs:
        continue

    random.shuffle(imgs)
    cut = int(len(imgs) * (1 - VAL_RATIO))
    train_imgs = imgs[:cut]
    val_imgs = imgs[cut:]

    out_cls = cls if cls else "UNSORTED"
    for p in train_imgs:
        safe_copy(p, os.path.join(OUT_ROOT, "train", out_cls), folder)
    for p in val_imgs:
        safe_copy(p, os.path.join(OUT_ROOT, "val", out_cls), folder)

    print(out_cls, "<-", folder, "train:", len(train_imgs), "val:", len(val_imgs))
    total += len(imgs)

print("\nTotal images processed:", total)
print("Output created at:", OUT_ROOT)
print("If you see UNSORTED folders, check their names/spelling.\n")
