import os
from typing import List, Tuple

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0


# Absolute path to the trained EfficientNet-B0 checkpoint
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "runs_plants_efficientnetb0", "best_efficientnetb0_plants.pt")

# Globals initialized lazily on first use
_TORCH_MODEL = None
_CLASS_NAMES: List[str] = []
_NORM_MEAN: List[float] = [0.485, 0.456, 0.406]  # sensible defaults (ImageNet) if checkpoint missing stats
_NORM_STD: List[float] = [0.229, 0.224, 0.225]

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_model_loaded() -> bool:
    """
    Lazily load the EfficientNet-B0 model and associated metadata
    (class names, normalization statistics) from the checkpoint.

    Returns True if the model is ready, False otherwise.
    """
    global _TORCH_MODEL, _CLASS_NAMES, _NORM_MEAN, _NORM_STD

    if _TORCH_MODEL is not None:
        return True

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"⚠ Torch checkpoint not found at {CHECKPOINT_PATH}")
        return False

    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

        # Common patterns: full dict with metadata + state dict, or a plain state dict
        state_dict = None
        if isinstance(checkpoint, dict):
            # Try typical keys first
            state_dict = (checkpoint.get("state_dict")
                        or checkpoint.get("model_state_dict")
                        or checkpoint.get("model_state") )

            # Classes
            classes = checkpoint.get("class_names") or checkpoint.get("classes")
            if isinstance(classes, (list, tuple)):
                _CLASS_NAMES = list(classes)

            # Normalization statistics
            mean = (
                checkpoint.get("mean")
                or checkpoint.get("normalize_mean")
                or checkpoint.get("norm_mean")
            )
            std = (
                checkpoint.get("std")
                or checkpoint.get("normalize_std")
                or checkpoint.get("norm_std")
            )
            if isinstance(mean, (list, tuple)) and len(mean) == 3:
                _NORM_MEAN = [float(m) for m in mean]
            if isinstance(std, (list, tuple)) and len(std) == 3:
                _NORM_STD = [float(s) for s in std]

            # Optional metadata for number of classes
            num_classes = checkpoint.get("num_classes")
        else:
            # If it's not a dict, assume it's already a state dict
            state_dict = checkpoint
            num_classes = None

        if state_dict is None:
            # Fallback: maybe the dict itself is a state dict
            if isinstance(checkpoint, dict):
                state_dict = checkpoint

        if state_dict is None:
            print("⚠ Torch checkpoint did not contain a recognizable state dict.")
            return False

        # Determine number of classes for the classifier head
        if _CLASS_NAMES:
            num_classes = len(_CLASS_NAMES)
        elif not num_classes:
            # Last-resort fallback if metadata is missing
            num_classes = 5

        model = efficientnet_b0(weights=None, num_classes=num_classes)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.to(_DEVICE)

        _TORCH_MODEL = model

        print(f"✓ Loaded EfficientNet-B0 checkpoint from {CHECKPOINT_PATH}")
        print(f"  Using device: {_DEVICE}")
        print(f"  Num classes: {num_classes}")
        if _CLASS_NAMES:
            print(f"  Class names (from checkpoint): {_CLASS_NAMES}")
        else:
            print("  ⚠ No class_names found in checkpoint; falling back to index-based labels.")
        print(f"  Normalization mean: {_NORM_MEAN}, std: {_NORM_STD}")

        return True

    except Exception as e:
        print(f"⚠ Error loading torch model from {CHECKPOINT_PATH}: {e}")
        _TORCH_MODEL = None
        return False


def _build_transform() -> transforms.Compose:
    """
    Build the preprocessing transform using the mean/std from the checkpoint
    (or sensible defaults if not present).
    """
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=_NORM_MEAN, std=_NORM_STD),
        ]
    )


def predict_image(image_path: str) -> Tuple[str, float, List[Tuple[str, float]]]:
    """
    Run inference on a single image.

    Returns:
        (label, confidence, all_probs)

        - label: predicted class label (string) or None if prediction failed
        - confidence: probability of the predicted class (0.0–1.0)
        - all_probs: list of (label, probability) for all classes, sorted desc
    """
    if not _ensure_model_loaded():
        return None, 0.0, []

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image for torch inference: {e}")
        return None, 0.0, []

    transform = _build_transform()
    tensor = transform(img).unsqueeze(0).to(_DEVICE)

    try:
        with torch.no_grad():
            outputs = _TORCH_MODEL(tensor)
            probs_tensor = torch.softmax(outputs, dim=1)[0].cpu()

        probs = probs_tensor.numpy()
        idx = int(probs.argmax())
        confidence = float(probs[idx])

        # Build labels from class names where available, otherwise indices
        def _label_for_index(i: int) -> str:
            if _CLASS_NAMES and 0 <= i < len(_CLASS_NAMES):
                return str(_CLASS_NAMES[i])
            return str(i)

        label = _label_for_index(idx)

        all_probs: List[Tuple[str, float]] = [
            (_label_for_index(i), float(probs[i])) for i in range(len(probs))
        ]
        all_probs.sort(key=lambda x: -x[1])

        # Lightweight fingerprint for debugging input variability
        print(
            f"Torch model prediction: label={label}, confidence={confidence:.4f}, "
            f"mean={float(tensor.mean()):.4f}, std={float(tensor.std()):.4f}"
        )

        return label, confidence, all_probs

    except Exception as e:
        print(f"Error during torch model prediction: {e}")
        return None, 0.0, []

