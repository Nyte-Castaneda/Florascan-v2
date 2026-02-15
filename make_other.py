import os, random
from PIL import Image, ImageEnhance

SRC_DIR = "raw_other"
OUT_DIR = os.path.join("dataset", "train", "OTHER")
TARGET_COUNT = 220
SIZE = (224, 224)

os.makedirs(OUT_DIR, exist_ok=True)

def aug(img: Image.Image) -> Image.Image:
    img = img.convert("RGB").resize(SIZE)

    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    angle = random.uniform(-15, 15)
    img = img.rotate(angle, expand=False)

    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.3))
    return img

imgs = [f for f in os.listdir(SRC_DIR) if f.lower().endswith((".jpg",".jpeg",".png",".webp"))]
if not imgs:
    raise SystemExit(f"No images found in {SRC_DIR}")

count = 0
while count < TARGET_COUNT:
    fname = random.choice(imgs)
    img = Image.open(os.path.join(SRC_DIR, fname))
    out = aug(img)
    out.save(os.path.join(OUT_DIR, f"other_{count:04d}.jpg"), quality=92)
    count += 1

print("Generated:", count, "images in", OUT_DIR)
