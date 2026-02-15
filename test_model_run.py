#!/usr/bin/env python3
"""Quick test: load model and run prediction on a few uploads to verify it works."""
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS = os.path.join(PROJECT_DIR, 'uploads')

# Same as app.py
CLASS_LABELS = ['JADE PLANT', 'PANDAKAKI', 'SNEK PLANT', 'SPIDER PLANT', 'TI PLANT']
MODEL_CANDIDATES = [
    os.path.join(PROJECT_DIR, 'plant_model_extracted'),
    os.path.join(PROJECT_DIR, 'plant_model.keras'),
    os.path.join(PROJECT_DIR, 'plant_model.h5'),
]

def main():
    print("Loading model...")
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess
    import numpy as np
    from PIL import Image

    plant_model = None
    for path in MODEL_CANDIDATES:
        if not os.path.exists(path):
            continue
        try:
            plant_model = load_model(path)
            print(f"  Loaded from: {path}")
            break
        except Exception as e:
            print(f"  Skip {path}: {e}")
    if plant_model is None:
        print("ERROR: Could not load model.")
        sys.exit(1)

    def preprocess(image_path):
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        arr = np.array(img)
        arr = mobilenet_v2_preprocess(arr)
        return np.expand_dims(arr, axis=0).astype('float32')

    # Test on a few images
    test_files = [
        '20260207_184832_Spider-plant.jpg',
        '20260207_194135_jade-plant-in-glass-bowl_2.webp',
        '20260202_234122_Pandakaki_Puti_11.JPG',
    ]
    for name in test_files:
        path = os.path.join(UPLOADS, name)
        if not os.path.isfile(path):
            print(f"  (skip {name} - not found)")
            continue
        x = preprocess(path)
        preds = plant_model.predict(x, verbose=0)[0]
        idx = int(np.argmax(preds))
        label = CLASS_LABELS[idx]
        conf = float(preds[idx])
        print(f"  {name[:40]:40} -> {label:15} ({conf:.2%})")
    print("Done. Model and prediction pipeline work.")

if __name__ == '__main__':
    main()
