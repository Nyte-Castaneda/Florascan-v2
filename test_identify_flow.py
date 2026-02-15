#!/usr/bin/env python3
"""Test identify flow offline: load model, predict on an image, compute response (invalid_result, alternatives)."""
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS = os.path.join(PROJECT_DIR, 'uploads')
# Test multiple images to see if model gets different inputs but same output
TEST_IMAGES = [
    '20260207_184832_Spider-plant.jpg',
    '20260202_234122_Pandakaki_Puti_11.JPG',
    '20260207_194135_jade-plant-in-glass-bowl_2.webp',
]

# Must match app.py
CLASS_LABELS = ['JADE PLANT', 'PANDAKAKI', 'SNEK PLANT', 'SPIDER PLANT', 'TI PLANT']
MODEL_TO_DB_NAME = {
    'JADE PLANT': 'Jade Plant', 'PANDAKAKI': 'Pandakaki', 'SNEK PLANT': 'Snake Plant',
    'SPIDER PLANT': 'Spider Plant', 'TI PLANT': 'Ti Plant',
}
INVALID_CONFIDENCE_THRESHOLD = 0.35

def main():
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess
    import numpy as np
    from PIL import Image

    model_path = os.path.join(PROJECT_DIR, 'plant_model_extracted')
    if not os.path.exists(model_path):
        model_path = os.path.join(PROJECT_DIR, 'plant_model.keras')
    print("Loading model...")
    model = load_model(model_path)
    print("  Loaded from", model_path, "\n")

    for name in TEST_IMAGES:
        path = os.path.join(UPLOADS, name)
        if not os.path.isfile(path):
            print(f"Skip (not found): {name}\n")
            continue
        img = Image.open(path).convert('RGB')
        img = img.resize((224, 224))
        arr = np.array(img)
        arr = mobilenet_v2_preprocess(arr)
        x = np.expand_dims(arr, axis=0).astype('float32')
        inp_mean, inp_std = float(np.mean(x)), float(np.std(x))
        preds = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(preds))
        conf = float(preds[idx])
        label = CLASS_LABELS[idx]
        main_name = MODEL_TO_DB_NAME.get(label, label)
        print(f"Image: {name[:45]}")
        print(f"  Input fingerprint: mean={inp_mean:.4f} std={inp_std:.4f}")
        print(f"  Prediction: {main_name} ({conf:.1%})")
        print(f"  All probs: JADE={preds[0]:.2f} PANDAKAKI={preds[1]:.2f} SNEK={preds[2]:.2f} SPIDER={preds[3]:.2f} TI={preds[4]:.2f}")
        print()

    print("---")
    print("If 'Input fingerprint' (mean/std) differs per image but 'Prediction' is always the same,")
    print("the model is not distinguishing between plants — retrain with more/balanced images per class.")

if __name__ == "__main__":
    main()
