#!/usr/bin/env python3
"""
Inspect plant_model.h5: input shape, output classes, and any saved class names.
Run from project folder: python inspect_model.py
"""
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'plant_model.h5')

if not os.path.exists(MODEL_PATH):
    print(f"Model not found: {MODEL_PATH}")
    exit(1)

print("Loading model (this may take a moment)...")
from tensorflow.keras.models import load_model

model = load_model(MODEL_PATH)

# Input shape (height, width, channels)
if hasattr(model, 'input_shape'):
    print("\n--- Input shape ---")
    print(model.input_shape)
if hasattr(model, 'input'):
    inp = model.input
    if hasattr(inp, 'shape'):
        print("\n--- Input shape ---")
        print(inp.shape)

# Output: number of classes
if hasattr(model, 'output_shape'):
    print("\n--- Output shape ---")
    print(model.output_shape)
out = model.output
if hasattr(out, 'shape'):
    num_classes = out.shape[-1]
    print(f"\n--- Number of classes: {num_classes} ---")

# Check for saved class names (some training scripts save them)
class_names = None
if hasattr(model, 'class_names'):
    class_names = model.class_names
elif hasattr(model, 'config') and isinstance(model.config, dict):
    class_names = model.config.get('class_names')
if class_names is not None:
    print("\n--- Class names (order = model output index) ---")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")
else:
    print("\n--- No class names stored in model ---")
    print("You need to set CLASS_LABELS in app.py to match the order used when training.")
    print("Common order: alphabetical by folder name (e.g. flow_from_directory).")

print("\n--- Model summary ---")
model.summary()
