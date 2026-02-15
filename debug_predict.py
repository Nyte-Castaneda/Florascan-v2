import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "plant_model.keras"  # same file you load in app.py
model = load_model(MODEL_PATH)

def show(name, x):
    p = model.predict(x, verbose=0)[0]
    print(name, "->", np.round(p, 3), "sum=", float(np.sum(p)))

# 3 extreme inputs
x_black = np.zeros((1,224,224,3), dtype=np.float32)        # all zeros
x_white = np.ones((1,224,224,3), dtype=np.float32) * 255.0 # all 255
x_noise = np.random.rand(1,224,224,3).astype(np.float32)*255.0

# If your training used MobileNetV2 preprocess_input, apply it here:
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
x_black = preprocess_input(x_black)
x_white = preprocess_input(x_white)
x_noise = preprocess_input(x_noise)

show("BLACK", x_black)
show("WHITE", x_white)
show("NOISE", x_noise)
