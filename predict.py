# predict.py
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import sys
import os

# -----------------------------
# 1. Load trained model
# -----------------------------
MODEL_PATH = "models/pneumonia_model.h5"
if not os.path.exists(MODEL_PATH):
    print("❌ Model file not found!")
    sys.exit(1)

model = load_model(MODEL_PATH)

# -----------------------------
# 2. Preprocess image
# -----------------------------
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # 224 since DenseNet was used
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# -----------------------------
# 3. Predict function
# -----------------------------
def predict(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array, verbose=0)[0][0]

    if prediction > 0.5:
        print("Pneumonia Detected")
    else:
        print("Normal")

# -----------------------------
# 4. Run from command line
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
    else:
        img_path = sys.argv[1]
        if not os.path.exists(img_path):
            print("❌ Image not found:", img_path)
            sys.exit(1)
        predict(img_path)
