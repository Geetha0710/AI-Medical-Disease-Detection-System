from tensorflow.keras.models import load_model
import sys
import numpy as np
from tensorflow.keras.preprocessing import image

# -----------------------------
# 1. Load trained model
# -----------------------------
model = load_model("models/pneumonia_model.h5")
print("✅ Model loaded successfully!")
