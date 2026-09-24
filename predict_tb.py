from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import sys

# Load your trained TB model
model = load_model(r"models\tb_model.h5")

# Provide image path from command line or manually
img_path = sys.argv[1] if len(sys.argv) > 1 else input("Enter image path: ")

# Preprocess the image
img = image.load_img(img_path, target_size=(224, 224))  # ✅ use same size as training
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x /= 255.0

# Predict
prediction = model.predict(x)
if prediction[0][0] > 0.5:
    print("Tuberculosis Detected")
else:
    print("Normal")
