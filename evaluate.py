import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.mechrtrics import classification_report, confusion_matrix
import numpy as np

# -----------------------------
# 1. Load trained model
# -----------------------------
model = load_model("models/pneumonia_model.h5")
print("✅ Model loaded successfully!")

# -----------------------------
# 2. Load test data
# -----------------------------
test_dir = "data/chest_xray/test"
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode="binary",
    shuffle=False
)

# -----------------------------
# 3. Evaluate model
# -----------------------------
loss, acc = model.evaluate(test_generator)
print(f"✅ Test Accuracy: {acc*100:.2f}%")

# -----------------------------
# 4. Predictions & Report
# -----------------------------
y_pred = model.predict(test_generator)
y_pred_classes = (y_pred > 0.5).astype("int32")

print("\n📊 Classification Report:")
print(classification_report(test_generator.classes, y_pred_classes, target_names=["NORMAL", "PNEUMONIA"]))

print("\n🧩 Confusion Matrix:")
print(confusion_matrix(test_generator.classes, y_pred_classes))
