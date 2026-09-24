import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# -----------------------------
# 1. Paths
# -----------------------------
train_dir = "data/chest_xray/train"
val_dir = "data/chest_xray/val"

# -----------------------------
# 2. Data Generators
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode="binary"
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode="binary"
)

# -----------------------------
# 3. Model (DenseNet121)
# -----------------------------
base_model = DenseNet121(weights="imagenet", include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
predictions = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model layers for transfer learning
for layer in base_model.layers:
    layer.trainable = False

# -----------------------------
# 4. Compile
# -----------------------------
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# 5. Train
# -----------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=2  # 🔹 use 2 for quick test, increase later (10–20 for better accuracy)
)

# -----------------------------
# 6. Save model
# -----------------------------
os.makedirs("models", exist_ok=True)
model.save("models/pneumonia_model.h5")
print("✅ Model saved at models/pneumonia_model.h5")
