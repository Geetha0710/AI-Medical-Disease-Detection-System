from flask import Flask, render_template, request
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import re
from difflib import get_close_matches

app = Flask(__name__)

# ----------------- Model Paths -----------------
PNEUMONIA_MODEL_PATH = r"D:\medical_ai_miniproject\models\pneumonia_model.h5"
TB_MODEL_PATH = r"D:\medical_ai_miniproject\models\tb_model.h5"

# ----------------- Load Models -----------------
try:
    pneumonia_model = tf.keras.models.load_model(PNEUMONIA_MODEL_PATH)
    print("✅ Pneumonia model loaded successfully.")
except Exception as e:
    pneumonia_model = None
    print(f"⚠ Error loading pneumonia model: {e}")

try:
    tb_model = tf.keras.models.load_model(TB_MODEL_PATH)
    print("✅ Tuberculosis model loaded successfully.")
except Exception as e:
    tb_model = None
    print(f"⚠ Error loading TB model: {e}")

# ----------------- Folder Setup -----------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ----------------- Spell Correction for Symptoms -----------------
valid_symptoms = [
    "cough", "fever", "chest pain", "persistent cough", "weight loss", "night sweats",
    "dry cough", "loss of smell", "shortness of breath", "fatigue"
]

def correct_symptom(word):
    match = get_close_matches(word.lower(), valid_symptoms, n=1, cutoff=0.7)
    return match[0] if match else word

# ----------------- Validate Medical Image -----------------
def is_valid_medical_image(img_array):
    """
    Accepts medical images (X-ray, MRI) and rejects highly colorful non-medical images.
    """
    if img_array.shape[-1] == 3:
        gray = np.mean(img_array, axis=2)
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        mean_diff = np.mean(np.abs(r - g) + np.abs(r - b) + np.abs(g - b))
        if mean_diff > 40:  # Too colorful, likely not X-ray/MRI
            return False
    else:
        gray = img_array

    var = np.var(gray)
    if var < 10:  # completely blank
        return False

    return True

# ----------------- Predict from Image -----------------
def predict_image(file_path):
    try:
        img = image.load_img(file_path, target_size=(224, 224), color_mode='rgb')
        x = image.img_to_array(img)
    except:
        return "Invalid medical image"

    if not is_valid_medical_image(x):
        return "Invalid medical image"

    x = np.expand_dims(x, axis=0) / 255.0

    tb_pred = tb_model.predict(x)[0][0] if tb_model else None
    pneumonia_pred = pneumonia_model.predict(x)[0][0] if pneumonia_model else None

    if tb_pred is not None and tb_pred > 0.5:
        return "Tuberculosis"
    elif pneumonia_pred is not None and pneumonia_pred > 0.5:
        return "Pneumonia"
    else:
        return "Normal"

# ----------------- Symptom Prediction -----------------
symptom_map = {
    "Pneumonia": ["cough", "fever", "chest pain"],
    "Tuberculosis": ["persistent cough", "weight loss", "night sweats"],
    "Covid": ["fever", "dry cough", "loss of smell"],
    "Lung Cancer": ["shortness of breath", "fatigue"]
}

symptom_cures = {
    "cough": ["Drink warm fluids", "Use a humidifier"],
    "fever": ["Stay hydrated", "Take paracetamol"],
    "chest pain": ["See a doctor immediately"],
    "persistent cough": ["Consult a doctor"],
    "weight loss": ["Balanced diet"],
    "night sweats": ["Keep room cool"],
    "dry cough": ["Honey", "Steam inhalation"],
    "loss of smell": ["Consult ENT specialist"],
    "shortness of breath": ["Rest"],
    "fatigue": ["Rest and nutritious diet"]
}

def is_valid_symptom_input(symptom_text):
    return bool(re.match(r'^[A-Za-z,\s]+$', symptom_text))

def predict_from_symptoms(symptoms_list):
    symptoms_list = [correct_symptom(s) for s in symptoms_list]
    symptoms_set = set(symptoms_list)

    best_match = None
    max_matches = 0
    for disease, disease_symptoms in symptom_map.items():
        matches = len(symptoms_set.intersection(set(disease_symptoms)))
        if matches > max_matches:
            max_matches = matches
            best_match = disease

    disease_result = best_match if max_matches >= 2 else "No major disease detected"

    cure_suggestions = []
    for s in symptoms_set:
        if s in symptom_cures:
            cure_suggestions.extend(symptom_cures[s])

    if disease_result.lower() == "pneumonia":
        cure_suggestions = [
            "Take prescribed antibiotics",
            "Rest and stay hydrated",
            "Avoid smoking and polluted air"
        ]
    elif disease_result.lower() == "tuberculosis":
        cure_suggestions = [
            "Consult a pulmonologist immediately",
            "Take anti-TB medicines regularly",
            "Avoid close contact with others"
        ]

    return disease_result, cure_suggestions

# ----------------- Karnataka Locations -----------------
locations = [
    "Bengaluru", "Mysuru", "Ballari", "Hubbali", "Mangaluru",
    "Belagavi", "Kalaburagi", "Bidar", "Hassan", "Shimoga",
    "Chitradurga", "Dharwad", "Raichur", "Kolar", "Bangalore Rural",
    "Tumakuru", "Chikmagalur", "Bagalkot"
]

# ----------------- Load Doctors -----------------
def load_doctor_suggestions(file="doctors.txt"):
    doctor_map = {}
    try:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 4:
                    disease, name, hospital, location = parts
                    doctor_map.setdefault(disease.strip(), []).append({
                        "name": name.strip(),
                        "hospital": hospital.strip(),
                        "location": location.strip()
                    })
    except:
        pass
    return doctor_map

# ----------------- Routes -----------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", locations=locations)

@app.route("/predict", methods=["POST"])
def predict():
    input_type = request.form.get("input_type")
    patient_location = request.form.get("location")
    cure = []

    if input_type == "xray":
        f = request.files.get("file")
        if not f or f.filename == "":
            return render_template("index.html", result="No file uploaded", locations=locations)

        file_path = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(file_path)

        disease = predict_image(file_path)
        if disease == "Invalid medical image":
            return render_template("index.html", result="Invalid medical image. Please upload a valid X-ray/MRI.", locations=locations)

    else:
        symptoms_str = request.form.get("symptoms", "")
        if not symptoms_str.strip():
            return render_template("index.html", result="No symptoms entered", locations=locations)

        if not is_valid_symptom_input(symptoms_str):
            return render_template("index.html", result="Invalid symptoms. Use letters only.", locations=locations)

        selected_symptoms = [s.strip() for s in symptoms_str.split(",") if s.strip()]
        disease, cure = predict_from_symptoms(selected_symptoms)

    doctor_map = load_doctor_suggestions("doctors.txt")
    doctors = doctor_map.get(disease, [])
    location_doctors = [d for d in doctors if d["location"].lower() == patient_location.lower()]

    return render_template("index.html",
                           result=disease,
                           doctors=location_doctors,
                           cure=cure,
                           locations=locations,
                           selected_location=patient_location)

# ----------------- Run -----------------
if __name__ == "__main__":
    app.run(debug=True)
