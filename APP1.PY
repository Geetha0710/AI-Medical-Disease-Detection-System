from flask import Flask, render_template, request
import os
import subprocess

app = Flask(__name__, template_folder="data/chest_xray/templates")

# Upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ----------------- Doctor Loader -----------------
def load_doctor_suggestions(file="doctors.txt"):
    doctor_map = {}
    try:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 4:
                    disease, name, hospital, location = parts
                    disease = disease.strip()
                    if disease not in doctor_map:
                        doctor_map[disease] = []
                    doctor_map[disease].append({
                        "name": name.strip(),
                        "hospital": hospital.strip(),
                        "location": location.strip()
                    })
    except FileNotFoundError:
        print("⚠️ doctors.txt not found! Place it in the same folder as app.py")
    return doctor_map

# ----------------- Predict Disease from Image -----------------
def predict_disease(file_path):
    result = subprocess.run(
        ["python", "predict.py", file_path],
        capture_output=True,
        text=True
    )
    output = result.stdout.strip()
    disease = output.replace("Detected", "").replace("detected", "").strip()
    return disease

# ----------------- Predict Disease from Symptoms -----------------
symptom_map = {
    "Pneumonia": ["cough", "fever", "chest pain"],
    "Tuberculosis": ["persistent cough", "weight loss", "night sweats"],
    "Covid": ["fever", "dry cough", "loss of smell"],
    "Lung Cancer": ["shortness of breath", "fatigue"]
}

# Symptom-specific cures
symptom_cures = {
    "cough": ["Drink warm fluids", "Use a humidifier", "Avoid smoking"],
    "fever": ["Stay hydrated", "Rest", "Take paracetamol if needed"],
    "chest pain": ["See a doctor immediately", "Avoid strenuous activity"],
    "persistent cough": ["Consult a doctor", "Follow prescribed medication"],
    "weight loss": ["Balanced diet", "Consult a nutritionist"],
    "night sweats": ["Keep room cool", "Wear breathable clothing"],
    "dry cough": ["Honey", "Steam inhalation"],
    "loss of smell": ["Consult ENT specialist"],
    "shortness of breath": ["Rest", "Avoid heavy exertion"],
    "fatigue": ["Rest", "Balanced diet"]
}

def predict_from_symptoms(symptoms_list):
    symptoms_set = set([s.lower() for s in symptoms_list])
    best_match = None
    max_matches = 0

    # Find disease with max symptom matches
    for disease, disease_symptoms in symptom_map.items():
        matches = len(symptoms_set.intersection(set(disease_symptoms)))
        if matches > max_matches:
            max_matches = matches
            best_match = disease

    # Only assign disease if 2 or more symptoms match
    disease_result = best_match if max_matches >= 2 else "No major disease detected"

    # Generate cure suggestions for entered symptoms
    cure_suggestions = []
    for symptom in symptoms_set:
        if symptom in symptom_cures:
            cure_suggestions.extend(symptom_cures[symptom])

    # If Pneumonia is detected as a major disease, override with full Pneumonia cure
    if disease_result.lower() == "pneumonia":
        cure_suggestions = [
            "Take prescribed antibiotics",
            "Rest and stay hydrated",
            "Avoid smoking and polluted air",
            "Use a humidifier to ease breathing",
            "Follow up with your doctor regularly"
        ]

    return disease_result, cure_suggestions

# ----------------- Karnataka Districts -----------------
locations = ["Bengaluru", "Mysuru", "Ballari","Hubbali", "Mangaluru", "Belagavi", "Kalaburagi",
             "Bidar", "Hassan", "Shimoga", "Chitradurga", "Dharwad", "Raichur",
             "Kolar", "Bangalore Rural", "Tumakuru", "Chikmagalur", "Bagalkot"]

# ----------------- Routes -----------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", locations=locations)

@app.route("/predict", methods=["POST"])
def predict():
    input_type = request.form.get("input_type")
    patient_location = request.form.get("location")
    cure = []

    # ----------------- Determine Disease -----------------
    if input_type == "xray":
        f = request.files.get("file")
        if not f or f.filename == "":
            return render_template("index.html", result="No file uploaded", locations=locations)
        file_path = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(file_path)
        disease = predict_disease(file_path)

        # Provide cure if disease is Pneumonia
        if disease.lower() == "pneumonia":
            cure = [
                "Take prescribed antibiotics",
                "Rest and stay hydrated",
                "Avoid smoking and polluted air",
                "Use a humidifier to ease breathing",
                "Follow up with your doctor regularly"
            ]
    else:  # symptom input
        symptoms_str = request.form.get("symptoms", "")
        if not symptoms_str.strip():
            return render_template("index.html", result="No symptoms entered", locations=locations)
        selected_symptoms = [s.strip() for s in symptoms_str.split(",") if s.strip()]
        disease, cure = predict_from_symptoms(selected_symptoms)

    # ----------------- Fetch Doctor Suggestions -----------------
    doctor_map = load_doctor_suggestions("doctors.txt")
    doctors = doctor_map.get(disease, [])
    location_doctors = [d for d in doctors if d["location"] == patient_location]

    return render_template(
        "index.html",
        result=disease,
        doctors=location_doctors,
        locations=locations,
        selected_location=patient_location,
        cure=cure
    )

# ----------------- Run App -----------------
if __name__ == "__main__":
    app.run(debug=True)











from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__, template_folder="data/chest_xray/templates")

# Upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ----------------- Load Models -----------------
tb_model = load_model(r"models\tb_model.h5")
pneumonia_model = load_model(r"models\pneumonia_model.h5")

# ----------------- Doctor Loader -----------------
def load_doctor_suggestions(file="doctors.txt"):
    doctor_map = {}
    try:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 4:
                    disease, name, hospital, location = parts
                    disease = disease.strip()
                    if disease not in doctor_map:
                        doctor_map[disease] = []
                    doctor_map[disease].append({
                        "name": name.strip(),
                        "hospital": hospital.strip(),
                        "location": location.strip()
                    })
    except FileNotFoundError:
        print("⚠️ doctors.txt not found! Place it in the same folder as app.py")
    return doctor_map


# ----------------- Predict Disease from Image -----------------
def predict_from_image(file_path, disease_type):
    img = image.load_img(file_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    if disease_type == "tb":
        model = tb_model
    elif disease_type == "pneumonia":
        model = pneumonia_model
    else:
        return "Unknown disease type"

    prediction = model.predict(x)

    # Model output logic
    if prediction[0][0] > 0.5:
        return "Tuberculosis Detected" if disease_type == "tb" else "Pneumonia Detected"
    else:
        return "Normal (No Disease Detected)"


# ----------------- Symptom Mapping -----------------
symptom_map = {
    "Pneumonia": ["cough", "fever", "chest pain"],
    "Tuberculosis": ["persistent cough", "weight loss", "night sweats"],
    "Covid": ["fever", "dry cough", "loss of smell"],
    "Lung Cancer": ["shortness of breath", "fatigue"]
}

# ----------------- Symptom Cures -----------------
symptom_cures = {
    "cough": ["Drink warm fluids", "Use a humidifier", "Avoid smoking"],
    "fever": ["Stay hydrated", "Rest", "Take paracetamol if needed"],
    "chest pain": ["See a doctor immediately", "Avoid strenuous activity"],
    "persistent cough": ["Consult a doctor", "Follow prescribed medication"],
    "weight loss": ["Balanced diet", "Consult a nutritionist"],
    "night sweats": ["Keep room cool", "Wear breathable clothing"],
    "dry cough": ["Honey", "Steam inhalation"],
    "loss of smell": ["Consult ENT specialist"],
    "shortness of breath": ["Rest", "Avoid heavy exertion"],
    "fatigue": ["Rest", "Balanced diet"]
}

def predict_from_symptoms(symptoms_list):
    symptoms_set = set([s.lower() for s in symptoms_list])
    best_match = None
    max_matches = 0

    for disease, disease_symptoms in symptom_map.items():
        matches = len(symptoms_set.intersection(set(disease_symptoms)))
        if matches > max_matches:
            max_matches = matches
            best_match = disease

    disease_result = best_match if max_matches >= 2 else "No major disease detected"

    cure_suggestions = []
    for symptom in symptoms_set:
        if symptom in symptom_cures:
            cure_suggestions.extend(symptom_cures[symptom])

    if disease_result.lower() == "pneumonia":
        cure_suggestions = [
            "Take prescribed antibiotics",
            "Rest and stay hydrated",
            "Avoid smoking and polluted air",
            "Use a humidifier to ease breathing",
            "Follow up with your doctor regularly"
        ]

    return disease_result, cure_suggestions


# ----------------- Karnataka Districts -----------------
locations = ["Bengaluru", "Mysuru", "Ballari", "Hubbali", "Mangaluru", "Belagavi", "Kalaburagi",
             "Bidar", "Hassan", "Shimoga", "Chitradurga", "Dharwad", "Raichur",
             "Kolar", "Bangalore Rural", "Tumakuru", "Chikmagalur", "Bagalkot"]

# ----------------- Routes -----------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", locations=locations)

@app.route("/predict", methods=["POST"])
def predict():
    input_type = request.form.get("input_type")
    patient_location = request.form.get("location")
    disease_type = request.form.get("disease_type")  # tb or pneumonia
    cure = []

    if input_type == "xray":
        f = request.files.get("file")
        if not f or f.filename == "":
            return render_template("index.html", result="No file uploaded", locations=locations)
        file_path = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(file_path)
        disease = predict_from_image(file_path, disease_type)

        if "Pneumonia" in disease:
            cure = [
                "Take prescribed antibiotics",
                "Rest and stay hydrated",
                "Avoid smoking and polluted air",
                "Use a humidifier to ease breathing",
                "Follow up with your doctor regularly"
            ]
        elif "Tuberculosis" in disease:
            cure = [
                "Consult a pulmonologist immediately",
                "Take anti-TB medications as prescribed",
                "Get proper nutrition and adequate rest",
                "Avoid close contact with others until cleared"
            ]

    else:  # symptom input
        symptoms_str = request.form.get("symptoms", "")
        if not symptoms_str.strip():
            return render_template("index.html", result="No symptoms entered", locations=locations)
        selected_symptoms = [s.strip() for s in symptoms_str.split(",") if s.strip()]
        disease, cure = predict_from_symptoms(selected_symptoms)

    doctor_map = load_doctor_suggestions("doctors.txt")
    doctors = doctor_map.get(disease_type.capitalize(), [])
    location_doctors = [d for d in doctors if d["location"] == patient_location]

    return render_template(
        "index.html",
        result=disease,
        doctors=location_doctors,
        locations=locations,
        selected_location=patient_location,
        cure=cure
    )


if __name__ == "__main__":
    app.run(debug=True)
