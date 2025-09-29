import os
import logging
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# Setup
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Constants
MODEL_PATH = "models/eye_disease_model.h5"
UPLOAD_FOLDER = "static/uploads"
CATEGORIES = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]

# Config
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
logging.info(f"Loading model from {MODEL_PATH}...")
model = load_model(MODEL_PATH)
logging.info("Model loaded successfully.")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save the file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Process the image
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict using inference mode (Dropout OFF)
    prediction = model(img_array, training=False).numpy()[0]
    class_index = np.argmax(prediction)
    predicted_class = CATEGORIES[class_index]
    confidence = round(float(prediction[class_index]) * 100, 2)

    # Debug logs (optional)
    logging.info(f"Prediction array: {prediction}")
    logging.info(f"Predicted: {predicted_class} with {confidence}% confidence")

    return jsonify({
        "class": predicted_class,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)
