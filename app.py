from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import os
import secrets
import numpy as np
import json
from PIL import Image
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = "secret_key"

# Config folders
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["STATIC_FOLDER"] = "static"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(os.path.join(app.config["STATIC_FOLDER"], "images"), exist_ok=True)

model = None
class_names = None


# Load model
def load_model_file():
    global model, class_names

    model_path = os.path.join("model", "plant_disease_model.h5")
    label_path = os.path.join("model", "class_labels.json")

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    if not os.path.exists(label_path):
        print(f"Error: Label file not found at {label_path}")
        return

    model = load_model(model_path)

    with open(label_path) as f:
        labels = json.load(f)

    # Convert keys to a list, assuming labels is a dict where keys are class names
    class_names = list(labels.keys())

    print("Model loaded successfully")


# Image prediction   
def predict_image(image_path):

    img = Image.open(image_path).convert("RGB")
    # Resize to model input size
    img = img.resize((224, 224))

    # Convert to numpy array
    img_array = np.array(img, dtype=np.float32)

    # Normalize pixel values
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)

    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))

    return class_names[predicted_class_idx], confidence


# Home page (Landing)
@app.route("/")
def index():
    return render_template("index.html")


# Upload page
@app.route("/upload")
def upload():
    return render_template("upload.html")


# About page
@app.route("/about")
def about():
    return render_template("about.html")


# Predict route
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if file:

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        file.save(filepath)

        try:
            prediction, confidence = predict_image(filepath)

            # Save image to static folder
            static_filename = f"upload_{secrets.token_hex(8)}.jpg"
            static_path = os.path.join(
                app.config["STATIC_FOLDER"], "images", static_filename
            )

            # Convert to RGB before saving as JPEG to avoid RGBA error
            rgba_image = Image.open(filepath)
            rgb_image = rgba_image.convert("RGB")
            rgb_image.save(static_path, "JPEG")

            # Store result in session
            session["prediction"] = prediction
            session["confidence"] = round(confidence * 100, 2)
            session["image_path"] = f"images/{static_filename}"

            os.remove(filepath)

            return jsonify({"success": True})

        except Exception as e:

            if os.path.exists(filepath):
                os.remove(filepath)

            return jsonify({"error": str(e)}), 500


# Result page
@app.route("/result")
def result():

    prediction = session.get("prediction")
    confidence = session.get("confidence")
    image_path = session.get("image_path")

    is_healthy = False
    if prediction and "healthy" in prediction.lower():
        is_healthy = True

    if not prediction:
        return redirect(url_for("upload"))

    return render_template(
        "result.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path,
        is_healthy=is_healthy
    )


# Run server
if __name__ == "__main__":

    load_model_file()

    app.run(debug=True)