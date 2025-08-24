import os
import base64
import requests
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# ==== Load your model ====
MODEL_PATH = "densenet_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# ==== Flask setup ====
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==== Preprocess ====
def prepare_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0
    return x

# ==== Prediction ====
def predict_local(img_path):
    img_array = prepare_image(img_path)
    preds = model.predict(img_array)[0]
    classes = ["Normal", "Pneumonia"]
    return classes[np.argmax(preds)], float(np.max(preds))

# ==== OpenAI API for infected overlay ====
def get_openai_overlay(img_path):
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    url = "https://api.openai.com/v1/images/edits"

    with open(img_path, "rb") as f:
        files = {"image": f}
        data = {
            "model": "gpt-image-1",
            "prompt": "Highlight the infected pneumonia regions in this lung CT scan using a heatmap-style overlay."
        }
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

        response = requests.post(url, headers=headers, files=files, data=data)
        if response.status_code == 200:
            b64_img = response.json()["data"][0]["b64_json"]
            return b64_img
        else:
            print("OpenAI API error:", response.text)
            return None

# ==== Routes ====
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            img_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(img_path)

            label, conf = predict_local(img_path)
            heatmap_b64 = None

            if label == "Pneumonia":
                heatmap_b64 = get_openai_overlay(img_path)

            return render_template("result.html",
                                   label=label,
                                   confidence=round(conf * 100, 2),
                                   heatmap=heatmap_b64)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
