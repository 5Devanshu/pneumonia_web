import os
import base64
from flask import Flask, request, render_template, abort
import numpy as np
from PIL import Image
import torch
import cv2
import io
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ==== Config ====
HF_CT_MODEL_NAME = "oohtmeel/swin-tiny-patch4-finetuned-lung-cancer-ct-scans"  # public CT classifier
ALLOWED_EXT = {".png"}
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==== Load model ====
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoImageProcessor.from_pretrained(HF_CT_MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(HF_CT_MODEL_NAME).to(device).eval()

# ==== Flask ====
app = Flask(__name__)

def allowed_file(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXT

def pil_to_b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def overlay_heatmap_on_image(pil_img, heatmap_2d: np.ndarray, alpha: float = 0.4):
    # Normalize heatmap 0..1
    hm = heatmap_2d.astype(np.float32)
    hm -= hm.min()
    denom = (hm.max() - hm.min() + 1e-8)
    hm = hm / denom

    # Resize to original
    img_np = np.array(pil_img.convert("RGB"))
    H, W = img_np.shape[:2]
    hm_resized = cv2.resize(hm, (W, H), interpolation=cv2.INTER_LINEAR)

    # Colorize
    heatmap_color = cv2.applyColorMap(np.uint8(hm_resized * 255), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap_color, alpha, 0)
    return Image.fromarray(overlay)

def predict_and_saliency(png_path: str):
    # Load image
    pil_img = Image.open(png_path).convert("RGB")

    # Preprocess
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    # Enable gradients on input tensor for saliency
    inputs["pixel_values"].requires_grad_(True)

    # Forward
    outputs = model(**inputs)
    logits = outputs.logits  # [1, num_classes]
    probs = torch.softmax(logits, dim=-1)[0]
    pred_idx = int(torch.argmax(probs).item())

    # Backprop on predicted logit to get dLogit/dInput
    model.zero_grad(set_to_none=True)
    logits[0, pred_idx].backward()

    # Gradient wrt input pixels
    grads = inputs["pixel_values"].grad.detach().cpu().numpy()[0]  # [C, H, W]
    # Use channel-wise L2 (or abs-mean) to get 2D saliency
    saliency = np.mean(np.abs(grads), axis=0)  # [H, W]

    # Map label
    id2label = model.config.id2label
    label = id2label[pred_idx] if id2label and pred_idx in id2label else str(pred_idx)
    confidence = float(probs[pred_idx].item())

    # Build overlay
    overlay_img = overlay_heatmap_on_image(pil_img, saliency, alpha=0.4)
    heatmap_b64 = pil_to_b64(overlay_img)

    return label, confidence, heatmap_b64

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            abort(400, "No file uploaded.")
        if not allowed_file(file.filename):
            abort(400, "Only PNG files are accepted.")

        save_path = os.path.join(UPLOAD_FOLDER, os.path.basename(file.filename))
        file.save(save_path)

        label, conf, heatmap_b64 = predict_and_saliency(save_path)

        return render_template(
            "result.html",
            label=f"Prediction: {label}",
            confidence=f"{conf * 100:.2f}",
            heatmap=heatmap_b64
        )

    return render_template("index.html")

if __name__ == "__main__":
    # Flask dev server
    app.run(host="0.0.0.0", port=5001, debug=True)
