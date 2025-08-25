import os
import base64
import requests
from flask import Flask, request, render_template
import numpy as np
from PIL import Image
import torch
import cv2
import io
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoModel

# ==== Load Hugging Face models ====
# Model for Pneumonia Classification
HF_CLASSIFICATION_MODEL_NAME = "dima806/chest_xray_pneumonia_detection"
processor_clf = AutoImageProcessor.from_pretrained(HF_CLASSIFICATION_MODEL_NAME)
model_clf = AutoModelForImageClassification.from_pretrained(HF_CLASSIFICATION_MODEL_NAME)

# Model for Segmentation and View Classification
HF_SEGMENTATION_MODEL_NAME = "ianpan/chest-x-ray-basic"
device = "cuda" if torch.cuda.is_available() else "cpu"
model_seg = AutoModel.from_pretrained(HF_SEGMENTATION_MODEL_NAME, trust_remote_code=True)
model_seg = model_seg.eval().to(device)

# ==== Flask setup ====
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==== Preprocess for Classification Model ====
def prepare_image_clf(img_path):
    image = Image.open(img_path).convert("RGB")
    inputs = processor_clf(images=image, return_tensors="pt")
    return inputs

# ==== Prediction using Classification Model ====
def predict_classification_hf(img_path):
    inputs = prepare_image_clf(img_path)
    with torch.no_grad():
        outputs = model_clf(**inputs)
    logits = outputs.logits
    predicted_label_idx = logits.argmax(-1).item()
    
    id2label = model_clf.config.id2label
    predicted_label = id2label[predicted_label_idx]
    
    probabilities = torch.softmax(logits, dim=-1)[0]
    confidence = probabilities[predicted_label_idx].item()
    
    return predicted_label, confidence

# ==== Prediction and Segmentation using Segmentation Model ====
def predict_segmentation_hf(img_path):
    img = cv2.imread(img_path, 0) # Load image in grayscale
    x = model_seg.preprocess(img) # Preprocess for the model
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0) # Add channel, batch dims
    x = x.float()

    with torch.inference_mode():
        out = model_seg(x.to(device))

    # View Classification
    view_labels = ["AP", "PA", "lateral"]
    predicted_view_idx = out["view"].argmax(-1).item()
    predicted_view = view_labels[predicted_view_idx]
    
    # Segmentation mask
    mask = out["mask"].argmax(1).squeeze(0).cpu().numpy() # 1=right lung, 2=left lung, 3=heart

    return predicted_view, mask

# ==== Generate Heatmap from Segmentation Mask ====
def generate_heatmap_from_mask(original_img_path, mask):
    original_img = cv2.imread(original_img_path)
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Resize the mask to the original image's dimensions
    mask_resized = cv2.resize(mask.astype(np.uint8), 
                              (original_img_rgb.shape[1], original_img_rgb.shape[0]), 
                              interpolation=cv2.INTER_NEAREST)

    # Create a blank image for the heatmap
    heatmap = np.zeros_like(original_img_rgb, dtype=np.uint8)

    # Define colors for different segments (e.g., red for lungs, blue for heart)
    # You can adjust these colors as needed
    lung_color = [255, 0, 0] # Red for lungs
    heart_color = [0, 0, 255] # Blue for heart

    # Apply colors to the heatmap based on the resized mask
    # Right lung (1) and Left lung (2)
    heatmap[mask_resized == 1] = lung_color
    heatmap[mask_resized == 2] = lung_color
    # Heart (3)
    heatmap[mask_resized == 3] = heart_color

    # Blend the heatmap with the original image
    alpha = 0.4 # Transparency factor
    blended_img = cv2.addWeighted(original_img_rgb, 1 - alpha, heatmap, alpha, 0)

    # Convert to PIL Image and then to base64
    pil_img = Image.fromarray(blended_img)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# ==== Routes ====
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            img_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(img_path)

            # Use the classification model for pneumonia prediction
            pneumonia_label, pneumonia_conf = predict_classification_hf(img_path)
            
            # Use the segmentation model for view prediction and heatmap generation
            predicted_view, segmentation_mask = predict_segmentation_hf(img_path)
            
            # Generate heatmap from the segmentation mask
            heatmap_b64 = generate_heatmap_from_mask(img_path, segmentation_mask)

            # Combine labels for display
            label = f"Pneumonia: {pneumonia_label} (View: {predicted_view})"
            confidence = pneumonia_conf * 100 # Convert to percentage

            return render_template("result.html",
                                   label=label,
                                   confidence=round(confidence, 2),
                                   heatmap=heatmap_b64)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
