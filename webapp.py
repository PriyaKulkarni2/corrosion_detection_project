import os
os.environ["PYTORCH_NO_WATCHER_WARNING"] = "1"

import streamlit as st
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import numpy as np
import pandas as pd
from datetime import datetime
import json
from huggingface_hub import hf_hub_download

# Streamlit config
st.set_page_config(page_title="Corrosion Detection", layout="wide")

# Define Model
class Corrosion_Detection(nn.Module):
    def __init__(self):
        super(Corrosion_Detection, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(256 * 28 * 28, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        return self.network(x)

# Load model weights from Hugging Face
MODEL_PATH = hf_hub_download(
    repo_id="PriyaKulkarni2/corrosion_detection_model",
    filename="Corrosion_Detection.pth"
)

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    model = Corrosion_Detection()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

# Global transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Prediction function (no caching due to tensor input)
def predict_single(_image, _model, confidence_threshold=0.5):
    tensor = transform(_image).unsqueeze(0)  # [1, 3, 224, 224]
    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    if confidence < confidence_threshold:
        return "Low Confidence", confidence

    return ("Corrosion Detected" if pred_idx == 1 else "No Corrosion Detected", confidence)

# --- Session State Setup ---
for key in ['corrosion_count', 'no_corrosion_count', 'total_predictions', 'session_results']:
    if key not in st.session_state:
        st.session_state[key] = 0 if "count" in key or "total" in key else []

# --- Sidebar ---
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

    st.markdown("---")
    st.markdown("## 📊 Session Statistics")
    st.metric("Total Predictions", st.session_state['total_predictions'])
    st.metric("Corrosion Detected", st.session_state['corrosion_count'])
    st.metric("No Corrosion Detected", st.session_state['no_corrosion_count'])

    if st.session_state['total_predictions'] > 0:
        corrosion_percentage = (st.session_state['corrosion_count'] / st.session_state['total_predictions']) * 100
        st.metric("Corrosion Rate", f"{corrosion_percentage:.1f}%")

    if st.button("🔄 Reset Session"):
        for key in ['corrosion_count', 'no_corrosion_count', 'total_predictions', 'session_results']:
            st.session_state[key] = 0 if "count" in key or "total" in key else []

# --- Load Model ---
with st.spinner("Loading model..."):
    model = load_model(MODEL_PATH)

# --- UI Header ---
st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>🛠️ Corrosion Detection</h1>
    <p style='text-align: center; color: #555;'>Upload an image to detect corrosion.</p>
""", unsafe_allow_html=True)
st.divider()

# --- File Upload ---
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    prediction, confidence = predict_single(image, model, confidence_threshold)

    # --- Result Box ---
    color = "#F1948A" if prediction == "Corrosion Detected" else "#82E0AA"
    icon = "❌" if prediction == "Corrosion Detected" else "✅"
    if prediction == "Low Confidence":
        color, icon = "#F7DC6F", "⚠️"
    st.markdown(f"""
        <div style='background-color:{color};padding:1em;border-radius:12px;text-align:center;'>
            <span style='font-size:2em;'>{icon}</span><br>
            <b style='font-size:1.3em;'>{prediction}</b><br>
            <small>Confidence: {confidence:.2f}</small>
        </div>
    """, unsafe_allow_html=True)

    # --- Update stats ---
    if prediction != "Low Confidence":
        st.session_state['total_predictions'] += 1
        if prediction == "Corrosion Detected":
            st.session_state['corrosion_count'] += 1
        else:
            st.session_state['no_corrosion_count'] += 1

    # --- Session record ---
    st.session_state['session_results'].append({
        "filename": uploaded_file.name,
        "prediction": prediction,
        "confidence": confidence,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    # --- Feedback ---
    st.markdown("<h5 style='text-align:center;'>Is this prediction correct?</h5>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    feedback = None
    with col1:
        if st.button("👍 Yes"):
            feedback = "Yes"
    with col2:
        if st.button("👎 No"):
            feedback = "No"

    if feedback == "No":
        st.markdown("<h5 style='text-align:center;'>What should it be?</h5>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        correct_label = None
        with col1:
            if st.button("✅ No Corrosion"):
                correct_label = "No Corrosion Detected"
        with col2:
            if st.button("❌ Corrosion"):
                correct_label = "Corrosion Detected"
        if correct_label:
            st.success(f"Feedback recorded: Correct label is **{correct_label}**")

# --- Feedback Download ---
if st.session_state['session_results']:
    st.divider()
    st.markdown("### 📥 Download Session Results")
    df = pd.DataFrame(st.session_state['session_results'])
    st.download_button("Download CSV", df.to_csv(index=False), file_name="corrosion_session.csv", mime="text/csv")
