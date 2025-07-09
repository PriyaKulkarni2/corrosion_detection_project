# 1G6lgXM7kI4O4NLde42umWwEu6olcbKX5 <- model fle id

import os
os.environ["PYTORCH_NO_WATCHER_WARNING"] = "1"

import streamlit as st
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from torchvision import models
from datetime import datetime
import json

import streamlit as st
st.set_page_config(page_title="Corrosion Detection", layout="wide")

from huggingface_hub import hf_hub_download

# 1) Define the exact same model architecture
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

# 2) Pull the weights from Hugging Face Hub (caches locally)
MODEL_PATH = hf_hub_download(
    repo_id="PriyaKulkarni2/corrosion_detection_model",
    filename="Corrosion_Detection.pth"          # whatever you named it on HF
)

# 3) Load into your model
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    st.write("Loading model…")
    model = Corrosion_Detection()
    # If you saved state_dict:
    state = torch.load(path, map_location="cpu")
    # If it's a full-save, switch to `model = torch.load(path)`
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model(MODEL_PATH)

# 4) Define any transforms once
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 5) Streamlit UI
st.title("Corrosion Detection")
st.write("Upload an image to see corrosion vs. no-corrosion")

uploaded = st.file_uploader("Choose an image…", type=["png", "jpg", "jpeg"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input", use_column_width=True)

    tensor = transform(img).unsqueeze(0)  # add batch dimension
    with st.spinner("Running inference…"):
        preds = model(tensor)
        label = torch.argmax(preds, dim=1).item()
        st.success("Prediction: **Corrosion**" if label == 1 else "Prediction: **No Corrosion**")

# Rest of your Streamlit app (image upload, prediction)
# ...

# changes for deploy


# Initialize session state for statistics
if 'corrosion_count' not in st.session_state:
    st.session_state['corrosion_count'] = 0
if 'no_corrosion_count' not in st.session_state:
    st.session_state['no_corrosion_count'] = 0
if 'total_predictions' not in st.session_state:
    st.session_state['total_predictions'] = 0
if 'session_results' not in st.session_state:
    st.session_state['session_results'] = []

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load('corrosion_resnet18.pth', map_location='cpu'))
    model.eval()
    return model

def predict_single(image, model, confidence_threshold=0.5):
    xb = image.unsqueeze(0)
    preds = model(xb)
    prediction = preds[0]
    a = prediction[0].item()
    b = prediction[1].item()
    confidence = max(a, b) / (abs(a) + abs(b) + 1e-8)
    
    # Apply confidence threshold
    if confidence < confidence_threshold:
        return "Low Confidence", confidence
    
    if a > b:
        return "Corrosion Detected", confidence
    else:
        return "No Corrosion Detected", confidence

def import_and_predict(image_data, model, confidence_threshold=0.5):
    if isinstance(image_data, np.ndarray):
        image = image_data
    else:
        image = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = transform(image)
    prediction, confidence = predict_single(tensor, model, confidence_threshold)
    return prediction, confidence

def update_statistics(prediction):
    st.session_state['total_predictions'] += 1
    if prediction == "Corrosion Detected":
        st.session_state['corrosion_count'] += 1
    elif prediction == "No Corrosion Detected":
        st.session_state['no_corrosion_count'] += 1

def add_to_session_results(filename, prediction, confidence, timestamp):
    st.session_state['session_results'].append({
        'filename': filename,
        'prediction': prediction,
        'confidence': confidence,
        'timestamp': timestamp
    })

def generate_session_report():
    report = {
        'session_info': {
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_predictions': st.session_state['total_predictions'],
            'corrosion_detected': st.session_state['corrosion_count'],
            'no_corrosion_detected': st.session_state['no_corrosion_count'],
            'corrosion_percentage': (st.session_state['corrosion_count'] / max(st.session_state['total_predictions'], 1)) * 100
        },
        'predictions': st.session_state['session_results']
    }
    return report

# Session state for feedback
if 'feedback_records' not in st.session_state:
    st.session_state['feedback_records'] = []
if 'input_method' not in st.session_state:
    st.session_state['input_method'] = 'Upload Image'

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # Confidence threshold
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence required for a prediction to be considered valid"
    )
    
    st.markdown("---")
    
    # Statistics
    st.markdown("## 📊 Session Statistics")
    st.metric("Total Predictions", st.session_state['total_predictions'])
    st.metric("Corrosion Detected", st.session_state['corrosion_count'])
    st.metric("No Corrosion Detected", st.session_state['no_corrosion_count'])
    
    if st.session_state['total_predictions'] > 0:
        corrosion_percentage = (st.session_state['corrosion_count'] / st.session_state['total_predictions']) * 100
        st.metric("Corrosion Rate", f"{corrosion_percentage:.1f}%")
    
    st.markdown("---")
    
    # Session Report
    st.markdown("## 📄 Session Report")
    if st.button("📥 Download Session Report", type="primary"):
        if st.session_state['session_results']:
            report = generate_session_report()
            
            # Create JSON report
            json_report = json.dumps(report, indent=2)
            st.download_button(
                label="Download JSON Report",
                data=json_report,
                file_name=f"corrosion_session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            # Create CSV report
            df_report = pd.DataFrame(st.session_state['session_results'])
            csv_report = df_report.to_csv(index=False)
            st.download_button(
                label="Download CSV Report",
                data=csv_report,
                file_name=f"corrosion_session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No predictions made yet in this session.")
    
    # Reset session
    if st.button("🔄 Reset Session"):
        st.session_state['corrosion_count'] = 0
        st.session_state['no_corrosion_count'] = 0
        st.session_state['total_predictions'] = 0
        st.session_state['session_results'] = []
        st.session_state['feedback_records'] = []
        st.rerun()

# Main content
st.markdown("""
    <h1 style='text-align: center; color: #2E86C1; margin-bottom:0;'>🛠️ Corrosion Detection</h1>
    <p style='text-align: center; color: #555; margin-top:0;'>Upload an image to detect corrosion. Your feedback helps us improve!</p>
""", unsafe_allow_html=True)
st.divider()

with st.spinner('Loading corrosion detection model...'):
    model = load_model()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# --- Input method as persistent buttons with emoji checkmark ---
col1, col2, col3 = st.columns(3)
selected = st.session_state['input_method']
with col1:
    if st.button(('✅ ' if selected == 'Upload Image' else '') + 'Upload Image', key="btn_upload"):
        st.session_state['input_method'] = 'Upload Image'
with col2:
    if st.button(('✅ ' if selected == 'Use Webcam' else '') + 'Use Webcam', key="btn_webcam"):
        st.session_state['input_method'] = 'Use Webcam'
with col3:
    if st.button(('✅ ' if selected == 'Batch Upload' else '') + 'Batch Upload', key="btn_batch"):
        st.session_state['input_method'] = 'Batch Upload'

results = []

# Helper for colored prediction box
def prediction_box(prediction, confidence):
    if prediction == "Corrosion Detected":
        color = "#F1948A"
        icon = "❌"
    elif prediction == "Low Confidence":
        color = "#F7DC6F"
        icon = "⚠️"
    else:
        color = "#82E0AA"
        icon = "✅"
    st.markdown(f"""
        <div style='background-color:{color};padding:1.2em 1em 1.2em 1em;border-radius:12px;text-align:center;margin-bottom:1em;'>
            <span style='font-size:2em;'>{icon}</span><br>
            <span style='font-size:1.3em;font-weight:bold;'>{prediction}</span><br>
            <span style='color:#555;'>Confidence: <b>{confidence:.2f}</b></span>
        </div>
    """, unsafe_allow_html=True)

# Feedback button logic
def feedback_buttons(key_prefix):
    col1, col2 = st.columns(2)
    feedback = None
    with col1:
        if st.button("👍 Correct", key=key_prefix+"_correct"):
            feedback = "Yes"
    with col2:
        if st.button("👎 Incorrect", key=key_prefix+"_incorrect"):
            feedback = "No"
    return feedback

def correct_label_buttons(key_prefix):
    col1, col2 = st.columns(2)
    label = None
    with col1:
        if st.button("❌ Corrosion Detected", key=key_prefix+"_corrosion"):
            label = "Corrosion Detected"
    with col2:
        if st.button("✅ No Corrosion Detected", key=key_prefix+"_nocorrosion"):
            label = "No Corrosion Detected"
    return label

if selected == "Upload Image":
    file = st.file_uploader("Upload a picture", type=["jpg", "png"])
    if file is not None:
        image = Image.open(file)
        st.image(image, use_container_width=True)
        prediction, confidence = import_and_predict(image, model, confidence_threshold)
        prediction_box(prediction, confidence)
        
        # Update statistics and session results
        if prediction != "Low Confidence":
            update_statistics(prediction)
        add_to_session_results(file.name, prediction, confidence, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        st.markdown("<h5 style='text-align:center;'>Is this prediction correct?</h5>", unsafe_allow_html=True)
        feedback = feedback_buttons(file.name)
        correct_label = None
        if feedback == "No":
            st.markdown("<h5 style='text-align:center;'>What is the correct label?</h5>", unsafe_allow_html=True)
            correct_label = correct_label_buttons(file.name)
        if feedback:
            if feedback == "Yes" or (feedback == "No" and correct_label):
                if st.button("Submit Feedback", key=file.name+"_submit"):
                    st.session_state['feedback_records'].append({
                        "filename": file.name,
                        "prediction": prediction,
                        "confidence": confidence,
                        "feedback": feedback,
                        "correct_label": correct_label
                    })
                    st.success("Thank you for your feedback!")
elif selected == "Use Webcam":
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        image = Image.open(camera_image)
        st.image(image, use_container_width=True)
        prediction, confidence = import_and_predict(image, model, confidence_threshold)
        prediction_box(prediction, confidence)
        
        # Update statistics and session results
        if prediction != "Low Confidence":
            update_statistics(prediction)
        add_to_session_results("webcam_capture", prediction, confidence, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        st.markdown("<h5 style='text-align:center;'>Is this prediction correct?</h5>", unsafe_allow_html=True)
        feedback = feedback_buttons("webcam")
        correct_label = None
        if feedback == "No":
            st.markdown("<h5 style='text-align:center;'>What is the correct label?</h5>", unsafe_allow_html=True)
            correct_label = correct_label_buttons("webcam")
        if feedback:
            if feedback == "Yes" or (feedback == "No" and correct_label):
                if st.button("Submit Feedback", key="webcam_submit"):
                    st.session_state['feedback_records'].append({
                        "filename": "webcam_capture",
                        "prediction": prediction,
                        "confidence": confidence,
                        "feedback": feedback,
                        "correct_label": correct_label
                    })
                    st.success("Thank you for your feedback!")
elif selected == "Batch Upload":
    files = st.file_uploader("Upload multiple images", type=["jpg", "png"], accept_multiple_files=True)
    if files:
        for file in files:
            image = Image.open(file)
            st.image(image, caption=file.name, use_container_width=True)
            prediction, confidence = import_and_predict(image, model, confidence_threshold)
            prediction_box(prediction, confidence)
            
            # Update statistics and session results
            if prediction != "Low Confidence":
                update_statistics(prediction)
            add_to_session_results(file.name, prediction, confidence, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            
            st.markdown(f"<h5 style='text-align:center;'>Is the prediction for <b>{file.name}</b> correct?</h5>", unsafe_allow_html=True)
            feedback = feedback_buttons(file.name)
            correct_label = None
            if feedback == "No":
                st.markdown(f"<h5 style='text-align:center;'>What is the correct label for <b>{file.name}</b>?</h5>", unsafe_allow_html=True)
                correct_label = correct_label_buttons(file.name)
            if feedback:
                if feedback == "Yes" or (feedback == "No" and correct_label):
                    if st.button(f"Submit Feedback for {file.name}", key=file.name+"_submit"):
                        st.session_state['feedback_records'].append({
                            "filename": file.name,
                            "prediction": prediction,
                            "confidence": confidence,
                            "feedback": feedback,
                            "correct_label": correct_label
                        })
                        st.success(f"Thank you for your feedback on {file.name}!")

st.divider()

if st.session_state['feedback_records']:
    st.markdown("<h4 style='color:#117A65;'>Download Your Feedback</h4>", unsafe_allow_html=True)
    df = pd.DataFrame(st.session_state['feedback_records'])
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Feedback as CSV", data=csv, file_name="corrosion_feedback.csv", mime="text/csv")
