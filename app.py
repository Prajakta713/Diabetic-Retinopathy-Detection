import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import vgg16
import torchvision.transforms as transforms
import cv2
import numpy as np
import gdown
import os

# ----- Google Drive Model URL -----
MODEL_URL = "https://drive.google.com/uc?id=10rv2BL3IYGif1_4jnOBPAPEAMiPE5Z1W"
MODEL_PATH = "vgg16_custom_model_diabetic_retinopathy.pth"

# ----- Function to download model from Google Drive -----
@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model... please wait ⏳")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)  # <--- added fuzzy=True
        if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
            st.error("Model download failed. Please check the Google Drive link.")
            st.stop()
    return MODEL_PATH

# ----- Define Custom VGG16 Architecture -----
class CustomVGG16(nn.Module):
    def __init__(self):
        super(CustomVGG16, self).__init__()
        self.features = vgg16(weights='IMAGENET1K_V1').features  # Updated for PyTorch >= 0.13
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 5)  # 5 output classes
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

# ----- Load Model -----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_file = download_model()
model = CustomVGG16().to(device)

# Load state_dict safely (ensure model file is correct)
try:
    state_dict = torch.load(model_file, map_location=device)
    model.load_state_dict(state_dict)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

model.eval()

# ----- Preprocess Uploaded Image -----
def preprocess_image(image_bytes):
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    img = img.unsqueeze(0)
    return img.to(device)

# ----- Predict Function -----
def predict_image(img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
    _, predicted = torch.max(output, 1)
    return predicted.item()

# ----- Class Labels -----
class_labels = {
    0: "No DR (Healthy Retina)",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}

# ----- Streamlit Web App -----
st.title("🩺 Diabetic Retinopathy Detection")
st.write("Upload a **retinal fundus image** to predict the stage of **Diabetic Retinopathy (DR)** using a fine-tuned VGG16 deep learning model.")

uploaded_file = st.file_uploader("📸 Upload an Eye Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    st.write("Processing image... 🔍")

    image_bytes = uploaded_file.read()
    img_tensor = preprocess_image(image_bytes)
    prediction = predict_image(img_tensor)

    st.success(f"### 🧠 Prediction: {class_labels[prediction]}")
    st.write(f"**Class Index:** {prediction}")
    st.balloons()

