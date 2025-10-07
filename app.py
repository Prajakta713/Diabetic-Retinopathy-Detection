import streamlit as st
import torch
from torchvision.models import vgg16
import gdown
import os

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Google Drive file ID and local path
FILE_ID = '10rv2BL3IYGif1_4jnOBPAPEAMiPE5Z1W'
MODEL_PATH = 'vgg16_custom_model_diabetic_retinopathy.pth'

# Function to download model from Google Drive
def download_model():
    if not os.path.exists(MODEL_PATH):
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        gdown.download(url, MODEL_PATH, quiet=False)
        st.success("Model downloaded successfully!")

# Function to load model
def load_model():
    # Create model
    model = vgg16(pretrained=False)
    model.classifier[6] = torch.nn.Linear(4096, 5)  # Assuming 5 classes
    # Load state dict
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# Download and load model
download_model()
model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose a fundus image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    from PIL import Image
    import torchvision.transforms as transforms

    # Preprocess the image
    image = Image.open(uploaded_file).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    # Make prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    
    st.write(f"Predicted class: {predicted.item()}")
