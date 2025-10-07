import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import gdown
import os

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Diabetic Retinopathy Detection", layout="centered")

# Google Drive file ID
FILE_ID = "https://drive.google.com/uc?id=10rv2BL3IYGif1_4jnOBPAPEAMiPE5Z1W"  # Replace with your model's Google Drive file ID
MODEL_PATH = "vgg16_custom_model_diabetic_retinopathy.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- MODEL DOWNLOAD --------------------
@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        st.info("Downloading model from Google Drive...")
        gdown.download(url, MODEL_PATH, quiet=False)
    st.success("Model ready!")
    return MODEL_PATH

# -------------------- MODEL LOADING --------------------
@st.cache_resource
def load_model():
    from torchvision.models import vgg16

    # Create model
    model = vgg16(pretrained=False)
    model.classifier[6] = torch.nn.Linear(4096, 5)  # Assuming 5 classes for DR
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# -------------------- IMAGE TRANSFORM --------------------
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

# -------------------- PREDICTION --------------------
def predict(image: Image.Image):
    tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# -------------------- STREAMLIT APP --------------------
st.title("🩸 Diabetic Retinopathy Detection")
st.write("Upload a fundus image and the model will predict the DR stage.")

# Download and load model
download_model()
model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose a fundus image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            class_idx = predict(image)
            classes = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
            st.success(f"Prediction: **{classes[class_idx]}**")
