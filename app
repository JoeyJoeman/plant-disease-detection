import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from src.model import build_model

st.title("Plant Disease Classifier")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    img = transform(image).unsqueeze(0)

    model = build_model(num_classes=38)  # adjust to your classes
    model.load_state_dict(torch.load("outputs/models/best_model.pth", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        prediction = model(img)
        predicted_class = torch.argmax(prediction, dim=1).item()

    # Load class names (e.g., from a list or class index)
    classes = ["Apple___Black_rot", "Apple___healthy", "..."]
    st.success(f"Prediction: {classes[predicted_class]}")
