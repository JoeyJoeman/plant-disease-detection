import os
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names
class_names = [
    "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy", "Potato___Early_blight", "Potato___healthy",
    "Potato___Late_blight", "Tomato__Target_Spot", "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato_Bacterial_spot",
    "Tomato_Early_blight", "Tomato_healthy", "Tomato_Late_blight",
    "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites_Two_spotted_spider_mite"
]

# Load feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

# Download model if not already downloaded
model_url = "https://huggingface.co/JoeyJoeman/plantvillage-vit/blob/main/vit-plantvillage.pth"
local_model_path = "vit-plantvillage.pth"

if not os.path.exists(local_model_path):
    import requests
    with open(local_model_path, "wb") as f:
        f.write(requests.get(model_url).content)

# Initialize and load model
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=len(class_names),
    ignore_mismatched_sizes=True
)
model.load_state_dict(torch.load(local_model_path, map_location=device))
model.to(device)
model.eval()

# Define preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])

# Streamlit UI
st.title("Plant Disease Detector (ViT)")

uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs.logits, dim=1)
        top_probs, top_idxs = probs.topk(3, dim=1)
    
    st.subheader("Top Predictions:")
    for i in range(3):
        pred_class = class_names[top_idxs[0][i].item()]
        pred_prob = top_probs[0][i].item()
        
        st.write(f"**{pred_class}** - {pred_prob:.2%} confidence")
        st.progress(pred_prob)  # Adds a clean progress bar
