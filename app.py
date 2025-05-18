import os
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Load feature extractor (for mean/std normalization values)
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

# Download model weights if not already downloaded
model_url = "https://huggingface.co/JoeyJoeman/plantvillage-vit/resolve/main/vit-plantvillage.pth"
local_model_path = "vit-plantvillage.pth"

if not os.path.exists(local_model_path):
    import requests
    with open(local_model_path, "wb") as f:
        f.write(requests.get(model_url).content)

# Initialize model architecture with correct number of classes
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=len(class_names),
    ignore_mismatched_sizes=True
)

# Load saved weights into model
state_dict = torch.load(local_model_path, map_location=device)
model.load_state_dict(state_dict)

model.to(device)
model.eval()

# Preprocessing pipeline
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

    # Preprocess input image
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
        st.progress(pred_prob)  # progress bar for confidence
