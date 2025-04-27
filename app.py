import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
from transformers import ViTForImageClassification, AutoImageProcessor

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class names
class_names = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___healthy', 
    'Potato___Late_blight', 'Tomato__Target_Spot', 'Tomato__Tomato_mosaic_virus',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato_Bacterial_spot', 
    'Tomato_Early_blight', 'Tomato_healthy', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite'
]

# Load image processor
feature_extractor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')

# Download your model (fixed)
model_url = 'https://huggingface.co/JoeyJoeman/plantvillage-vit/resolve/main/vit-plantvillage.pth'
local_model_path = 'vit-plantvillage.pth'

if not os.path.exists(local_model_path):
    import requests
    with open(local_model_path, 'wb') as f:
        f.write(requests.get(model_url).content)

# Initialize ViT model
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=len(class_names),
    ignore_mismatched_sizes=True
)
# Load fine-tuned weights
model.load_state_dict(torch.load(local_model_path, map_location=device))
model.to(device)
model.eval()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])

# Streamlit UI
st.title('🌿 Plant Disease Classifier (ViT)')

uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs.logits, 1)
    
    prediction = class_names[preds.item()]
    
    st.success(f"Prediction: **{prediction}**")
