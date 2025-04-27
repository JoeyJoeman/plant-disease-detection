import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class names
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
               'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
               'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
               'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
               'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Potato___Early_blight']

# Load feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# Download your model
model_path = 'https://huggingface.co/JoeyJoeman/plantvillage-vit/blob/main/vit-plantvillage.pth'
local_model_path = 'vit-plantvillage.pth'

# If not already downloaded
if not os.path.exists(local_model_path):
    import requests
    with open(local_model_path, 'wb') as f:
        f.write(requests.get(model_path).content)

# Initialize a fresh ViT model
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=len(class_names),
    ignore_mismatched_sizes=True  # <--- key fix
)
# Load your fine-tuned weights
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
st.title('🌿 Plant Disease Classifier (ViT)')

uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs.logits, 1)
    
    prediction = class_names[preds.item()]
    
    st.success(f"Prediction: **{prediction}**")
