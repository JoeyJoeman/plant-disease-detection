import os
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class names
class_names = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

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
    ignore_mismatched_sizes=True
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

# --- Streamlit UI ---

st.title('🌿 Plant Disease Classifier (ViT)')

uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Preprocess
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs.logits, 1)
    
    prediction = class_names[preds.item()]
    
    st.success(f"Prediction: **{prediction}**")

    # Button to show evaluation
    if st.button("📊 Show Model Evaluation"):
        
        # --- Classification Report Data ---
        report_data = {
            'Class': [
                'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
                'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
                'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite',
                'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
                'Tomato_healthy'
            ],
            'Precision': [0.99, 1.00, 1.00, 1.00, 1.00, 1.00, 0.80, 0.99, 1.00, 1.00, 1.00, 0.98, 0.99, 0.43, 0.99],
            'Recall': [0.99, 1.00, 1.00, 0.98, 1.00, 1.00, 1.00, 0.99, 0.98, 0.99, 0.66, 0.86, 1.00, 1.00, 1.00],
            'F1-Score': [0.99, 1.00, 1.00, 0.99, 1.00, 1.00, 0.89, 0.99, 0.99, 0.99, 0.80, 0.92, 0.99, 0.61, 1.00]
        }
        report_df = pd.DataFrame(report_data)

        st.subheader("🔎 Classification Report")
        st.dataframe(report_df)

        # --- Confusion Matrix ---
        confusion_matrix = [
            [193,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
            [1,305,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,225,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,200,0,0,0,4,0,0,0,0,0,0,0],
            [0,0,0,0,32,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,440,1,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,207,0,0,0,0,0,1,0,0],
            [0,0,0,0,0,1,3,360,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,1,0,166,0,0,0,1,1,0],
            [0,0,0,0,0,0,3,0,0,361,0,0,0,2,0],
            [0,0,0,0,0,0,7,0,0,0,230,4,6,101,0],
            [0,0,0,0,0,0,36,0,0,0,0,228,0,0,2],
            [0,0,0,0,0,1,0,0,0,0,0,0,612,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,80,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,312]
        ]

        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=report_data['Class'], yticklabels=report_data['Class'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')

        st.subheader("🧩 Confusion Matrix")
        st.pyplot(fig)
