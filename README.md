This project is for a Troy University computer science master's level course. It is intended to detect disease in plant images. There are 38 possible categories.

This may be useful if you're a farmer or a gardener. This app is still in development, so it is not completely reliable yet.

A previous CNN baseline achieved 88 percent accuracy. This fine-tuned pretrained Vision Transformer achieved nearly 100 percent accuracy.

The pretrained model can be found here: https://huggingface.co/google/vit-base-patch16-224. 

My fine-tuned model is available here: huggingface.co/JoeyJoeman/plantvillage-vit

This ViT was trained on the colored, nonsegmented images in the PlantVillage dataset available here: https://www.kaggle.com/datasets/mohitsingh1804/plantvillage

Images were resized to 224×224 and normalized using ViT image mean and std. I applied ViTFeatureExtractor from HuggingFace

Accuracy: near 100%
Macro / Weighted Average Precision: 99 / 100%
Macro / Weighted Average Recall: 99 / 100%
Macro / Weighted Average F1-Score: 99 / 100%

You can run my version of the app here:
https://plant-disease-detection-rueztgdffhmrxcrqqqkecv.streamlit.app/

Acknowledgments:
HuggingFace Transformers
timm (PyTorch Image Models)
PlantVillage Dataset
Streamlit
Scikit-learn (metrics)
