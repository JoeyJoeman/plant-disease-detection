This Project is for a class. It is intended to detect disease in plant images. There are 38 possible categories.

This may be useful if you're a farmer or a gardener. This app is still in development, so it is not completely reliable yet.

The CNN baseline achieved 88 percent accuracy. The Vision Transformer achieved nearly 100 percent accuracy.

This ViT was fine-tuned on the colored, nonsegmented images in the PlantVillage dataset from Kaggle.


Images resized to 224×224

Normalized using ViT image mean and std

Applied ViTFeatureExtractor from HuggingFace



You can run my version of the app here:
https://plant-disease-detection-rueztgdffhmrxcrqqqkecv.streamlit.app/

Acknowledgments:

HuggingFace Transformers
timm (PyTorch Image Models)
PlantVillage Dataset
Streamlit
Scikit-learn (metrics)
Thanks to all contributors and open-source maintainers!
