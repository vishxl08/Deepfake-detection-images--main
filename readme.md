🧠 Deepfake Detection using ResNet-18
=====================================

This project implements a Deepfake Image Classification system using ResNet-18 to detect whether a given human face image is Real or Fake (Manipulated).
The model is trained on the Deepfake and Real Images Dataset and deployed through a Streamlit frontend for interactive use.

------------------------------------------------------------
📊 Dataset Overview
------------------------------------------------------------

Dataset Source: https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images

This dataset consists of manipulated (deepfake) and authentic human face images.
- Each image is 256×256 JPG.
- The dataset was further processed for better performance and balance.

------------------------------------------------------------
⚙️ Model Architecture
------------------------------------------------------------

- Base Model: ResNet-18 (pretrained on ImageNet)
- Final Layer: Fully connected layer with 2 output neurons (Real, Fake)
- Optimizer: Adam (lr=1e-4)
- Loss Function: CrossEntropyLoss
- Scheduler: StepLR (step_size=7, gamma=0.1)
- Epochs: 10
- Batch Size: 32
- Accuracy: ~85% on test set

------------------------------------------------------------
🧠 Training Details
------------------------------------------------------------

Data Transformations:
Normalize([0.485, 0.456, 0.406],
          [0.229, 0.224, 0.225])

Images were resized to 256×256, then center-cropped to 224×224 during training and testing.

Key Libraries:
- torch, torchvision
- matplotlib, seaborn
- scikit-learn (for confusion matrix)

------------------------------------------------------------
🚀 Streamlit Frontend
------------------------------------------------------------

The app.py script provides an interactive interface for users to upload images and get predictions.

Features:
- Upload an image (JPG, PNG)
- Model predicts Real or Fake
- Displays prediction confidence
- Clean and responsive UI

------------------------------------------------------------
💻 How to Run the App
------------------------------------------------------------

1️⃣ Install dependencies
-----------------------
```
git clone "https://github.com/Hero0p/Deepfake-detection-images-"

pip install -r requirements.txt
```

2️⃣ Run the Streamlit app
------------------------
```
streamlit run app.py
```
If running on Google Colab:
```
!streamlit run app.py & npx localtunnel --port 8501
```

------------------------------------------------------------
🧩 Requirements
------------------------------------------------------------

torch
torchvision
streamlit
Pillow
numpy
matplotlib
seaborn
scikit-learn

------------------------------------------------------------
🧠 Example Prediction
------------------------------------------------------------

Uploaded Image: Example Face
Prediction: Fake
Confidence: 94.2%

------------------------------------------------------------
🧾 Results Summary
------------------------------------------------------------

| Metric      | Train | Validation | Test |
|--------------|--------|-------------|------|
| Accuracy     | 90%    | 86%         | 85%  |

Confusion Matrix Example:
| True / Pred | Fake | Real |
|--------------|------|------|
| Fake         | 420  | 80   |
| Real         | 65   | 435  |

------------------------------------------------------------
🧩 Future Improvements
------------------------------------------------------------

- Add Grad-CAM heatmaps for interpretability
- Support video deepfake detection
- Add model retraining pipeline for incremental learning

------------------------------------------------------------
🪶 License
------------------------------------------------------------

This project is for research and educational purposes only.  
Please respect dataset and model usage rights.

------------------------------------------------------------
