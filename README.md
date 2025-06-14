# 🧬 Breast Cancer Histopathology Classification

A collection of deep learning models for binary classification (cancer / non-cancer) of breast histopathology images. This project explores:

* Custom Convolutional Neural Networks (CNN)

* Transfer Learning using pretrained models:

  -VGG16

  -ResNet50

  -Xception

## 📁 Dataset

Source:

[🔗 Breast Histopathology Images - Kaggle](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)

This dataset contains 277,524 image patches (50x50 px) extracted from whole-slide images. 
Each image is labeled as either cancerous (class1) or non-cancerous (class0).

## 📦 Installation & Setup

1. Clone the Repository
   
```bash
git clone https://github.com/your-username/breast-cancer-histopathology.git
cd breast-cancer-histopathology
  ```
2. Install Requirements

```bash
pip install tensorflow keras opencv-python matplotlib seaborn scikit-learn kaggle
  ```
## 📥 Dataset Download (Colab + Kaggle)

To use the dataset in Google Colab:

1. Upload your `kaggle.json` API key to your Google Drive.

2. Mount your drive and download the dataset:

```bash
from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# Set Kaggle config directory
os.environ['KAGGLE_CONFIG_DIR'] = "/content/drive/MyDrive/Colab Notebooks/dosya"

# Navigate to target directory
%cd /content/drive/MyDrive/Colab Notebooks/dosya

# Download and unzip
!kaggle datasets download -d paultimothymooney/breast-histopathology-images
!unzip \*.zip && rm *.zip
  ```
## 🧪 Models Included

Each model is provided in a separate Python script:

| Model Type         | File                          | Description                                 |
|--------------------|-------------------------------|---------------------------------------------|
| CNN (Basic)        | `cnn_model1.py`               | Shallow CNN with input size 50x50           |
| CNN (Improved)     | `cnn_model2.py`               | Deeper CNN with 75x75 images + augmentation |
| VGG16 Transfer     | `transfer_learning_vgg16.py`  | Transfer learning using frozen VGG16        |
| ResNet50 Transfer  | `transfer_learning_resnet50.py`| Transfer learning using ResNet50            |
| Xception Transfer  | `transfer_learning_xception.py`| Transfer learning using Xception            |




## 🧠 Training & Evaluation

Each script includes:

* Image loading and preprocessing

* Train/test splitting

* CNN/Transfer Learning model definition

* Model training with validation

* Evaluation using:

  - Accuracy plots

  - Confusion Matrix

  - Classification Report
 
## 📌 Notes

* All models assume the dataset path: `/content/drive/MyDrive/Colab Notebooks/dosya/
` 
* All training scripts are designed for binary classification: cancer vs. non-cancer

* For custom usage, you can modify input size and model structure.

## 📚 References

- Breast Histopathology Dataset on Kaggle  
  ([https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images))

- Keras Applications Documentation - VGG16  
  ([https://keras.io/api/applications/vgg/#vgg16-function](https://keras.io/api/applications/vgg/#vgg16-function))

- Keras Applications Documentation - ResNet50  
  ([https://keras.io/api/applications/resnet/#resnet50-function](https://keras.io/api/applications/resnet/#resnet50-function))

- Keras Applications Documentation - Xception  
  ([https://keras.io/api/applications/xception/](https://keras.io/api/applications/xception/))

## 📄 License

This project is open source under the MIT License.
See the [LICENSE](LICENSE) file for details.


