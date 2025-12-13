# Zebrafish Epithelial Cell Classification  
**CS184A – Group 5 Final Project**

---

## 1. Project Overview

This project develops a deep learning pipeline to automatically classify **zebrafish epithelial cells** as **Wild-Type (WT)** or **Transgenic (TG)** from confocal microscopy images.

Manual annotation of microscopy images is time-consuming, subjective, and not scalable. Our system addresses this challenge by:

- Automatically segmenting individual cells from full microscopy images  
- Automatically labeling cells into WT or TG classes  
- Training CNN-based models to classify each cell  
- Comparing a baseline CNN with transfer learning models (ResNet50 and EfficientNet-B0)

## 2. Repository Structure

Project/  
├── CS184A_Group_5_Final_Project.ipynb  
├── Data/  
│ ├── image01.tif  
│ ├── image02.tif  
│ ├── ...  
│ └── image41.tif  
├── Output/  
│ ├── WT/  
│ └── TG/  
├── requirements.txt  
└── README.md  

## 3. Environment Setup

### Python Version
- Python **3.9 – 3.10** recommended  
- Tested on **Google Colab (GPU enabled)**

### Install Dependencies

Install required packages using:
pip install -r requirements.txt


## 4. Data Organization
Full Dataset (Used in Training)
The dataset consists of 41 confocal microscopy images:

Data/  
├── image01.tif  
├── image02.tif  
├── ...  
└── image41.tif  
Upload the Data/ folder to Google Drive and update the path in the notebook:

DATA_ROOT = "/content/drive/MyDrive/Data"
Output Dataset (Auto-generated)
After segmentation and auto-labeling, cropped single-cell images are saved to:
Output/  
├── WT/  
└── TG/  


## 5. How to Run the Project
Run on Google Colab (Recommended)
Open Google Colab

Upload the notebook:
CS184A_Group_5_Final_Project.ipynb

Enable GPU:
Runtime → Change runtime type → Hardware accelerator → GPU

Mount Google Drive:
from google.colab import drive
drive.mount('/content/drive')

## 6. Execution Pipeline
### Step 1: Cell Segmentation & Auto-labeling
Segments individual cells from the 41 original microscopy images

Automatically assigns each cell to WT or TG

Cropped cells are saved to:
/content/drive/MyDrive/CS184A/Project/Output/  
├── WT/  
└── TG/  

### Step 2: Baseline CNN Training (Student Y Nhi Tran)
Includes:
Dataset loading
Data preprocessing
Baseline CNN training
Validation and testing
Confusion matrix and performance metrics

Best model saved to:
/content/drive/MyDrive/CS184A/Project/baseline_best.pth

### Step 3: Transfer Learning Models (Student Yasmeen Soe)
Models trained:
ResNet50 (ImageNet pretrained)
EfficientNet-B0 (ImageNet pretrained)

Best models saved as:
resnet50_best.pth
efficientnet_best.pth

### Step 4: Evaluation & Comparison
Final evaluation includes:
Accuracy
Precision
Recall
F1-score
Confusion matrix
Bar plot comparing Baseline CNN vs Transfer Learning models

## 7. Results Summary (Final Run)
Model 
Baseline CNN has F1 Score: 0.70  
ResNet50 has F1 Score: 0.83  
EfficientNet-B0	has F1 Score: 0.91  

EfficientNet-B0 achieved the best overall performance.

## 8. Reproducibility Notes
Results may vary slightly between runs due to random initialization
Best models are selected based on validation F1-score
Fixed train/validation/test splits are used

## 9. Authors & Contributions
Y Nhi Tran: Data preprocessing, cell segmentation, baseline CNN, evaluation
Yi Wang: Dataset acquisition, annotation guidance, validation, report writing
Yasmeen Soe: Transfer learning models, hyperparameter tuning, model comparison

## 10. Contact
For questions regarding the project or code execution, please contact the team via the GitHub repository.
