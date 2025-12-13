Zebrafish Epithelial Cell Classification

CS184A – Group 5 Final Project

========= Project Overview =========

This project develops a deep learning pipeline to automatically classify zebrafish epithelial cells as Wild-Type (WT) or Transgenic (TG) from confocal microscopy images.

Manual annotation of microscopy images is time-consuming and error-prone. Our system addresses this problem by:

1. Automatically segmenting individual cells from full microscopy images

2. Training CNN-based models to classify each cell as WT or TG

3. Comparing a baseline CNN with transfer learning models (ResNet50, EfficientNet-B0)


========= Repository Structure =========
.
├── Project/
    ├── CS184A_Group 5_Final project.ipynb
    │
    │
    ├── Data/   		                # 41 original pictures
    ├── Output/
    │	 ├── WT							# WT output
    │	 ├── TG							# TG output
    │
    ├── requirements.txt
    ├── README.md

========= Environment Setup =========
1. Python Version

Python 3.9 – 3.10 recommended

Tested on Google Colab (GPU)

2. Install Dependencies

Run the following command:

	pip install -r requirements.txt


========= Data Organization =========
Full Dataset (Used in Training)

If you have access to the full dataset:

Data/
├── image01.tif
├── image02.tif
├── ...
├── image41.tif


Upload this folder to Google Drive and update the path in the notebook:

DATA_ROOT = "/content/drive/MyDrive/Data"

Sample Dataset (For Demo)


========= How to Run the Project =========
Run on Google Colab (Recommended)

1. Open Google Colab
2. Upload notebooks from notebooks/
3. Enable GPU:
	Runtime -> Change runtime type -> Hardware accelerator -> GPU
4. Mount Google Drive:

	from google.colab import drive
	drive.mount('/content/drive')


Step 1: Cell Segmentation & Auto-labeling
Run Cell 3:
Segments individual cells from the 41 original images
Saves cropped cells into:
/content/drive/MyDrive/CS184A/Project/Output/
├── WT/
└── TG/

Step 2: Baseline CNN Training (Student Y Nhi)
Run all cells in and below:
# === Y Nhi's part

This includes:
Cell segmentation & auto-labeling (step 1 above)
Dataset loading
Baseline CNN training
Validation & testing
Confusion matrix & metrics

Best model is saved to:
/content/drive/MyDrive/CS184A/Project/baseline_best.pth

Step 3: Transfer Learning Models (Student Yasmeen)
Run:
# === Yasmeen Part ...

Models trained:
ResNet50 (ImageNet pretrained)
EfficientNet-B0 (ImageNet pretrained)

Best models saved as:
resnet50_best.pth
efficientnet_best.pth

Step 4: Evaluation & Comparison

Final comparison includes:
Accuracy
Precision
Recall
F1 Score
Confusion Matrix
Bar plot comparing Baseline vs Transfer Learning


========= Results Summary (Final Run) =========
Model	F1 Score
Baseline CNN	0.70
ResNet50	0.83
EfficientNet-B0	0.91

EfficientNet-B0 achieved the best overall performance.

========= requirements.txt =========
torch
torchvision
numpy
opencv-python
scikit-image
scikit-learn
matplotlib
pillow
cellpose==2.2.2

========= Reproducibility Notes =========

Results may vary slightly between runs due to random initialization

Best models are selected based on validation F1-score

Fixed train/validation/test splits are used

========= Authors & Contributions =========

Student Y Nhi Tran: Data preprocessing, cell segmentation, baseline CNN, evaluation

Student Yi Wang: Dataset acquisition, annotation guidance, validation, report writing

Student Yasmeen Soe: Transfer learning models, hyperparameter tuning, model comparison


========= Contact =========

For questions regarding the project or code execution, please contact the team via the GitHub repository.
