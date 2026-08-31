# 📂 Dataset Information & Instructions

<div align="center">

[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

</div>

## ⚠️ Important Note
Due to GitHub's file size limits, the actual image files and datasets are **not included** in this repository. You need to download them independently and place them in this folder before running the scripts.

## 🔗 Dataset Source
We use the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset from Kaggle.
👉 **[Download GTSRB Dataset Here](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)**

## 🚀 How to Setup the Data

### Method 1: Automatic Download (Recommended)
If you run the `train.ipynb` notebook provided in the `code/` folder, it uses the `kagglehub` library to automatically download and cache the dataset on your local machine or Google Colab environment.

### Method 2: Manual Download
If you want to evaluate the models locally using the Python scripts, please follow these steps:
1. Go to the dataset link above and click **Download (312 MB)**.
2. Extract the downloaded `.zip` archive.
3. Move all the extracted folders and files directly into this `data/` folder.

### 📁 Expected Directory Structure
Based on the official Kaggle extraction, your `data` folder must look exactly like this:

```text
data/
 ┣ 📂 Meta/              # Contains metadata images for the traffic sign classes
 ┣ 📂 Test/              # Contains 12,630 unseen testing images
 ┣ 📂 Train/             # Contains 43 subfolders (0 to 42) with training images
 ┣ 📜 Meta.csv           # Metadata mapping class IDs to their visual characteristics
 ┣ 📜 Test.csv           # Ground truth labels and exact relative paths for the test set
 ┣ 📜 Train.csv          # Ground truth labels and exact relative paths for the training set
 ┗ 📜 README.md          # This instruction file
