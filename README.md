# 🚦 German Traffic Sign Recognition Benchmark (GTSRB): CNN vs. MLP

<div align="center">

[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)]()
[![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)]()
[![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)]()

</div>

---

## 📌 Project Overview
This project was designed and implemented as a comprehensive deep learning image classification task. The primary objective is to develop an intelligent computer vision system capable of automatically recognizing and classifying German traffic signs under various conditions.

Using the standard **German Traffic Sign Recognition Benchmark (GTSRB)** dataset, traffic signs are classified into **43 distinct categories**. The dataset presents significant real-world challenges, including:
- 🌤️ **Environmental Diversity:** Extreme lighting variations (heavy shadows, glare, or direct sunlight).
- 📐 **Geometric Variations:** Different viewing angles, motion blur, and physical damage to the signs.
- ⚖️ **Class Imbalance:** A massive disparity in the number of training samples between highly common classes (e.g., speed limits) and rare ones.

---

## 🧠 Architectures & Learning Strategies
To achieve the highest accuracy and perform a robust comparative analysis, two distinct neural network architectures were designed, trained, and evaluated:

1. **Multi-Layer Perceptron (MLP) (Baseline):** A deep dense network with 3 hidden layers (512, 256, and 128 neurons). This model flattens the 32x32x3 images into 1D arrays of 3072 features, resulting in a massive parameter count (over 1.7 million).
2. **Convolutional Neural Network (CNN) (Proposed):** A highly efficient 3-block convolutional architecture designed to preserve the 2D spatial structure of images. By utilizing local filters (kernels) and MaxPooling layers, it extracts hierarchical features (from simple edges to complex shapes) with only **~350K parameters** (almost 5 times lighter than the MLP).

### 💡 Technical Highlights & Strategies:
- 🛡️ **Preventing Data Leakage:** The data was split into Training (80%) and Validation (20%) sets using **Stratified Sampling** to ensure the imbalanced class distribution is perfectly maintained across both sets.
- 📉 **Robust Training Callbacks:**
  - `EarlyStopping`: Monitored `val_loss` with a patience of 10 epochs to prevent overfitting.
  - `ModelCheckpoint`: Continuously saved the best model weights based on the lowest validation loss.
  - `ReduceLROnPlateau`: Reduced the learning rate by a factor of 0.5 when `val_loss` plateaued to help the optimizer escape local minima.

---

## 📊 Models & Results
The CNN architecture demonstrated absolute superiority by preserving spatial relationships, avoiding the fatal flaw of the MLP which loses geometric structure during the flattening phase.

| Architecture | Total Parameters | Validation Accuracy | Unseen Test Accuracy |
|-------------|-----------------|---------------------|----------------------|
| **MLP** (Baseline) | ~1,746,731 | 98.66% | **89.47%** |
| **CNN** (Proposed) | ~356,939 | 99.63% | **97.26%** |

---

## 📂 Repository Structure
This repository is cleanly organized following Clean Architecture principles to separate source code, data, models, and documentation:

```text
📦 Traffic-Sign-Recognition-CNN-vs-MLP
 ┣ 📂 code/              # Python scripts (main.py) and Jupyter notebooks (train.ipynb)
 ┣ 📂 data/              # Test.csv and instructions/scripts to download the GTSRB dataset
 ┣ 📂 docs/              # Final project report PDF and presentation slides
 ┣ 📂 media/             # Training history plots, confusion matrices, and charts
 ┣ 📂 models/            # Saved pre-trained models (best_mlp.keras, MyCNNModel.h5)
 ┣ 📜 .gitignore         # Ignores large zip files, cache, and virtual environments
 ┣ 📜 requirements.txt   # List of required Python libraries
 ┗ 📜 README.md          # Main project documentation and overview
