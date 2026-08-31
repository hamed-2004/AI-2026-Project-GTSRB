import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
from sklearn.metrics import accuracy_score

# ==========================================
# 1. Configuration & Smart Paths
# ==========================================
# Automatically detect the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMG_HEIGHT = 32
IMG_WIDTH = 32

# Create absolute paths based on the script's location
TEST_CSV_PATH = os.path.join(BASE_DIR, 'Test.csv')
MLP_MODEL_PATH = os.path.join(BASE_DIR, 'best_mlp.keras')
CNN_MODEL_PATH = os.path.join(BASE_DIR, 'MyCNNModel.h5')

# ==========================================
# 2. Load Models
# ==========================================
print("Loading models...")
try:
    mlp_model = load_model(MLP_MODEL_PATH)
    cnn_model = load_model(CNN_MODEL_PATH)
    print("Both MLP and CNN models loaded successfully!")
except Exception as e:
    print(f"\n[ERROR] Could not load models. Reason: {e}")
    print(f"Please ensure both '{MLP_MODEL_PATH}' and '{CNN_MODEL_PATH}' exist.")
    exit()

# ==========================================
# 3. Load and Preprocess Test Data
# ==========================================
print(f"\nLoading test data from: {TEST_CSV_PATH}")
try:
    test_data = pd.read_csv(TEST_CSV_PATH)
    labels = test_data['ClassId'].values
    img_paths = test_data['Path'].values
except FileNotFoundError:
    print(f"[ERROR] {TEST_CSV_PATH} not found.")
    exit()

X_test = []
y_test = []

print("Processing test images...")
for i, relative_path in enumerate(img_paths):
    try:
        # Join BASE_DIR with the relative path inside the CSV
        full_img_path = os.path.join(BASE_DIR, relative_path)
        
        # Open image, resize to 32x32, and convert to numpy array
        img = Image.open(full_img_path)
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.array(img)
        
        X_test.append(img_array)
        y_test.append(labels[i])
    except Exception as e:
        # Skip corrupted or missing images silently
        pass 

# Convert to numpy arrays and normalize pixels to [0, 1] range
X_test = np.array(X_test) / 255.0
y_test = np.array(y_test)
print(f"Successfully processed {len(X_test)} valid test images.")

if len(X_test) == 0:
    print("[ERROR] No images were loaded. Check if the 'Test' folder exists next to this script.")
    exit()

# ==========================================
# 4. Evaluate Models on Unseen Test Data
# ==========================================
print("\nEvaluating MLP model...")
# FIX: Flatten the (32, 32, 3) images into a 1D vector of 3072 for MLP
X_test_mlp = X_test.reshape(len(X_test), -1)
mlp_predictions = np.argmax(mlp_model.predict(X_test_mlp, verbose=0), axis=1)
mlp_acc = accuracy_score(y_test, mlp_predictions)

print("Evaluating CNN model...")
# CNN uses the original 3D shape images
cnn_predictions = np.argmax(cnn_model.predict(X_test, verbose=0), axis=1)
cnn_acc = accuracy_score(y_test, cnn_predictions)

# ==========================================
# 5. Print Final Results and Plot
# ==========================================
print("\n" + "="*50)
print("FINAL TEST ACCURACY ON UNSEEN DATA")
print("="*50)
print(f"MLP Model ('best_mlp.keras'): {mlp_acc * 100:.2f}%")
print(f"CNN Model ('MyCNNModel.h5'):  {cnn_acc * 100:.2f}%")
print("="*50)

# Generate a professional Bar Chart for documentation
models = ['MLP Model', 'CNN Model']
accuracies = [mlp_acc * 100, cnn_acc * 100]

plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracies, color=['#1f77b4', '#ff7f0e'])
plt.ylim(0, 100)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('MLP vs CNN - Final Test Set Evaluation', fontsize=14, fontweight='bold')

# Add percentage text on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, 
             f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

# Save the plot securely in the SAME directory
chart_path = os.path.join(BASE_DIR, 'Final_Models_Comparison.png')
plt.savefig(chart_path, dpi=300, bbox_inches='tight')
print(f"\nComparison chart saved successfully at:\n{chart_path}")