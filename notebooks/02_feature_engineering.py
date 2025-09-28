"""
02_feature_engineering.py
- Cleans features.csv and target.csv
- Removes non-numeric columns
- Standardizes features
- Aligns features and target by index length
- Saves X_seq.npy and y_seq.npy into Data/processed_features
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Base paths
DATA_DIR = r"C:\Users\Riya\IIT_EDA_Internship\Data"
OUTPUT_DIR = os.path.join(DATA_DIR, "processed_features")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# File paths
features_path = os.path.join(DATA_DIR, "features.csv")
target_path = os.path.join(DATA_DIR, "target.csv")

# Load datasets
features = pd.read_csv(features_path)
target = pd.read_csv(target_path)

print("Original Features shape:", features.shape)
print("Original Target shape:", target.shape)

# Keep only numeric columns from features
features_numeric = features.select_dtypes(include=[np.number])
print("Numeric Features shape:", features_numeric.shape)

# Keep only numeric columns from target
target_numeric = target.select_dtypes(include=[np.number])
print("Numeric Target shape:", target_numeric.shape)

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_numeric)

# Align target length with features
min_len = min(len(features_scaled), len(target_numeric))
X = np.array(features_scaled[:min_len], dtype=np.float32)
y = np.array(target_numeric.iloc[:min_len, 0].values, dtype=np.float32)  # take first numeric column

print("Final X shape:", X.shape)
print("Final y shape:", y.shape)

# Save processed numpy arrays
np.save(os.path.join(OUTPUT_DIR, "X_seq.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "y_seq.npy"), y)

print(f"✅ Saved processed features to {OUTPUT_DIR}")
