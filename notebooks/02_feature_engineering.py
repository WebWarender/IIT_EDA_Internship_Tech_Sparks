# 02_feature_engineering.py

import os
import pandas as pd
import numpy as np

# Paths
DATA_DIR = r"C:\Users\Riya\IIT_EDA_Internship\Data"
FEATURES_PATH = os.path.join(DATA_DIR, "features.csv")
TARGET_PATH = os.path.join(DATA_DIR, "target.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "processed_features")

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ---------------------------
# 1. Load data
# ---------------------------
features = pd.read_csv(FEATURES_PATH)
target = pd.read_csv(TARGET_PATH)

print(f"✅ Loaded features: {features.shape}")
print(f"✅ Loaded target: {target.shape}")

# ---------------------------
# 2. Clean features
# ---------------------------
features = features.drop_duplicates().ffill().bfill().fillna(0)
numeric_features = features.select_dtypes(include=[np.number])
print(f"✅ Numeric Features: {numeric_features.shape}")

# ---------------------------
# 3. Clean target
# ---------------------------
target = target.ffill().bfill()

# If first column is datetime → convert to numeric index
if np.issubdtype(target.iloc[:, 0].dtype, np.datetime64) or "date" in target.columns[0].lower():
    y = pd.to_datetime(target.iloc[:, 0]).astype(np.int64) // 10**9  # convert to Unix seconds
    print("⏱ Converted datetime target → numeric (Unix timestamp)")
else:
    # Otherwise, use the 2nd column if it's numeric
    if target.shape[1] > 1:
        y = pd.to_numeric(target.iloc[:, 1], errors="coerce").fillna(0)
        print("🎯 Using 2nd column of target as numeric")
    else:
        y = pd.to_numeric(target.iloc[:, 0], errors="coerce").fillna(0)
        print("🎯 Using 1st column of target (numeric conversion)")

# ---------------------------
# 4. Align feature-target lengths
# ---------------------------
min_len = min(len(numeric_features), len(y))
numeric_features = numeric_features.iloc[:min_len]
y = y.iloc[:min_len] if isinstance(y, pd.Series) else y[:min_len]

# ---------------------------
# 5. Merge into processed DataFrame
# ---------------------------
processed = numeric_features.copy()
processed["target"] = y.values

print(f"✨ Features engineered: {processed.shape}")

# ---------------------------
# 6. Save outputs
# ---------------------------
np.save(os.path.join(OUTPUT_PATH, "X.npy"), numeric_features.values.astype(np.float32))
np.save(os.path.join(OUTPUT_PATH, "y.npy"), y.values.astype(np.float32))
processed.to_csv(os.path.join(OUTPUT_PATH, "processed.csv"), index=False)

print(f"✅ Saved to {OUTPUT_PATH}")
