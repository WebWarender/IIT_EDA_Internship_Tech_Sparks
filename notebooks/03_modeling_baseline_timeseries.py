# 03_modeling_baseline_timeseries.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np

# ---------------------------
# Load features and target
# ---------------------------
features = pd.read_csv(r"C:\Users\Riya\IIT_EDA_Internship\data\features.csv", index_col=0)
target = pd.read_csv(r"C:\Users\Riya\IIT_EDA_Internship\data\target.csv", index_col=0)

print("🔹 Original features shape:", features.shape)
print("🔹 Original target shape:", target.shape)

# ---------------------------
# Align target with features
# ---------------------------
target = target.reindex(features.index).ffill().bfill()

# Join into one dataset
data = features.join(target, how="inner")

# Drop rows where target is still missing
data = data.dropna(subset=["engagement"])

print("✅ Final dataset shape after alignment:", data.shape)

# ---------------------------
# Split into X and y
# ---------------------------
X = data.drop(columns=["engagement"])
y = data["engagement"]

# Safety check
if X.empty or y.empty:
    raise ValueError("❌ Dataset is still empty after alignment. Check your CSV files.")

# ---------------------------
# Train-test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print(f"✅ Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# ---------------------------
# Train Random Forest
# ---------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------------------
# Evaluate
# ---------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"📊 MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

# ---------------------------
# Save model
# ---------------------------
joblib.dump(model, r"C:\Users\Riya\IIT_EDA_Internship\models\baseline_forecast.pkl")
print("✅ Model saved at models/baseline_forecast.pkl")
