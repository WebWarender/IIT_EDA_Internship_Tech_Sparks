"""
06_analysis_forecasting.py
- Loads saved models / predictions and computes final metrics and plots.
- Produces simple visualizations: true vs predicted, residual histogram
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = r"C:\Users\Riya\IIT_EDA_Internship\Data\processed_features"
MODEL_DIR = r"C:\Users\Riya\IIT_EDA_Internship\models"

# Load validation portion used previously (assuming same split)
X = np.load(os.path.join(DATA_DIR, "X_seq.npy"))
y = np.load(os.path.join(DATA_DIR, "y_seq.npy")).ravel()
split = int(0.8 * len(X))
X_val = X[split:]; y_val = y[split:]

# Load LSTM predictions by reloading model and running inference (requires model.py)
import torch
from model import LSTMForecast
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lstm = LSTMForecast(input_dim=X.shape[2])
lstm.load_state_dict(torch.load(os.path.join(MODEL_DIR, "lstm_model.pt"), map_location=device))
lstm.to(device).eval()

with torch.no_grad():
    preds = []
    batch_size = 128
    for i in range(0, len(X_val), batch_size):
        xb = torch.tensor(X_val[i:i+batch_size], dtype=torch.float32).to(device)
        p = lstm(xb).cpu().numpy()
        preds.append(p)
preds = np.concatenate(preds)

# Metrics
mae = mean_absolute_error(y_val, preds)
rmse = mean_squared_error(y_val, preds, squared=False)
r2 = r2_score(y_val, preds)
print("Final LSTM Metrics -> MAE:", mae, "RMSE:", rmse, "R2:", r2)

# Plots
plt.figure(figsize=(10,4))
plt.plot(y_val[:500], label="True", alpha=0.8)
plt.plot(preds[:500], label="Predicted", alpha=0.7)
plt.legend(); plt.title("True vs Predicted (first 500 samples)")
plt.show()

plt.figure(figsize=(6,4))
resid = y_val - preds
plt.hist(resid, bins=50)
plt.title("Residual histogram")
plt.show()
