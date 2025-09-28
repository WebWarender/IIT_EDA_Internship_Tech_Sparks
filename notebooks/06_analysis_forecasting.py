"""
06_analysis_forecasting.py
- Loads saved models / predictions and computes final metrics and plots.
- Produces simple visualizations: true vs predicted, residual histogram
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ======================
# Define LSTM Forecast Model
# ======================
class LSTMForecast(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # take last timestep
        return out.squeeze(-1)


# ======================
# Load Data
# ======================
DATA_DIR = r"C:\Users\Riya\IIT_EDA_Internship\Data\processed_features"
MODEL_DIR = r"C:\Users\Riya\IIT_EDA_Internship\models"

X = np.load(os.path.join(DATA_DIR, "X.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))

print(f"Loaded X, y: {X.shape}, {y.shape}")

# Reshape for LSTM [samples, timesteps=1, features]
X = X.reshape(X.shape[0], 1, X.shape[1])

# Train-test split (80/20)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# Load Model
# ======================
input_dim = X.shape[2]
model = LSTMForecast(input_dim=input_dim).to(device)

MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.pt")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ======================
# Prediction
# ======================
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_pred = model(X_test_tensor).cpu().numpy()

# ======================
# Evaluation
# ======================
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"✅ Evaluation Metrics:")
print(f"MAE: {mae:.5f}")
print(f"RMSE: {rmse:.5f}")
print(f"R2: {r2:.5f}")

# ======================
# Plot Actual vs Predicted
# ======================
def plot_forecast(y_true, y_pred, last_n=None):
    if last_n:  # optional zoom-in
        y_true = y_true[-last_n:]
        y_pred = y_pred[-last_n:]

    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="Actual", color="blue")
    plt.plot(y_pred, label="Predicted", color="red", linestyle="--")
    plt.title("Actual vs Predicted (LSTM Forecast)")
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

# Show full series and zoom-in view
plot_forecast(y_test, y_pred)          # full
plot_forecast(y_test, y_pred, last_n=200)  # zoom-in (last 200 points)
, r2_score

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

