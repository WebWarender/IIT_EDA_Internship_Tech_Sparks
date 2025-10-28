# ---------------------------------------------
# 04_modeling_LSTM_GRU.py (NaN-safe version)
# ---------------------------------------------
import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = r"C:\Users\Riya\IIT_EDA_Internship\Data\processed_features"
MODEL_DIR = r"C:\Users\Riya\IIT_EDA_Internship\models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
X = np.load(os.path.join(DATA_DIR, "X_seq.npy"))
y = np.load(os.path.join(DATA_DIR, "y_seq.npy")).ravel()
print("Loaded X, y:", X.shape, y.shape)

# -----------------------------
# Check for NaNs / infs
# -----------------------------
if np.isnan(X).any() or np.isinf(X).any():
    print("⚠️ Found NaN/Inf in X — replacing with 0")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

if np.isnan(y).any() or np.isinf(y).any():
    print("⚠️ Found NaN/Inf in y — replacing with 0")
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

# Optional normalization (recommended)
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

# -----------------------------
# Reshape for LSTM/GRU input
# -----------------------------
X = X.reshape(X.shape[0], 1, X.shape[1])
input_dim = X.shape[2]
print("Reshaped X:", X.shape)

# -----------------------------
# Tensor setup
# -----------------------------
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
dataset = TensorDataset(X_tensor, y_tensor)

split = int(0.8 * len(dataset))
train_ds, val_ds = random_split(dataset, [split, len(dataset) - split])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

# -----------------------------
# Models
# -----------------------------
class LSTMForecast(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class GRUForecast(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# -----------------------------
# Training function
# -----------------------------
def train_model(model, model_name, train_loader, val_loader, device, epochs=20):
    opt = optim.Adam(model.parameters(), lr=5e-4)  # slightly smaller LR
    criterion = nn.MSELoss()
    best_mae = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent NaN gradients
            opt.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        ys_true, ys_pred = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                ys_true.append(yb.cpu().numpy())
                ys_pred.append(preds.cpu().numpy())

        ys_true = np.concatenate(ys_true).ravel()
        ys_pred = np.concatenate(ys_pred).ravel()

        # Check for NaNs before metrics
        if np.isnan(ys_pred).any():
            print(f"⚠️ NaN detected in predictions at epoch {epoch}! Skipping metrics.")
            continue

        mae = mean_absolute_error(ys_true, ys_pred)
        rmse = mean_squared_error(ys_true, ys_pred, squared=False)
        r2 = r2_score(ys_true, ys_pred)

        print(f"{model_name} | Epoch {epoch:02d}: "
              f"TrainLoss={np.mean(train_losses):.5f} | "
              f"Val MAE={mae:.5f} | RMSE={rmse:.5f} | R2={r2:.5f}")

        if mae < best_mae:
            best_mae = mae
            save_path = os.path.join(MODEL_DIR, f"{model_name.lower()}_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best {model_name} model (MAE={best_mae:.5f})")

    print(f"Training finished for {model_name}. Best MAE: {best_mae:.5f}\n")


# -----------------------------
# Run training for LSTM & GRU
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

lstm_model = LSTMForecast(input_dim=input_dim).to(device)
train_model(lstm_model, "LSTM", train_loader, val_loader, device)

gru_model = GRUForecast(input_dim=input_dim).to(device)
train_model(gru_model, "GRU", train_loader, val_loader, device)

print("✅ Both LSTM and GRU training complete.")
