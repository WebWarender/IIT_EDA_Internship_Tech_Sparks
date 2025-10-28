"""
05_modeling_transformer.py
- Lightweight Transformer encoder for time-series forecasting
- Uses PyTorch's nn.TransformerEncoder
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error

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
# Check & clean data
# -----------------------------
if np.isnan(X).any() or np.isinf(X).any():
    print("⚠️ Found NaN/Inf in X — replacing with 0")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
if np.isnan(y).any() or np.isinf(y).any():
    print("⚠️ Found NaN/Inf in y — replacing with 0")
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

# Normalize features (optional but helps Transformers)
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

# -----------------------------
# Reshape for Transformer input
# -----------------------------
# Treat each sample as a single timestep with multiple features
X = X.reshape(X.shape[0], 1, X.shape[1])
print("Reshaped X:", X.shape)
input_dim = X.shape[2]

# -----------------------------
# Train/Val Split
# -----------------------------
N = len(X)
split = int(0.8 * N)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# -----------------------------
# Transformer Forecast Model
# -----------------------------
class TransformerForecast(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        h = self.input_proj(x)
        h2 = self.transformer(h)
        out = h2[:, -1, :]  # last token representation
        return self.reg_head(out).squeeze(-1)

# -----------------------------
# Data Loaders
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_ds = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)
)
val_ds = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.float32)
)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

# -----------------------------
# Initialize model & optimizer
# -----------------------------
model = TransformerForecast(input_dim=input_dim).to(device)
opt = torch.optim.Adam(model.parameters(), lr=5e-4)
loss_fn = nn.MSELoss()
best_mae = float("inf")

# -----------------------------
# Training loop
# -----------------------------
for epoch in range(1, 31):
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss = loss_fn(preds, yb)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

    ys_true = np.concatenate(ys_true)
    ys_pred = np.concatenate(ys_pred)

    # Handle NaN predictions safely
    if np.isnan(ys_pred).any():
        print(f"⚠️ NaN detected in predictions at epoch {epoch}, skipping metrics.")
        continue

    mae = mean_absolute_error(ys_true, ys_pred)
    print(f"Epoch {epoch:02d}: TrainLoss={np.mean(train_losses):.5f} | Val MAE={mae:.5f}")

    if mae < best_mae:
        best_mae = mae
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "transformer_model.pt"))
        print(f"✅ Saved best transformer model (MAE={best_mae:.5f})")

print("✅ Transformer training complete.")
