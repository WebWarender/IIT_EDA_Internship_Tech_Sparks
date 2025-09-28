# 04_modeling_LSTM_GRU.py
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

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

# Dataset
dataset = TensorDataset(X_tensor, y_tensor)

# Split train/val
N = len(dataset)
split = int(0.8 * N)
train_ds, val_ds = random_split(dataset, [split, N - split])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

# -----------------------------
# Define LSTM model
# -----------------------------
class LSTMForecast(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # take last hidden state
        return self.fc(out)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X.shape[2]
model = LSTMForecast(input_dim=input_dim).to(device)
opt = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# -----------------------------
# Training loop
# -----------------------------
best_mae = float("inf")
EPOCHS = 20

for epoch in range(1, EPOCHS + 1):
    # Training
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss = criterion(preds, yb)
        opt.zero_grad()
        loss.backward()
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

    mae = mean_absolute_error(ys_true, ys_pred)
    rmse = mean_squared_error(ys_true, ys_pred, squared=False)
    r2 = r2_score(ys_true, ys_pred)

    print(f"Epoch {epoch}: TrainLoss={np.mean(train_losses):.5f} | "
          f"Val MAE={mae:.5f} | RMSE={rmse:.5f} | R2={r2:.5f}")

    # Save best model
    if mae < best_mae:
        best_mae = mae
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "lstm_model.pt"))
        print("✅ Saved best model with MAE:", best_mae)
