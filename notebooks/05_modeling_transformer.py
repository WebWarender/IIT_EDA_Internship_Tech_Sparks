"""
05_modeling_transformer.py
- Lightweight Transformer encoder for time-series forecasting
- Uses PyTorch's nn.TransformerEncoder
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error

DATA_DIR = r"C:\Users\Riya\IIT_EDA_Internship\Data\processed_features"
MODEL_DIR = r"C:\Users\Riya\IIT_EDA_Internship\models"
os.makedirs(MODEL_DIR, exist_ok=True)

X = np.load(os.path.join(DATA_DIR, "X_seq.npy"))
y = np.load(os.path.join(DATA_DIR, "y_seq.npy")).ravel()

# simple train/val split
N = len(X)
split = int(0.8 * N)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

class TransformerForecast(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.reg_head = nn.Sequential(nn.Linear(d_model, d_model//2), nn.ReLU(), nn.Linear(d_model//2, 1))

    def forward(self, x):
        # x: batch, seq_len, input_dim
        h = self.input_proj(x)            # -> d_model
        h2 = self.transformer(h)          # -> (batch, seq_len, d_model)
        out = h2[:, -1, :]                # last token representation
        return self.reg_head(out).squeeze(-1)

# dataset loaders
import torch
from torch.utils.data import TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

model = TransformerForecast(input_dim=X.shape[2]).to(device)
opt = torch.optim.Adam(model.parameters(), lr=5e-4)
loss_fn = nn.MSELoss()
best_mae = float('inf')

for epoch in range(1, 31):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss = loss_fn(preds, yb)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    ys_true, ys_pred = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            ys_true.append(yb.cpu().numpy()); ys_pred.append(preds.cpu().numpy())
    ys_true = np.concatenate(ys_true)
    ys_pred = np.concatenate(ys_pred)
    mae = mean_absolute_error(ys_true, ys_pred)
    print(f"Epoch {epoch}: Val MAE {mae:.5f}")
    if mae < best_mae:
        best_mae = mae
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "transformer_model.pt"))
        print("Saved best transformer (MAE)", best_mae)
