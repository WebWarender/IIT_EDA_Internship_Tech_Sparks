import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =========================
# Transformer Forecast Model
# =========================
class TransformerForecast(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1, output_dim=1):
        super(TransformerForecast, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)   # average pooling across sequence length
        out = self.fc_out(x)
        return out

# =========================
# Load Data
# =========================
DATA_PATH = r"C:\Users\Riya\IIT_EDA_Internship\Data\processed_features"
X = np.load(os.path.join(DATA_PATH, "X.npy"))
y = np.load(os.path.join(DATA_PATH, "y.npy"))

print(f"Loaded X, y: {X.shape} {y.shape}")

# Reshape X to 3D for transformer (batch, seq_len, input_dim)
X = X.reshape(X.shape[0], 1, X.shape[1])

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

# =========================
# Model Setup
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerForecast(input_dim=X.shape[2]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# =========================
# Training Loop
# =========================
epochs = 20
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# =========================
# Evaluation
# =========================
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        y_true.extend(yb.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

mae = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
r2 = r2_score(y_true, y_pred)

print(f"\nTransformer Performance:\nMAE: {mae:.5f} | RMSE: {rmse:.5f} | R2: {r2:.5f}")

# =========================
# Save Model
# =========================
MODEL_PATH = r"C:\Users\Riya\IIT_EDA_Internship\models\transformer_model.pt"
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Transformer model saved to: {MODEL_PATH}")
