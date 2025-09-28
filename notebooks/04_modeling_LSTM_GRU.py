import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# Paths
# -------------------------------
DATA_DIR = r"C:\Users\Riya\IIT_EDA_Internship\Data\processed_features"

X = np.load(os.path.join(DATA_DIR, "X.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))

print(f"Loaded X, y: {X.shape}, {y.shape}")

# -------------------------------
# Reshape for LSTM/GRU
# -------------------------------
if X.ndim == 2:
    X = X.reshape(X.shape[0], 1, X.shape[1])  # (samples, seq_len=1, features)
print("Reshaped X for LSTM/GRU:", X.shape)

# -------------------------------
# Torch Dataset
# -------------------------------
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

dataset = TensorDataset(X_tensor, y_tensor)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

# -------------------------------
# LSTM / GRU Models
# -------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])

# -------------------------------
# Training Function
# -------------------------------
def train_model(model, train_loader, test_loader, epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            pred = model(Xb)
            preds.append(pred.cpu().numpy())
            true.append(yb.cpu().numpy())

    preds = np.concatenate(preds).ravel()
    true = np.concatenate(true).ravel()

    mae = mean_absolute_error(true, preds)
    rmse = mean_squared_error(true, preds, squared=False)
    r2 = r2_score(true, preds)

    print(f"MAE: {mae:.5f} | RMSE: {rmse:.5f} | R2: {r2:.5f}")
    return model

# -------------------------------
# Run Training
# -------------------------------
input_dim = X.shape[2]

print("\n🔹 Training LSTM Model")
lstm_model = LSTMModel(input_dim)
lstm_model = train_model(lstm_model, train_loader, test_loader, epochs=20)

print("\n🔹 Training GRU Model")
gru_model = GRUModel(input_dim)
gru_model = train_model(gru_model, train_loader, test_loader, epochs=20)

# -------------------------------
# Save Models in models/ folder
# -------------------------------
MODEL_DIR = r"C:\Users\Riya\IIT_EDA_Internship\models"
os.makedirs(MODEL_DIR, exist_ok=True)

torch.save(lstm_model.state_dict(), os.path.join(MODEL_DIR, "lstm_model.pt"))
torch.save(gru_model.state_dict(), os.path.join(MODEL_DIR, "gru_model.pt"))
print(f"✅ Models saved in: {MODEL_DIR}")
