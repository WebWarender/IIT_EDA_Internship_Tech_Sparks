"""
06_analysis_forecasting.py
- Loads saved models / predictions and computes final metrics and plots.
- Evaluates both LSTM and Transformer models if available.
- Produces visualizations: true vs predicted, residuals, and metric comparison.
- (Adjusted to avoid sklearn mean_squared_error `squared=` keyword compatibility issues.)
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------
# Paths
# ---------------------------------------------------
DATA_DIR = r"C:\Users\Riya\IIT_EDA_Internship\Data\processed_features"
MODEL_DIR = r"C:\Users\Riya\IIT_EDA_Internship\models"

# ---------------------------------------------------
# Load validation data
# ---------------------------------------------------
X = np.load(os.path.join(DATA_DIR, "X_seq.npy"))
y = np.load(os.path.join(DATA_DIR, "y_seq.npy")).ravel()

# Handle 2D -> 3D reshape
if X.ndim == 2:
    X = X.reshape(X.shape[0], 1, X.shape[1])
    print(f"Reshaped X to {X.shape} for sequence modeling.")

split = int(0.8 * len(X))
X_val = X[split:]
y_val = y[split:]

# Clean data in case of NaN or inf
X_val = np.nan_to_num(X_val)
y_val = np.nan_to_num(y_val)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------
# Define Models (LSTM matches trained architecture)
# ---------------------------------------------------
class LSTMForecast(torch.nn.Module):
    """Matches the LSTM architecture used in training (hidden_dim=64, num_layers=1)."""
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class TransformerForecast(torch.nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3,
                 dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = torch.nn.Linear(input_dim, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.reg_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        h = self.input_proj(x)
        h2 = self.transformer(h)
        out = h2[:, -1, :]
        return self.reg_head(out).squeeze(-1)

# ---------------------------------------------------
# Helper function for model evaluation
# ---------------------------------------------------
def evaluate_model(model, model_name, model_path):
    if not os.path.exists(model_path):
        print(f"⚠️  {model_name} model not found at {model_path}. Skipping...")
        return None, None, None, None

    print(f"\nEvaluating {model_name} model...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    preds = []
    with torch.no_grad():
        batch_size = 128
        for i in range(0, len(X_val), batch_size):
            xb = torch.tensor(X_val[i:i+batch_size], dtype=torch.float32).to(device)
            p = model(xb).cpu().numpy()
            preds.append(p)
    preds = np.concatenate(preds)

    # compute metrics (avoid using `squared=` kw to keep compatibility)
    mae = mean_absolute_error(y_val, preds)
    mse = mean_squared_error(y_val, preds)       # returns MSE
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_val, preds)

    print(f"{model_name} Results:")
    print(f"  MAE  = {mae:.5f}")
    print(f"  RMSE = {rmse:.5f}")
    print(f"  R²   = {r2:.5f}")

    return preds, mae, rmse, r2

# ---------------------------------------------------
# Evaluate available models
# ---------------------------------------------------
input_dim = X.shape[2]
results = {}

lstm_model_path = os.path.join(MODEL_DIR, "lstm_model.pt")
transformer_model_path = os.path.join(MODEL_DIR, "transformer_model.pt")

# Evaluate LSTM
lstm_preds, lstm_mae, lstm_rmse, lstm_r2 = evaluate_model(
    LSTMForecast(input_dim=input_dim), "LSTM", lstm_model_path
)
if lstm_preds is not None:
    results["LSTM"] = {"preds": lstm_preds, "MAE": lstm_mae, "RMSE": lstm_rmse, "R2": lstm_r2}

# Evaluate Transformer
trans_preds, trans_mae, trans_rmse, trans_r2 = evaluate_model(
    TransformerForecast(input_dim=input_dim), "Transformer", transformer_model_path
)
if trans_preds is not None:
    results["Transformer"] = {"preds": trans_preds, "MAE": trans_mae, "RMSE": trans_rmse, "R2": trans_r2}

# ---------------------------------------------------
# Visualizations
# ---------------------------------------------------
if results:
    plt.figure(figsize=(12, 5))
    plt.plot(y_val[:500], label="True", alpha=0.8)

    for name, res in results.items():
        plt.plot(res["preds"][:500], label=f"{name} Predicted", alpha=0.7)

    plt.legend()
    plt.title("True vs Predicted (first 500 samples)")
    plt.xlabel("Samples")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()

    # Residual plots
    plt.figure(figsize=(6, 4))
    for name, res in results.items():
        plt.hist(y_val - res["preds"], bins=50, alpha=0.6, label=f"{name} Residuals")
    plt.legend()
    plt.title("Residual Histogram Comparison")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Metric comparison
    models = list(results.keys())
    mae_scores = [results[m]["MAE"] for m in models]
    rmse_scores = [results[m]["RMSE"] for m in models]
    r2_scores = [results[m]["R2"] for m in models]

    x = np.arange(len(models))
    plt.figure(figsize=(8, 5))
    plt.bar(x - 0.25, mae_scores, width=0.25, label="MAE")
    plt.bar(x, rmse_scores, width=0.25, label="RMSE")
    plt.bar(x + 0.25, r2_scores, width=0.25, label="R²")
    plt.xticks(x, models)
    plt.title("Model Metric Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n✅ Analysis complete. Visualizations generated successfully.")
else:
    print("⚠️ No trained models found. Please train LSTM or Transformer first.")
