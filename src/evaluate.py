# src/evaluate.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from model import TrafficGNN

# --------------------------------------------------
# 1. PATHS & DEVICE
# --------------------------------------------------
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_path, 'data', 'processed', 'final_dataset.npz')
model_path = os.path.join(base_path, 'data', 'processed', 'model_weights.pth')
output_dir = os.path.join(base_path, 'data', 'processed')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------
data = np.load(data_path)

X = torch.tensor(data['X'], dtype=torch.float32).to(device)   # [samples, T, N, F]
Y = torch.tensor(data['Y'], dtype=torch.float32).to(device)   # [samples, N]
edge_index = torch.tensor(data['adjacency'], dtype=torch.long).t().contiguous().to(device)
max_val = data['max_val']

num_zones = Y.shape[1]
print(f"📦 Loaded evaluation data | Zones: {num_zones}")

# --------------------------------------------------
# 3. LOAD MODEL
# --------------------------------------------------
model = TrafficGNN(input_dim=3, hidden_dim=32).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("✅ Model loaded successfully")

# --------------------------------------------------
# 4. RUN EVALUATION (ONE SAMPLE)
# --------------------------------------------------
# Pick a test index safely (not training example 0)
test_index = int(len(X) * 0.8)

with torch.no_grad():
    prediction = model(X[test_index:test_index+1], edge_index)  # [1, N]

# Convert back to numpy & rescale
y_pred = prediction.squeeze(0).cpu().numpy() * max_val
y_true = Y[test_index].cpu().numpy() * max_val

# --------------------------------------------------
# 5. SAVE RESULTS (IMPORTANT FOR METRICS)
# --------------------------------------------------
np.save(os.path.join(output_dir, 'y_pred.npy'), y_pred)
np.save(os.path.join(output_dir, 'y_true.npy'), y_true)

print("💾 Saved y_pred.npy and y_true.npy")

# --------------------------------------------------
# 6. PLOT RESULTS (FIRST 60 ZONES)
# --------------------------------------------------
zones_to_plot = 60

plt.figure(figsize=(14, 6))
plt.plot(
    y_true[:zones_to_plot],
    label="Actual Demand",
    marker='o',
    linewidth=2
)
plt.plot(
    y_pred[:zones_to_plot],
    label="Predicted Demand",
    linestyle='--',
    marker='x',
    linewidth=2
)

plt.title("NYC Taxi Demand Prediction (Zone-wise)", fontsize=14)
plt.xlabel("Taxi Zone Index", fontsize=12)
plt.ylabel("Number of Pickups", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)

plot_path = os.path.join(output_dir, 'evaluation_plot.png')
plt.savefig(plot_path)
plt.show()

print(f"📊 Evaluation plot saved to: {plot_path}")
print("✅ Evaluation complete")
