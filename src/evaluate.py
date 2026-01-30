import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from model import TrafficGNN

# 1. SETUP PATHS & LOAD DATA
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_path, 'data', 'processed', 'final_dataset.npz')
weights_path = os.path.join(base_path, 'data', 'processed', 'model_weights.pth')

data = np.load(data_path)
demand_raw = torch.tensor(data['demand'], dtype=torch.float32)

# --- THE KEY ADDITION: DEFINING MAX_VAL ---
max_val = demand_raw.max() 
demand = demand_raw / max_val  # Normalize just like in training
# ------------------------------------------

adj_data = data['adjacency']
edge_index = torch.tensor(adj_data[:, :2], dtype=torch.long).t().contiguous()

# 2. LOAD TRAINED MODEL
num_nodes = demand.shape[1]
model = TrafficGNN(num_nodes=num_nodes, input_dim=1, output_dim=1)
model.load_state_dict(torch.load(weights_path))
model.eval()

print(f"📈 Evaluating with max_val: {max_val:.2f}")

# 3. RUN PREDICTION
test_hour = 700 
x = demand[test_hour].view(num_nodes, 1)

with torch.no_grad():
    y_pred = model(x, edge_index).numpy()

# --- RE-SCALE FOR THE GRAPH ---
y_pred_real = y_pred * max_val.item()
y_true_real = demand_raw[test_hour+1].numpy()
# ------------------------------

# 4. VISUALIZE RESULTS
plt.figure(figsize=(12, 6))
plt.plot(y_true_real[:50], label='Actual Demand (NYC)', color='blue', marker='o')
plt.plot(y_pred_real[:50], label='AI Prediction', color='red', linestyle='--', marker='x')
plt.title(f'NYC Taxi Demand Prediction - Hour {test_hour+1}')
plt.xlabel('Taxi Zone ID')
plt.ylabel('Number of Pickups')
plt.legend()
plt.grid(True)

# Save the plot
plot_path = os.path.join(base_path, 'data', 'processed', 'prediction_results.png')
plt.savefig(plot_path)
print(f"✅ Success! Plot saved to: {plot_path}")
plt.show()