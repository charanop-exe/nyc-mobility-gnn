import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from model import TrafficGNN

# 1. SETUP PATHS
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_path, 'data', 'processed', 'final_dataset.npz')
weights_path = os.path.join(base_path, 'data', 'processed', 'model_weights.pth')

# 2. LOAD DATA
data_load = np.load(data_path)
data = torch.tensor(data_load['data'], dtype=torch.float32)
edge_index = torch.tensor(data_load['adjacency'], dtype=torch.long).t().contiguous()
max_val = data_load['max_val']

# 3. INITIALIZE & LOAD MODEL
num_nodes = data.shape[1]
model = TrafficGNN(num_nodes=num_nodes, input_dim=3)
model.load_state_dict(torch.load(weights_path))
model.eval()

print(f"📊 Evaluating the 49% GNN model on {num_nodes} zones...")

# 4. PREDICT
test_hour = 700 
# Features: [Demand, Hour, Day]
x = data[test_hour] 

with torch.no_grad():
    # In this version, model returns a single tensor
    prediction = model(x, edge_index).numpy().flatten()

# 5. RE-SCALE
y_pred_real = prediction * max_val
y_true_real = data[test_hour+1][:, 0].numpy() * max_val

# 6. PLOT RESULTS
plt.figure(figsize=(15, 6))
plt.plot(y_true_real[:60], label='Actual NYC Demand', color='#1f77b4', marker='o', alpha=0.8)
plt.plot(y_pred_real[:60], label='GNN AI Prediction', color='#d62728', linestyle='--', marker='x')

plt.title(f'GNN Result: NYC Taxi Demand Prediction (Hour {test_hour})', fontsize=14)
plt.xlabel('Taxi Zone Index', fontsize=12)
plt.ylabel('Number of Pickups', fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.5)

# Save the plot
plt.savefig(os.path.join(base_path, 'data', 'processed', 'gnn_49pct_eval.png'))
print("✅ Evaluation Complete! Plot saved to data/processed.")
plt.show()