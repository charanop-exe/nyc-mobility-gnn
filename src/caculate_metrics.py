import torch
import numpy as np
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

num_nodes = data.shape[1]
# Ensure input_dim=3 to match your Demand, Hour, Day features
model = TrafficGNN(num_nodes=num_nodes, input_dim=3)
model.load_state_dict(torch.load(weights_path))
model.eval()

# 3. CALCULATE ERROR OVER TEST SET
all_mae = []
all_actuals = []

print("🧪 Evaluating 3-Feature GNN performance...")

with torch.no_grad():
    for t in range(601, 743):
        x = data[t]
        y_true = data[t+1][:, 0].numpy() * max_val
        
        # FIX: Removed ', h' because your current model.forward() only takes (x, edge_index)
        prediction = model(x, edge_index)
        
        y_pred = prediction.numpy().flatten() * max_val
        
        mae_hour = np.mean(np.abs(y_true - y_pred))
        all_mae.append(mae_hour)
        all_actuals.append(np.mean(y_true))

# 4. FINAL RESULTS
final_mae = np.mean(all_mae)
avg_demand = np.mean(all_actuals)
accuracy_pct = (1 - (final_mae / avg_demand)) * 100

print("-" * 30)
print(f"📊 PERFORMANCE REPORT")
print(f"Mean Absolute Error: {final_mae:.2f} pickups")
print(f"Average Demand:      {avg_demand:.2f} pickups")
print(f"Model Accuracy:      {accuracy_pct:.2f}%")
print("-" * 30)