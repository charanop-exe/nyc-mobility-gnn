import torch
import torch.optim as optim
import numpy as np
import os
from model import TrafficGNN

# 1. SETUP PATHS & LOAD DATA
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_path, 'data', 'processed', 'final_dataset.npz')

data = np.load(data_path)
# Convert to tensor
demand_raw = torch.tensor(data['demand'], dtype=torch.float32)

# --- CORRECTION: NORMALIZATION ---
# Scaling data between 0 and 1 helps the model learn faster and prevents flatlining
max_val = demand_raw.max()
demand = demand_raw / max_val
print(f"⚖️ Data Normalized. Max pickups in one hour: {max_val.item()}")

# Adjacency setup
adj_data = data['adjacency']
edge_index = torch.tensor(adj_data[:, :2], dtype=torch.long).t().contiguous()

# --- CORRECTION: DYNAMIC NODE DETECTION ---
num_nodes = demand.shape[1] 
print(f"📊 Detected {num_nodes} nodes in dataset.")

# 2. INITIALIZE MODEL
# We use input_dim=1 because we only have one feature: 'demand'
model = TrafficGNN(num_nodes=num_nodes, input_dim=1, output_dim=1)
optimizer = optim.Adam(model.parameters(), lr=0.001) # Learning rate adjusted for stability
criterion = torch.nn.MSELoss()

print("🚀 Starting Training for 100 Epochs...")

# 3. TRAINING LOOP
epochs = 100 
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    # Training on the first 600 hours
    for t in range(600):
        optimizer.zero_grad()
        
        # Current hour (input) and Next hour (target)
        x = demand[t].view(num_nodes, 1)
        y = demand[t+1].view(num_nodes, 1)
        
        # Forward pass
        prediction = model(x, edge_index)
        loss = criterion(prediction, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:03d} | Avg Loss: {total_loss/600:.6f}")

# 4. SAVE THE TRAINED MODEL
weights_path = os.path.join(base_path, 'data', 'processed', 'model_weights.pth')
torch.save(model.state_dict(), weights_path)
print(f"✅ Training Complete! Weights saved to: {weights_path}")