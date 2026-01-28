import torch
import torch.optim as optim
import numpy as np
import os
from model import TrafficGNN

# 1. SETUP PATHS & LOAD DATA
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_path, 'data', 'processed', 'final_dataset.npz')

data = np.load(data_path)
# Demand shape: (Hours, 263)
demand = torch.tensor(data['demand'], dtype=torch.float32)
# Adjacency shape: (Edges, 3) -> We need only columns 0 and 1 for edge_index
adj_data = data['adjacency']
edge_index = torch.tensor(adj_data[:, :2], dtype=torch.long).t().contiguous()

# 2. INITIALIZE MODEL# 1. DYNAMICALLY DETECT THE NUMBER OF NODES
# Instead of hardcoding 263, we look at the actual data shape
num_nodes = demand.shape[1] 
print(f"📊 Detected {num_nodes} nodes in dataset.")

# 2. INITIALIZE MODEL WITH ACTUAL NODE COUNT
model = TrafficGNN(num_nodes=num_nodes, input_dim=1, output_dim=1)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

print("🚀 Starting Training...")

# 3. UPDATED TRAINING LOOP
for epoch in range(50):
    model.train()
    total_loss = 0
    
    for t in range(600):
        optimizer.zero_grad()
        
        # Use the dynamic num_nodes variable here
        x = demand[t].view(num_nodes, 1)
        y = demand[t+1].view(num_nodes, 1)
        
        prediction = model(x, edge_index)
        loss = criterion(prediction, y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Avg Loss: {total_loss/600:.4f}")
        
# 4. SAVE THE TRAINED MODEL
torch.save(model.state_dict(), os.path.join(base_path, 'data', 'processed', 'model_weights.pth'))
print("✅ Training Complete! Weights saved.")