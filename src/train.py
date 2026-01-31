import torch
import torch.optim as optim
import numpy as np
import os
from model import TrafficGNN

# 1. SETUP & LOAD
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_load = np.load(os.path.join(base_path, 'data', 'processed', 'final_dataset.npz'))
data = torch.tensor(data_load['data'], dtype=torch.float32)
edge_index = torch.tensor(data_load['adjacency'], dtype=torch.long).t().contiguous()
num_nodes = data.shape[1]

# 2. INITIALIZE
model = TrafficGNN(num_nodes=num_nodes, input_dim=3)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss() # Back to MSE for peak-chasing

print("🚀 Retraining the 49% Accuracy Model...")

for epoch in range(100):
    model.train()
    total_loss = 0
    for t in range(600):
        optimizer.zero_grad()
        
        x = data[t]
        y = data[t+1][:, 0].view(num_nodes, 1)
        
        prediction = model(x, edge_index)
        loss = criterion(prediction, y)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if epoch % 10 == 0 or epoch == 99:
        print(f"Epoch {epoch:03d} | Avg MSE Loss: {total_loss/600:.6f}")

# 3. SAVE
torch.save(model.state_dict(), os.path.join(base_path, 'data', 'processed', 'model_weights.pth'))
print("✅ Previous Best Model Saved!")