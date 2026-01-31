# src/train.py
import torch
import numpy as np
import os
from model import TrafficGNN

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = np.load(os.path.join(base_path, 'data', 'processed', 'final_dataset.npz'))

X = torch.tensor(data['X'], dtype=torch.float32)
Y = torch.tensor(data['Y'], dtype=torch.float32)
edge_index = torch.tensor(data['adjacency'], dtype=torch.long).t().contiguous()

model = TrafficGNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
loss_fn = torch.nn.HuberLoss()

print("🚀 Training Spatio-Temporal GNN")

for epoch in range(40):
    total_loss = 0
    for i in range(len(X)):
        optimizer.zero_grad()
        pred = model(X[i:i+1], edge_index).squeeze()
        loss = loss_fn(pred, Y[i])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch:02d} | Loss: {total_loss/len(X):.4f}")

torch.save(model.state_dict(), os.path.join(base_path, 'data', 'processed', 'model_weights.pth'))
print("✅ Model saved")
