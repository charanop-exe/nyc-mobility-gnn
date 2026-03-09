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

BATCH_SIZE = 16

print("🚀 Training Spatio-Temporal GNN")

for epoch in range(40):
    total_loss = 0
    num_batches = 0

    for start in range(0, len(X), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(X))
        x_batch = X[start:end]
        y_batch = Y[start:end]

        optimizer.zero_grad()
        pred = model(x_batch, edge_index)   # [B, N]
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    print(f"Epoch {epoch:02d} | Loss: {total_loss/num_batches:.4f}")

torch.save(model.state_dict(), os.path.join(base_path, 'data', 'processed', 'model_weights.pth'))
print("✅ Model saved")
