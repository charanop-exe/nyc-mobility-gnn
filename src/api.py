# src/api.py
import os
import numpy as np
import torch
from fastapi import FastAPI
from src.model import TrafficGNN

# --------------------------------------------------
# 1. APP INITIALIZATION
# --------------------------------------------------
app = FastAPI(
    title="NYC Taxi Demand Predictor",
    description="Spatio-Temporal GNN for Zone-Level Demand Forecasting",
    version="1.0"
)

# --------------------------------------------------
# 2. LOAD DATA & MODEL (ONCE)
# --------------------------------------------------
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data = np.load(os.path.join(base_path, "data", "processed", "final_dataset.npz"))
edge_index = torch.tensor(data["adjacency"], dtype=torch.long).t().contiguous()
max_val = data["max_val"]

model = TrafficGNN(input_dim=3, hidden_dim=32)
model.load_state_dict(
    torch.load(os.path.join(base_path, "data", "processed", "model_weights.pth"),
               map_location="cpu")
)
model.eval()

print("✅ Model and graph loaded successfully")

# --------------------------------------------------
# 3. PREDICT ENDPOINT
# --------------------------------------------------
@app.get("/predict")
def predict():
    """
    Predict next-hour demand for all taxi zones
    using the most recent temporal window
    """

    # Use last available temporal window
    X = torch.tensor(data["X"][-1:], dtype=torch.float32)

    with torch.no_grad():
        prediction = model(X, edge_index).squeeze(0).numpy()

    # Rescale to real demand
    prediction = prediction * max_val

    # Convert to JSON-friendly format
    response = {
        "zones": [
            {"zone_index": int(i), "predicted_demand": float(prediction[i])}
            for i in range(len(prediction))
        ]
    }

    return response
