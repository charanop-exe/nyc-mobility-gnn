# src/streamlit_app.py
import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from model import TrafficGNN


# --------------------------------------------------
# 1. PAGE SETUP
# --------------------------------------------------
st.set_page_config(page_title="NYC Taxi Demand Predictor", layout="wide")

st.title("🚕 NYC Taxi Demand Prediction")
st.write("Spatio-Temporal GNN | Hour-based Forecasting")

# --------------------------------------------------
# 2. LOAD DATA & MODEL (CACHE)
# --------------------------------------------------
@st.cache_resource
def load_model_and_data():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    data = np.load(os.path.join(base_path, "data", "processed", "final_dataset.npz"))

    X = torch.tensor(data["X"], dtype=torch.float32)
    edge_index = torch.tensor(data["adjacency"], dtype=torch.long).t().contiguous()
    max_val = data["max_val"]

    model = TrafficGNN(input_dim=3, hidden_dim=32)
    model.load_state_dict(
        torch.load(
            os.path.join(base_path, "data", "processed", "model_weights.pth"),
            map_location="cpu"
        )
    )
    model.eval()

    return X, edge_index, max_val, model

X, edge_index, max_val, model = load_model_and_data()

num_hours = X.shape[0]
num_zones = X.shape[2]

st.success(f"Loaded {num_hours} hours and {num_zones} zones")

# --------------------------------------------------
# 3. HOUR SELECTION
# --------------------------------------------------
hour_index = st.slider(
    "Select Hour Index",
    min_value=0,
    max_value=num_hours - 1,
    value=num_hours - 1
)

st.caption("Each index represents a specific hour in the dataset timeline")

# --------------------------------------------------
# 4. PREDICTION
# --------------------------------------------------
if st.button("🔮 Predict Demand"):
    with torch.no_grad():
        pred = model(X[hour_index:hour_index+1], edge_index)
        pred = pred.squeeze(0).numpy() * max_val

    st.subheader(f"Predicted Demand (Hour Index: {hour_index})")

    # --------------------------------------------------
    # 5. SHOW TABLE
    # --------------------------------------------------
    st.dataframe({
        "Zone Index": list(range(len(pred))),
        "Predicted Demand": pred.round(2)
    })

    # --------------------------------------------------
    # 6. PLOT TOP ZONES
    # --------------------------------------------------
    top_k = 20
    top_indices = np.argsort(pred)[-top_k:][::-1]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(
        [f"Z{int(i)}" for i in top_indices],
        pred[top_indices]
    )
    ax.set_title("Top 20 Zones by Predicted Demand")
    ax.set_ylabel("Pickups")

    st.pyplot(fig)
