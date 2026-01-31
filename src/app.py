# src/app.py
import os
import numpy as np
import pandas as pd
import torch
import streamlit as st
import matplotlib.pyplot as plt
from model import TrafficGNN

# --------------------------------------------------
# 1. PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="NYC Taxi Demand Predictor",
    layout="wide"
)

st.title("🚕 NYC Taxi Demand Prediction Dashboard")
st.write("Spatio-Temporal Graph Neural Network | Hour-based Forecasting")

# --------------------------------------------------
# 2. LOAD MODEL, DATA, ZONE NAMES (CACHED)
# --------------------------------------------------
@st.cache_resource
def load_everything():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load dataset
    data = np.load(os.path.join(base_path, "data", "processed", "final_dataset.npz"))
    X = torch.tensor(data["X"], dtype=torch.float32)
    edge_index = torch.tensor(data["adjacency"], dtype=torch.long).t().contiguous()
    max_val = data["max_val"]

    # Load model
    model = TrafficGNN(input_dim=3, hidden_dim=32)
    model.load_state_dict(
        torch.load(
            os.path.join(base_path, "data", "processed", "model_weights.pth"),
            map_location="cpu"
        )
    )
    model.eval()

    # Load zone lookup table
    zone_lookup = pd.read_csv(
        os.path.join(base_path, "data", "raw", "taxi_zone_lookup.csv")
    )

    location_to_name = dict(
        zip(zone_lookup["LocationID"], zone_lookup["Zone"])
    )

    # IMPORTANT: zone order must match dataset column order
    zone_ids = sorted(location_to_name.keys())

    zone_index_to_name = {
        i: location_to_name[zone_ids[i]]
        for i in range(len(zone_ids))
    }

    return X, edge_index, max_val, model, zone_index_to_name


X, edge_index, max_val, model, zone_index_to_name = load_everything()

num_hours = X.shape[0]
num_zones = X.shape[2]

st.success(f"Loaded {num_hours} hours | {num_zones} NYC taxi zones")

# --------------------------------------------------
# 3. HOUR SELECTION
# --------------------------------------------------
hour_index = st.slider(
    "Select Hour Index (timeline position)",
    min_value=0,
    max_value=num_hours - 1,
    value=num_hours - 1
)

st.caption("Hour index = number of hours since the dataset start")

# --------------------------------------------------
# 4. PREDICTION
# --------------------------------------------------
if st.button("🔮 Predict Demand"):
    with torch.no_grad():
        pred = model(X[hour_index:hour_index + 1], edge_index)
        pred = pred.squeeze(0).numpy() * max_val

    # --------------------------------------------------
    # 5. TABLE VIEW (ZONE NAME + DEMAND)
    # --------------------------------------------------
    result_df = pd.DataFrame({
        "Zone": [zone_index_to_name[i] for i in range(len(pred))],
        "Predicted Pickups": pred.round(2)
    }).sort_values("Predicted Pickups", ascending=False)

    st.subheader("📋 Zone-Level Demand Prediction")
    st.dataframe(result_df, use_container_width=True)

    # --------------------------------------------------
    # 6. CLEAN & DECORATED GRAPH (TOP ZONES)
    # --------------------------------------------------
    top_k = 15
    top_df = result_df.head(top_k)

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(
        top_df["Zone"][::-1],
        top_df["Predicted Pickups"][::-1],
        color=plt.cm.Blues(np.linspace(0.4, 0.9, top_k))
    )

    # Value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + max(top_df["Predicted Pickups"]) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.0f}",
            va="center",
            fontsize=10
        )

    ax.set_title(
        f"Top {top_k} NYC Taxi Zones by Predicted Demand\n(Hour Index {hour_index})",
        fontsize=14,
        weight="bold"
    )
    ax.set_xlabel("Predicted Pickups")
    ax.set_ylabel("Taxi Zone")

    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
