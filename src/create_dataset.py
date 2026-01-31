# src/create_dataset.py
import numpy as np
import pandas as pd
import os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

demand_path = os.path.join(base_path, 'data', 'processed', 'hourly_demand.csv')
adj_path = os.path.join(base_path, 'data', 'processed', 'adjacency_matrix.csv')
output_path = os.path.join(base_path, 'data', 'processed', 'final_dataset.npz')

df = pd.read_csv(demand_path)
df['hour'] = pd.to_datetime(df['hour'])

pivot = df.pivot(index='hour', columns='zone_id', values='demand').fillna(0)
zones = pivot.columns.tolist()
mapping = {z: i for i, z in enumerate(zones)}

# Load adjacency
adj = pd.read_csv(adj_path)
adj = adj[adj['from_zone'].isin(zones) & adj['to_zone'].isin(zones)]
adj['from_idx'] = adj['from_zone'].map(mapping)
adj['to_idx'] = adj['to_zone'].map(mapping)
edge_index = adj[['from_idx', 'to_idx']].values

# Features
demand = pivot.values.astype('float32')
max_val = demand.max()
demand = demand / max_val

hour_feat = pivot.index.hour.values / 23.0
day_feat = pivot.index.dayofweek.values / 6.0

hour_feat = np.tile(hour_feat, (demand.shape[1], 1)).T
day_feat = np.tile(day_feat, (demand.shape[1], 1)).T

data = np.stack([demand, hour_feat, day_feat], axis=-1)

# 🔥 Temporal windows
def make_sequences(data, window=6):
    X, Y = [], []
    for t in range(len(data) - window - 1):
        X.append(data[t:t+window])
        Y.append(data[t+window][:, 0])
    return np.array(X), np.array(Y)

X, Y = make_sequences(data, window=6)

np.savez(output_path, X=X, Y=Y, adjacency=edge_index, max_val=max_val)
print("✅ final_dataset.npz created with temporal windows")
