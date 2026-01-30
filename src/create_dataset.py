import pandas as pd
import numpy as np
import os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
demand_path = os.path.join(base_path, 'data', 'processed', 'hourly_demand.csv')
adj_path = os.path.join(base_path, 'data', 'processed', 'adjacency_matrix.csv')
output_path = os.path.join(base_path, 'data', 'processed', 'final_dataset.npz')

df_demand = pd.read_csv(demand_path)
df_adj = pd.read_csv(adj_path)

# Pivot data and fill missing zones 1-263
pivot_demand = df_demand.pivot(index='pickup_hour', columns='location_id', values='demand').fillna(0)
for i in range(1, 264):
    if i not in pivot_demand.columns: pivot_demand[i] = 0.0
pivot_demand = pivot_demand.reindex(columns=sorted(pivot_demand.columns))
pivot_demand.index = pd.to_datetime(pivot_demand.index)

# FEATURE ENGINEERING
demand_raw = pivot_demand.values.astype('float32')
max_val = demand_raw.max()
demand_layer = demand_raw / max_val # Normalization

# Create temporal layers (Hour and Day)
hour_layer = np.tile(pivot_demand.index.hour.values, (263, 1)).T / 23.0
day_layer = np.tile(pivot_demand.index.dayofweek.values, (263, 1)).T / 6.0

# Stack into 3D: (Time, Zones, 3 Features)
final_matrix = np.stack([demand_layer, hour_layer, day_layer], axis=-1)

np.savez(output_path, data=final_matrix, adjacency=df_adj.values, max_val=max_val)
print(f"✅ 3D Dataset Created: {final_matrix.shape}")