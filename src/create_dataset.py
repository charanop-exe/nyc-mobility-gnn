import pandas as pd
import numpy as np
import os
import sys

# 1. SETUP PATHS
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
demand_path = os.path.join(base_path, 'data', 'processed', 'hourly_demand.csv')
adj_path = os.path.join(base_path, 'data', 'processed', 'adjacency_matrix.csv')
output_path = os.path.join(base_path, 'data', 'processed', 'final_dataset.npz')

# 2. LOAD DATA
df_demand = pd.read_csv(demand_path)
df_adj = pd.read_csv(adj_path)

# Clean column names
df_demand.columns = df_demand.columns.str.strip()
df_adj.columns = df_adj.columns.str.strip()

# Detect columns
cols = df_demand.columns.tolist()
time_col = next((c for c in cols if 'hour' in c.lower()), 'hour')
zone_col = next((c for c in cols if 'id' in c.lower() or 'zone' in c.lower()), 'zone_id')
val_col = next((c for c in cols if 'demand' in c.lower()), 'demand')

# 3. PIVOT DATA
pivot_demand = df_demand.pivot(index=time_col, columns=zone_col, values=val_col).fillna(0)
pivot_demand.index = pd.to_datetime(pivot_demand.index)

# 4. MAPPING RAW IDS TO INDICES (The "Out of Bounds" Fix)
active_zones = list(pivot_demand.columns)
mapping = {id: i for i, id in enumerate(active_zones)}

# Filter and Map Adjacency
df_adj_filtered = df_adj[df_adj['from_zone'].isin(active_zones) & df_adj['to_zone'].isin(active_zones)].copy()
df_adj_filtered['from_idx'] = df_adj_filtered['from_zone'].map(mapping)
df_adj_filtered['to_idx'] = df_adj_filtered['to_zone'].map(mapping)

# 5. FEATURE ENGINEERING (3D TENSOR)
num_hours, num_zones = pivot_demand.shape
demand_raw = pivot_demand.values.astype('float32')
max_val = demand_raw.max()
demand_layer = demand_raw / max_val

hour_values = (pivot_demand.index.hour.values / 23.0).astype('float32')
hour_layer = np.tile(hour_values, (num_zones, 1)).T

day_values = (pivot_demand.index.dayofweek.values / 6.0).astype('float32')
day_layer = np.tile(day_values, (num_zones, 1)).T

final_matrix = np.stack([demand_layer, hour_layer, day_layer], axis=-1)

# 6. SAVE
np.savez(output_path, 
         data=final_matrix, 
         adjacency=df_adj_filtered[['from_idx', 'to_idx']].values, 
         max_val=max_val)

print(f"✅ SUCCESS: Dataset {final_matrix.shape} saved with {len(df_adj_filtered)} edges.")