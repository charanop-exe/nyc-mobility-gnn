import pandas as pd
import numpy as np
import os
import sys

# 1. SETUP PATHS
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
demand_path = os.path.join(base_path, 'data', 'processed', 'hourly_demand.csv')
adj_path = os.path.join(base_path, 'data', 'processed', 'adjacency_matrix.csv')
output_path = os.path.join(base_path, 'data', 'processed', 'final_dataset.npz')

print("🧪 Merging Spatial and Temporal data...")

# 2. LOAD DATA
df_demand = pd.read_csv(demand_path)
df_adj = pd.read_csv(adj_path)

# 3. AUTO-DETECT COLUMN NAMES (The fix for KeyError)
# We look for whatever names exist in your CSV
cols = df_demand.columns.tolist()
print(f"📊 Found columns in CSV: {cols}")

# Find the time column (might be 'hour', 'pickup_hour', etc.)
time_col = next((c for c in cols if 'hour' in c.lower()), None)
# Find the zone column (might be 'location_id', 'zone_id', etc.)
zone_col = next((c for c in cols if 'id' in c.lower() or 'zone' in c.lower()), None)
# Find the demand column
val_col = next((c for c in cols if 'demand' in c.lower() or 'count' in c.lower()), None)

if not all([time_col, zone_col, val_col]):
    print(f"❌ Error: Could not identify columns automatically. Please rename them in your CSV.")
    sys.exit()

print(f"🔗 Mapping: Time->'{time_col}', Zone->'{zone_col}', Value->'{val_col}'")

# 4. PIVOT THE DATA
# This turns the 'Long' list into a 'Wide' matrix (Rows=Time, Cols=Zones)
pivot_demand = df_demand.pivot(index=time_col, columns=zone_col, values=val_col).fillna(0)

# 5. ENSURE ALL 263 ZONES ARE PRESENT
for i in range(1, 264):
    if i not in pivot_demand.columns:
        pivot_demand[i] = 0.0

# Sort zones (1 to 263) and convert to float32 for the AI
pivot_demand = pivot_demand.reindex(columns=sorted(pivot_demand.columns))
demand_matrix = pivot_demand.values.astype('float32')

# 6. SAVE AS COMPRESSED NUMPY
np.savez(output_path, demand=demand_matrix, adjacency=df_adj.values)

print(f"✅ SUCCESS! Created dataset with shape: {demand_matrix.shape}")
print(f"💾 Saved to: {output_path}")