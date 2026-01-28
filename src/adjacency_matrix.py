import geopandas as gpd
import pandas as pd
import os

# 1. SETUP PATHS
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
shp_path = os.path.join(base_path, 'data', 'raw', 'taxi_zones.shp')
output_path = os.path.join(base_path, 'data', 'processed', 'adjacency_matrix.csv')

print("🗺️ Loading NYC Taxi Zones Shapefile...")

# 2. LOAD DATA
# GeoPandas reads the .shp but needs the .dbf and .prj in the same folder
zones = gpd.read_file(shp_path)

# Ensure zones are sorted by LocationID for consistent indexing
zones = zones.sort_values('LocationID')

# 3. COMPUTE ADJACENCY (Who touches whom?)
print("🔍 Computing spatial adjacencies...")
# 'touches' checks if two polygons share a boundary
adj_list = []

for i, zone_a in zones.iterrows():
    # We find all other zones that 'touch' zone_a
    neighbors = zones[zones.geometry.touches(zone_a.geometry)]
    
    for j, zone_b in neighbors.iterrows():
        adj_list.append({
            'from_zone': zone_a['LocationID'],
            'to_zone': zone_b['LocationID'],
            'weight': 1  # 1 means they are neighbors
        })

# 4. SAVE THE GRAPH EDGES
adj_df = pd.DataFrame(adj_list)
adj_df.to_csv(output_path, index=False)

print(f"✅ SUCCESS! Created connectivity graph with {len(adj_df)} edges.")
print(f"💾 Saved to: {output_path}")