import geopandas as gpd
import pandas as pd
import os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
shp_path = os.path.join(base_path, 'data', 'raw', 'taxi_zones.shp')
output_path = os.path.join(base_path, 'data', 'processed', 'adjacency_matrix.csv')

print("🗺️ Loading Shapefile and Computing Adjacency...")
zones = gpd.read_file(shp_path).sort_values('LocationID')

adj_list = []
for i, zone_a in zones.iterrows():
    neighbors = zones[zones.geometry.touches(zone_a.geometry)]
    for j, zone_b in neighbors.iterrows():
        adj_list.append({'from_zone': zone_a['LocationID'], 'to_zone': zone_b['LocationID']})

adj_df = pd.DataFrame(adj_list)
adj_df.to_csv(output_path, index=False)
print(f"✅ Success! Saved {len(adj_df)} connections to: {output_path}")