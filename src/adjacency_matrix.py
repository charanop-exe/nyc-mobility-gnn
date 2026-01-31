# src/adjacency_matrix.py
import geopandas as gpd
import pandas as pd
import os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
shp_path = os.path.join(base_path, 'data', 'raw', 'taxi_zones.shp')
output_path = os.path.join(base_path, 'data', 'processed', 'adjacency_matrix.csv')

zones = gpd.read_file(shp_path).sort_values('LocationID')

edges = []
for _, z1 in zones.iterrows():
    neighbors = zones[zones.geometry.touches(z1.geometry)]
    for _, z2 in neighbors.iterrows():
        edges.append({
            "from_zone": z1["LocationID"],
            "to_zone": z2["LocationID"]
        })

pd.DataFrame(edges).to_csv(output_path, index=False)
print("✅ adjacency_matrix.csv created")
