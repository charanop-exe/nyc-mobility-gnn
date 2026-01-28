import os
import duckdb

# 1. Dynamically find the project root (D:\spatial_temporal_mobility)
# __file__ is the path to this script. We go up two levels to get the root.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# 2. Define absolute paths to your data
input_path = os.path.join(project_root, 'data', 'raw', 'yellow_tripdata_2025-01.parquet')
output_path = os.path.join(project_root, 'data', 'processed', 'hourly_demand.csv')

# 3. Security Check: Print paths so you can see what Python sees
print(f"🔍 Checking input file: {input_path}")
if not os.path.exists(input_path):
    print("❌ Error: Still can't find the file! Double-check your folder names.")
    exit()

# 4. Use Forward Slashes for the SQL Query (DuckDB requirement)
input_sql = input_path.replace("\\", "/")
output_sql = output_path.replace("\\", "/")

# 5. Run the query using the Absolute Path
query = f"""
    COPY (
        SELECT 
            date_trunc('hour', tpep_pickup_datetime) AS hour,
            PULocationID AS zone_id,
            count(*) AS demand
        FROM read_parquet('{input_sql}')
        GROUP BY 1, 2
    ) TO '{output_sql}' (FORMAT CSV, HEADER);
"""

print("🏗️ Processing data with DuckDB...")
duckdb.execute(query)
print(f"✅ SUCCESS! Created: {output_path}")