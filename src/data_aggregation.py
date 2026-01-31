# src/data_aggregation.py
import os
import duckdb

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_path = os.path.join(base_path, 'data', 'raw', 'yellow_tripdata_2025-01.parquet')
output_path = os.path.join(base_path, 'data', 'processed', 'hourly_demand.csv')

print("🚕 Aggregating hourly demand...")

query = f"""
COPY (
    SELECT 
        date_trunc('hour', tpep_pickup_datetime) AS hour,
        PULocationID AS zone_id,
        COUNT(*) AS demand
    FROM read_parquet('{input_path.replace("\\\\","/")}')
    GROUP BY 1,2
) TO '{output_path.replace("\\\\","/")}' (FORMAT CSV, HEADER);
"""

duckdb.execute(query)
print("✅ hourly_demand.csv created")
