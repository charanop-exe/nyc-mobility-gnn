# src/calculate_metrics.py
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_path, 'data', 'processed')

y_true = np.load(os.path.join(data_path, 'y_true.npy'))
y_pred = np.load(os.path.join(data_path, 'y_pred.npy'))

print("📦 Loaded prediction results")
print(f"   y_true shape: {y_true.shape}")
print(f"   y_pred shape: {y_pred.shape}")

# --------------------------------------------------
# 2. METRICS (VERSION-SAFE)
# --------------------------------------------------
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

# --------------------------------------------------
# 3. PRINT RESULTS
# --------------------------------------------------
print("\n📊 Model Evaluation Metrics")
print("--------------------------------")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAE  : {mae:.2f}")
print(f"R²   : {r2:.3f}")
print("--------------------------------")
print("✅ Metrics calculation complete")
