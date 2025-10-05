# train_anomaly_detector.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import joblib

# Load cleaned dataset
df = pd.read_csv("nasa_exoplanet_data_clean.csv")

# Use ONLY confirmed planets (label == 1)
planets = df[df['label'] == 1].copy()

# Use the same 5 features as your main model
feature_cols = ['koi_period', 'koi_prad', 'koi_srad', 'koi_steff', 'koi_slogg']
X_planets = planets[feature_cols].dropna()

print(f"Training anomaly detector on {len(X_planets)} confirmed planets...")

# Scale features
scaler_anomaly = RobustScaler()
X_scaled = scaler_anomaly.fit_transform(X_planets)

# Train Isolation Forest
anomaly_detector = IsolationForest(
    contamination=0.05,  # Expect ~5% anomalies
    random_state=42,
    n_estimators=100
)
anomaly_detector.fit(X_scaled)

# Save models
joblib.dump(anomaly_detector, "anomaly_detector.pkl")
joblib.dump(scaler_anomaly, "anomaly_scaler.pkl")

print("✅ Anomaly detector saved!")