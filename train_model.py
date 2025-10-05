import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import joblib

# Load data
df = pd.read_csv("nasa_exoplanet_data_clean.csv")

# Features (exclude 'mission' for now)
feature_cols = [
    'koi_period', 'koi_prad', 'koi_srad', 'koi_steff', 'koi_slogg'
]
X = df[feature_cols]
y = df['label']

# Train/test split (stratify to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['False Positive', 'Planet']))

# Save model and scaler
joblib.dump(model, "exoplanet_model.pkl")
joblib.dump(scaler, "feature_scaler.pkl")
print("✅ Model and scaler saved.")

# Save feature statistics for uncertainty estimation
feature_stats = {
    'mean': X.mean().to_dict(),
    'std': X.std().to_dict()
}
joblib.dump(feature_stats, "feature_stats.pkl")

# train_model.py (add at the end)

# --- SHAP Explainer ---
import shap

# Use a sample of the training data for SHAP background
X_sample = pd.DataFrame(X_train_scaled, columns=feature_cols).sample(n=min(100, len(X_train_scaled)), random_state=42)
explainer = shap.Explainer(model, X_sample)
shap_values = explainer(X_sample)  # Precompute for faster UI

# Save SHAP explainer
joblib.dump(explainer, "shap_explainer.pkl")
print("✅ SHAP explainer saved!")