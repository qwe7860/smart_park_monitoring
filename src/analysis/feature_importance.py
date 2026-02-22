import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# CONFIG
# =========================
MODEL_PATH = r"D:\smart_park_monitoring\models\activity_rf_model.pkl"
DATA_PATH = "data/processed/master_labeled.csv"
OUTPUT_IMAGE = "data/processed/feature_importance.png"

os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)

# =========================
# LOAD MODEL
# =========================
model = joblib.load(MODEL_PATH)

# =========================
# LOAD TRAINING DATA
# =========================
df = pd.read_csv(DATA_PATH)

# EXACT SAME FEATURES USED IN TRAINING
feature_columns = [
    "avg_motion_ratio",
    "motion_std",
    "people_count"
]

X = df[feature_columns]

# =========================
# GET IMPORTANCES
# =========================
importances = model.feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nFeature Importance Ranking:\n")
print(importance_df)

# =========================
# PLOT
# =========================
plt.figure()
plt.bar(importance_df["feature"], importance_df["importance"])
plt.xticks(rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()

plt.savefig(OUTPUT_IMAGE)
plt.close()

print(f"\nFeature importance plot saved to {OUTPUT_IMAGE}")