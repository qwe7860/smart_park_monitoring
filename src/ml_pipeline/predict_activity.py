import pandas as pd
import joblib
import os

INPUT_FILE = "data/processed/master_dataset.csv"
MODEL_FILE = "models/activity_rf_model.pkl"
OUTPUT_FILE = "data/processed/activity_ml_predictions.csv"

# Load model
model = joblib.load(MODEL_FILE)

# Load dataset
df = pd.read_csv(INPUT_FILE)

# Features
X = df[[
    "avg_motion_ratio",
    "motion_std",
    "people_count"
]]

# Predict
df["predicted_activity"] = model.predict(X)

# Save
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("ML predictions saved successfully.")