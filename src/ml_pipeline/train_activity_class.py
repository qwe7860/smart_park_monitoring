import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

INPUT_FILE = "data/processed/master_labeled.csv"
MODEL_OUTPUT = "models/activity_rf_model.pkl"

# -------------------------
# Load Data
# -------------------------
df = pd.read_csv(INPUT_FILE)

# Keep only labeled rows
df = df.dropna(subset=["activity_label"])

# Features
X = df[[
    "avg_motion_ratio",
    "motion_std",
    "people_count"
]]

# Labels
y = df["activity_label"]

# -------------------------
# Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------
# Train Model (handle imbalance)
# -------------------------
model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# -------------------------
# Evaluate
# -------------------------
y_pred = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# -------------------------
# Save Model
# -------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_OUTPUT)

print("\nModel saved successfully.")

# import matplotlib.pyplot as plt

# feature_importance = model.feature_importances_
# features = ["avg_motion_ratio", "motion_std", "people_count"]

# plt.bar(features, feature_importance)
# plt.title("Feature Importance")
# plt.show()