import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

INPUT_FILE = "data/processed/master_labeled.csv"
MODEL_OUTPUT = "models/activity_rf_model.pkl"
FEATURE_COLUMNS = ["avg_motion_ratio", "motion_std", "people_count"]

def train_activity_model(input_file=INPUT_FILE, model_output=MODEL_OUTPUT):
    df = pd.read_csv(input_file)
    df = df.dropna(subset=["activity_label"])
    X = df[FEATURE_COLUMNS]
    y = df["activity_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, model_output)

    return {
        "model_path": model_output,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }


def self_train_from_predictions(
    video_name,
    master_labeled_file=INPUT_FILE,
    predictions_file="data/processed/activity_ml_predictions.csv",
):
    labeled_df = pd.read_csv(master_labeled_file)
    predictions_df = pd.read_csv(predictions_file)

    video_preds = predictions_df[predictions_df["video"] == video_name].copy()
    if video_preds.empty:
        return {"rows_added": 0}

    keep_cols = ["video", "second", "avg_motion_ratio", "motion_std", "people_count"]
    pseudo_df = video_preds[keep_cols + ["predicted_activity"]].rename(
        columns={"predicted_activity": "activity_label"}
    )
    pseudo_df["is_pseudo_label"] = 1

    if "is_pseudo_label" not in labeled_df.columns:
        labeled_df["is_pseudo_label"] = 0

    existing_keys = set(zip(labeled_df["video"], labeled_df["second"]))
    pseudo_df = pseudo_df[
        ~pseudo_df.apply(lambda r: (r["video"], r["second"]) in existing_keys, axis=1)
    ]
    if pseudo_df.empty:
        return {"rows_added": 0}

    merged = pd.concat([labeled_df, pseudo_df], ignore_index=True)
    merged.to_csv(master_labeled_file, index=False)
    return {"rows_added": len(pseudo_df)}


if __name__ == "__main__":
    result = train_activity_model()
    print("\nClassification Report:\n")
    print(result["classification_report"])
    print("\nConfusion Matrix:\n")
    print(result["confusion_matrix"])
    print("\nModel saved successfully.")
