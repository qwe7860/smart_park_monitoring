import pandas as pd
import joblib
import os

INPUT_FILE = "data/processed/master_dataset.csv"
MODEL_FILE = "models/activity_rf_model.pkl"
OUTPUT_FILE = "data/processed/activity_ml_predictions.csv"

FEATURE_COLUMNS = ["avg_motion_ratio", "motion_std", "people_count"]


def predict_activity(
    input_file=INPUT_FILE,
    model_file=MODEL_FILE,
    output_file=OUTPUT_FILE,
):
    model = joblib.load(model_file)
    df = pd.read_csv(input_file)
    df["predicted_activity"] = model.predict(df[FEATURE_COLUMNS])
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    return output_file


def upsert_predictions_for_video(
    video_name,
    input_file=INPUT_FILE,
    model_file=MODEL_FILE,
    output_file=OUTPUT_FILE,
):
    model = joblib.load(model_file)
    df = pd.read_csv(input_file)
    video_df = df[df["video"] == video_name].copy()
    if video_df.empty:
        return output_file

    video_df["predicted_activity"] = model.predict(video_df[FEATURE_COLUMNS])

    if os.path.exists(output_file):
        existing = pd.read_csv(output_file)
        existing = existing[existing["video"] != video_name]
        merged = pd.concat([existing, video_df], ignore_index=True)
    else:
        merged = video_df

    merged = merged.sort_values(["video", "second"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged.to_csv(output_file, index=False)
    return output_file


if __name__ == "__main__":
    predict_activity()
    print("ML predictions saved successfully.")
