import os
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from src.analysis.activity_distribution import compute_activity_distribution
from src.analysis.congestion_detection import upsert_congestion_for_video
from src.analysis.crowd_statistics import upsert_crowd_statistics_for_video
from src.analysis.feature_importance import generate_feature_importance_plot
from src.detection.yolo_people_detection import detect_people_in_video
from src.ml_pipeline.predict_activity import upsert_predictions_for_video
from src.ml_pipeline.train_activity_class import (
    self_train_from_predictions,
    train_activity_model,
)
from src.preprocessing.aggregate_motion import aggregate_motion_csv, upsert_motion_aggregated
from src.preprocessing.merge_motion_people import upsert_master_dataset_for_video
from src.preprocessing.motion_analysis import analyze_motion_in_video

RAW_VIDEO_DIR = "data/raw_videos"
PEOPLE_DIR = "data/processed/people_per_second"
MOTION_RAW_DIR = "data/processed/motion_raw"
MOTION_AGG_FILE = "data/processed/motion_aggregated.csv"
MASTER_DATASET_FILE = "data/processed/master_dataset.csv"
PREDICTIONS_FILE = "data/processed/activity_ml_predictions.csv"
MODEL_FILE = "models/activity_rf_model.pkl"
MASTER_LABELED_FILE = "data/processed/master_labeled.csv"


def ensure_pipeline_dirs():
    os.makedirs(RAW_VIDEO_DIR, exist_ok=True)
    os.makedirs(PEOPLE_DIR, exist_ok=True)
    os.makedirs(MOTION_RAW_DIR, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)


def save_uploaded_video(uploaded_file, video_dir=RAW_VIDEO_DIR):
    ensure_pipeline_dirs()
    base_name = Path(uploaded_file.name).stem
    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in base_name).strip("_")
    if not safe_name:
        safe_name = "uploaded_video"

    target = Path(video_dir) / f"{safe_name}.mp4"
    suffix = 1
    while target.exists():
        target = Path(video_dir) / f"{safe_name}_{suffix}.mp4"
        suffix += 1

    target.write_bytes(uploaded_file.getbuffer())
    return str(target), target.stem


def run_analysis_for_video(video_path, model_path=MODEL_FILE):
    ensure_pipeline_dirs()
    video_name = Path(video_path).stem
    people_csv = os.path.join(PEOPLE_DIR, f"{video_name}_people.csv")
    motion_csv = os.path.join(MOTION_RAW_DIR, f"{video_name}_motion.csv")

    yolo_model = YOLO("yolov8n.pt")
    detect_people_in_video(video_path=video_path, output_csv_path=people_csv, model=yolo_model)
    analyze_motion_in_video(video_path=video_path, output_csv_path=motion_csv)

    motion_rows = aggregate_motion_csv(motion_csv_path=motion_csv, video_name=video_name)
    upsert_motion_aggregated(motion_rows, output_file=MOTION_AGG_FILE)
    upsert_master_dataset_for_video(video_name=video_name, output_file=MASTER_DATASET_FILE)
    upsert_predictions_for_video(video_name=video_name, model_file=model_path, output_file=PREDICTIONS_FILE)
    compute_activity_distribution(input_file=PREDICTIONS_FILE)
    upsert_crowd_statistics_for_video(video_name=video_name)
    upsert_congestion_for_video(video_name=video_name)

    dataset_rows = 0
    if os.path.exists(MASTER_DATASET_FILE):
        df = pd.read_csv(MASTER_DATASET_FILE)
        dataset_rows = int((df["video"] == video_name).sum())

    return {
        "video_name": video_name,
        "people_csv": people_csv,
        "motion_csv": motion_csv,
        "dataset_rows": dataset_rows,
    }


def self_train_after_upload(video_name):
    result = self_train_from_predictions(
        video_name=video_name,
        master_labeled_file=MASTER_LABELED_FILE,
        predictions_file=PREDICTIONS_FILE,
    )
    rows_added = result.get("rows_added", 0)
    if rows_added == 0:
        return {
            "rows_added": 0,
            "trained": False,
            "reason": "No new pseudo-labeled rows were available.",
        }

    try:
        train_result = train_activity_model(
            input_file=MASTER_LABELED_FILE,
            model_output=MODEL_FILE,
        )
        generate_feature_importance_plot(model_path=MODEL_FILE, data_path=MASTER_LABELED_FILE)
    except Exception as exc:
        return {
            "rows_added": rows_added,
            "trained": False,
            "reason": f"Training failed: {exc}",
        }
    return {
        "rows_added": rows_added,
        "trained": True,
        "classification_report": train_result["classification_report"],
    }
