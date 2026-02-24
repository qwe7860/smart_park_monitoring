import os
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO

VIDEO_DIR = "data/raw_videos"
OUTPUT_DIR = "data/processed/people_per_second"
DEFAULT_MODEL_PATH = "yolov8n.pt"


def detect_people_in_video(video_path, output_csv_path, model, resize=(640, 360)):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(fps) if fps > 0 else 25
    frame_id = 0
    video_name = Path(video_path).stem

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "second", "people_count"])

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            if frame_id % fps != 0:
                continue

            second = frame_id // fps
            frame = cv2.resize(frame, resize)
            results = model(frame, verbose=False)

            people_count = 0
            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) == 0:
                        people_count += 1

            writer.writerow([video_name, second, people_count])

    cap.release()
    return output_csv_path


def process_all_videos(
    video_dir=VIDEO_DIR,
    output_dir=OUTPUT_DIR,
    model_path=DEFAULT_MODEL_PATH,
):
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)
    processed_files = []

    for video_name in os.listdir(video_dir):
        if not video_name.lower().endswith(".mp4"):
            continue
        video_path = os.path.join(video_dir, video_name)
        output_csv = os.path.join(output_dir, f"{Path(video_name).stem}_people.csv")
        print(f"Processing: {video_name}")
        detect_people_in_video(video_path, output_csv, model=model)
        processed_files.append(output_csv)

    return processed_files


if __name__ == "__main__":
    process_all_videos()
    print("YOLO 1-FPS detection complete.")
