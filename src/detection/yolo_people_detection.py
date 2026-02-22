import os
import cv2
import csv
from ultralytics import YOLO

# CONFIG
VIDEO_DIR = "data/raw_videos"
OUTPUT_DIR = "data/processed/people_per_second"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load lightweight YOLO model
model = YOLO("yolov8n.pt")

# PROCESS VIDEOS
for video_name in os.listdir(VIDEO_DIR):

    if not video_name.lower().endswith(".mp4"):
        continue

    video_path = os.path.join(VIDEO_DIR, video_name)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open {video_name}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(fps) if fps > 0 else 25

    output_csv = os.path.join(
        OUTPUT_DIR,
        video_name.replace(".mp4", "_people.csv")
    )

    print(f"Processing: {video_name}")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "second", "people_count"])

        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1

            #Process only 1 frame per second
            if frame_id % fps != 0:
                continue

            second = frame_id // fps

            # Resize for faster inference
            frame = cv2.resize(frame, (640, 360))

            results = model(frame, verbose=False)

            people_count = 0

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls == 0:  # class 0 = person
                        people_count += 1

            writer.writerow([
                video_name.replace(".mp4", ""),
                second,
                people_count
            ])

    cap.release()

print("YOLO 1-FPS detection complete.")