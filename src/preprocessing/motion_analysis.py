import os
import csv
from pathlib import Path

import cv2

VIDEO_DIR = "data/raw_videos"
OUTPUT_DIR = "data/processed/motion_raw"


def _new_motion_processor():
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=50,
        detectShadows=False,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return fgbg, kernel


def analyze_motion_in_video(video_path, output_csv_path, resize=(640, 360)):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 25
    fgbg, kernel = _new_motion_processor()
    frame_id = 0

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "second", "motion_pixels", "motion_ratio"])

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            second = int(frame_id // fps)
            frame = cv2.resize(frame, resize)

            fgmask = fgbg.apply(frame)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            motion_pixels = cv2.countNonZero(fgmask)
            total_pixels = frame.shape[0] * frame.shape[1]
            motion_ratio = motion_pixels / total_pixels

            writer.writerow([frame_id, second, motion_pixels, round(motion_ratio, 6)])

    cap.release()
    return output_csv_path


def process_all_videos(video_dir=VIDEO_DIR, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    outputs = []

    for video_name in os.listdir(video_dir):
        if not video_name.lower().endswith(".mp4"):
            continue

        video_path = os.path.join(video_dir, video_name)
        output_csv = os.path.join(output_dir, f"{Path(video_name).stem}_motion.csv")
        print(f"Processing: {video_name}")
        analyze_motion_in_video(video_path, output_csv)
        outputs.append(output_csv)

    return outputs


if __name__ == "__main__":
    process_all_videos()
    print("All videos processed successfully.")
