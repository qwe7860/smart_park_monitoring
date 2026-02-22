import os
import cv2
import csv

# =========================
# CONFIG
# =========================
VIDEO_DIR = "data/raw_videos"
OUTPUT_DIR = "data/processed/motion_raw"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Background subtractor (stable configuration)
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=False
)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


# =========================
# PROCESS VIDEOS
# =========================
for video_name in os.listdir(VIDEO_DIR):

    if not video_name.lower().endswith(".mp4"):
        continue

    video_path = os.path.join(VIDEO_DIR, video_name)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open {video_name}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 25

    output_csv = os.path.join(
        OUTPUT_DIR,
        video_name.replace(".mp4", "_motion.csv")
    )

    print(f"Processing: {video_name}")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # No activity column anymore
        writer.writerow([
            "frame",
            "second",
            "motion_pixels",
            "motion_ratio"
        ])

        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            second = int(frame_id // fps)

            # Resize for stability
            frame = cv2.resize(frame, (640, 360))

            fgmask = fgbg.apply(frame)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

            motion_pixels = cv2.countNonZero(fgmask)

            # Normalize motion (important for scale invariance)
            total_pixels = frame.shape[0] * frame.shape[1]
            motion_ratio = motion_pixels / total_pixels

            writer.writerow([
                frame_id,
                second,
                motion_pixels,
                round(motion_ratio, 6)
            ])

    cap.release()

print("All videos processed successfully.")