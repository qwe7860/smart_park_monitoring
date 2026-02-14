import csv
import os
import matplotlib.pyplot as plt
from collections import defaultdict

INPUT_FILE = "data/processed/motion_aggregated.csv"
OUTPUT_DIR = "data/processed/plots/motion_timelines"

os.makedirs(OUTPUT_DIR, exist_ok=True)

video_seconds = defaultdict(list)
video_motion = defaultdict(list)

with open(INPUT_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        video = row["video"]
        video_seconds[video].append(int(row["second"]))
        video_motion[video].append(float(row["avg_motion"]))

for video in video_seconds:
    plt.figure(figsize=(10, 4))
    plt.plot(video_seconds[video], video_motion[video])
    plt.xlabel("Time (seconds)")
    plt.ylabel("Average Motion")
    plt.title(f"Motion Timeline: {video}")
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f"{video}_motion.png")
    plt.savefig(out_path)
    plt.close()

print("Motion timeline plots saved.")