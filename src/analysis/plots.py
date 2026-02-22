import csv
import os
import matplotlib.pyplot as plt
from collections import defaultdict

INPUT_FILE = "data/processed/motion_aggregated.csv"
OUTPUT_DIR = "data/processed/plots_motion_timeline"

os.makedirs(OUTPUT_DIR, exist_ok=True)

video_seconds = defaultdict(list)
video_motion_ratio = defaultdict(list)

# LOAD DATA
with open(INPUT_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        video = row["video"]
        video_seconds[video].append(int(row["second"]))
        video_motion_ratio[video].append(float(row["avg_motion_ratio"]))

#PLOT
for video in video_seconds:

    seconds = video_seconds[video]
    motion = video_motion_ratio[video]

    plt.figure(figsize=(12, 5))

    plt.plot(
        seconds,
        motion,
        linewidth=1.5
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Average Motion Ratio")
    plt.title(f"Motion Intensity Timeline — {video}")
    plt.grid(alpha=0.3)

    plt.tight_layout()

    out_path = os.path.join(
        OUTPUT_DIR,
        f"{video}_motion_ratio.png"
    )

    plt.savefig(out_path)
    plt.close()

print("Motion ratio timeline plots saved.")