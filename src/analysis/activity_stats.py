import csv
from collections import defaultdict, Counter
import os

INPUT_FILE = "data/processed/activity_baseline.csv"
OUTPUT_FILE = "data/processed/activity_summary.csv"

video_activity = defaultdict(list)

with open(INPUT_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        video = row["video"]
        activity = row["activity"]
        video_activity[video].append(activity)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "video",
        "sitting_pct",
        "walking_pct",
        "high_activity_pct"
    ])

    for video, activities in video_activity.items():
        total = len(activities)
        counts = Counter(activities)

        writer.writerow([
            video,
            round(counts["sitting"] / total * 100, 2),
            round(counts["walking"] / total * 100, 2),
            round(counts["high_activity"] / total * 100, 2),
        ])

print("Activity summary saved.")