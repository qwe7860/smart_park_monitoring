import csv
from collections import defaultdict, Counter

INPUT_FILE = "data/processed/motion_aggregated.csv"
OUTPUT_FILE = "data/processed/activity_summary.csv"

video_activity = defaultdict(list)

with open(INPUT_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        video = row["video"]
        activity = row["dominant_activity"]
        video_activity[video].append(activity)

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["video", "stationary_pct", "walking_pct", "high_activity_pct"])

    for video, activities in video_activity.items():
        total = len(activities)
        counts = Counter(activities)

        writer.writerow([
            video,
            round(counts["low_motion"] / total * 100, 2),
            round(counts["moderate_motion"] / total * 100, 2),
            round(counts["high_motion"] / total * 100, 2),
        ])

print("Activity summary saved.")