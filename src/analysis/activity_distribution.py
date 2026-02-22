import os
import csv
from collections import defaultdict

# =========================
# CONFIG
# =========================
INPUT_FILE = "data/processed/activity_ml_predictions.csv"
OUTPUT_FILE = "data/processed/activity_distribution.csv"

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Structure:
# video -> activity -> count
activity_counts = defaultdict(lambda: defaultdict(int))
total_counts = defaultdict(int)

# =========================
# LOAD PREDICTIONS
# =========================
with open(INPUT_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        video = row["video"]
        activity = row["predicted_activity"]

        activity_counts[video][activity] += 1
        total_counts[video] += 1

# =========================
# COMPUTE DISTRIBUTION
# =========================
rows = []

for video in activity_counts:

    total = total_counts[video]

    sitting_pct = round((activity_counts[video].get("sitting", 0) / total) * 100, 2)
    walking_pct = round((activity_counts[video].get("walking", 0) / total) * 100, 2)
    high_pct = round((activity_counts[video].get("high_activity", 0) / total) * 100, 2)

    # Determine dominant activity
    dominant_activity = max(activity_counts[video], key=activity_counts[video].get)

    rows.append([
        video,
        sitting_pct,
        walking_pct,
        high_pct,
        dominant_activity
    ])

# =========================
# WRITE OUTPUT
# =========================
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow([
        "video",
        "sitting_percent",
        "walking_percent",
        "high_activity_percent",
        "dominant_activity"
    ])

    writer.writerows(rows)

print("Activity distribution analysis complete.")
print(f"Saved to {OUTPUT_FILE}")