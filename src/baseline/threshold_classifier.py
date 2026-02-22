import csv
import os

INPUT_FILE = "data/processed/motion_aggregated.csv"
OUTPUT_FILE = "data/processed/activity_baseline.csv"

#These thresholds will need tuning based on my data distribution
LOW_THRESHOLD = 0.0015
MEDIUM_THRESHOLD = 0.006

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(INPUT_FILE, newline="", encoding="utf-8") as f_in, \
     open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f_out:

    reader = csv.DictReader(f_in)
    writer = csv.writer(f_out)

    writer.writerow([
        "video",
        "second",
        "avg_motion_ratio",
        "activity"
    ])

    for row in reader:
        ratio = float(row["avg_motion_ratio"])

        if ratio < LOW_THRESHOLD:
            activity = "sitting"
        elif ratio < MEDIUM_THRESHOLD:
            activity = "walking"
        else:
            activity = "high_activity"   # playing/exercising combined for now

        writer.writerow([
            row["video"],
            row["second"],
            ratio,
            activity
        ])

print("Baseline activity classification complete.")
print(f"Saved to {OUTPUT_FILE}")