import os
import csv
from collections import defaultdict

# =========================
# CONFIG
# =========================
INPUT_FILE = "data/processed/activity_ml_predictions.csv"
OUTPUT_FILE = "data/processed/activity_distribution.csv"

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
FIELDNAMES = [
    "video",
    "sitting_percent",
    "walking_percent",
    "high_activity_percent",
    "dominant_activity",
]


def compute_activity_distribution(input_file=INPUT_FILE, output_file=OUTPUT_FILE):
    activity_counts = defaultdict(lambda: defaultdict(int))
    total_counts = defaultdict(int)

    with open(input_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video = row["video"]
            activity = row["predicted_activity"]
            activity_counts[video][activity] += 1
            total_counts[video] += 1

    rows = []
    for video in activity_counts:
        total = total_counts[video]
        sitting_pct = round((activity_counts[video].get("sitting", 0) / total) * 100, 2)
        walking_pct = round((activity_counts[video].get("walking", 0) / total) * 100, 2)
        high_pct = round((activity_counts[video].get("high_activity", 0) / total) * 100, 2)
        dominant_activity = max(activity_counts[video], key=activity_counts[video].get)

        rows.append(
            {
                "video": video,
                "sitting_percent": sitting_pct,
                "walking_percent": walking_pct,
                "high_activity_percent": high_pct,
                "dominant_activity": dominant_activity,
            }
        )

    rows = sorted(rows, key=lambda x: x["video"])
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return output_file


if __name__ == "__main__":
    compute_activity_distribution()
    print("Activity distribution analysis complete.")
    print(f"Saved to {OUTPUT_FILE}")
