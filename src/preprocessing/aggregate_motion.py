import os
import csv
from collections import Counter, defaultdict

INPUT_DIR = "data/processed/motion_raw"
OUTPUT_FILE = "data/processed/motion_aggregated.csv"

aggregated_rows = []

for file_name in os.listdir(INPUT_DIR):
    if not file_name.endswith(".csv"):
        continue

    video_name = file_name.replace(".csv", "")
    file_path = os.path.join(INPUT_DIR, file_name)

    # second -> list of activities & motion
    second_data = defaultdict(lambda: {"activities": [], "motion": []})

    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            sec = int(float(row["second"]))  # floor to 1-second window
            second_data[sec]["activities"].append(row["activity"])
            second_data[sec]["motion"].append(float(row["motion_pixels"]))

    for sec in sorted(second_data.keys()):
        activities = second_data[sec]["activities"]
        motion_vals = second_data[sec]["motion"]

        dominant_activity = Counter(activities).most_common(1)[0][0]
        avg_motion = round(sum(motion_vals) / len(motion_vals), 2)

        aggregated_rows.append([
            video_name,
            sec,
            dominant_activity,
            avg_motion
        ])

# Write final aggregated CSV
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["video", "second", "dominant_activity", "avg_motion"])
    writer.writerows(aggregated_rows)

print("Aggregation complete.")
print(f"Saved to {OUTPUT_FILE}")