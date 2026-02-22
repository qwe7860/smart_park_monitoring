import os
import csv
from collections import defaultdict
import statistics

# CONFIG
INPUT_DIR = "data/processed/motion_raw"
OUTPUT_FILE = "data/processed/motion_aggregated.csv"

aggregated_rows = []

# PROCESS EACH VIDEO CSV
for file_name in os.listdir(INPUT_DIR):

    if not file_name.endswith(".csv"):
        continue

    video_name = file_name.replace("_motion.csv", "")
    file_path = os.path.join(INPUT_DIR, file_name)

    # Dictionary:
    # second -> list of motion_ratio values
    second_data = defaultdict(list)

    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Ensure second is integer
            sec = int(float(row["second"]))

            motion_ratio = float(row["motion_ratio"])
            second_data[sec].append(motion_ratio)


    # AGGREGATE PER SECOND

    for sec in sorted(second_data.keys()):

        ratio_vals = second_data[sec]

        if len(ratio_vals) == 0:
            continue

        avg_ratio = round(sum(ratio_vals) / len(ratio_vals), 6)

        motion_std = (
            round(statistics.pstdev(ratio_vals), 6)
            if len(ratio_vals) > 1 else 0
        )

        max_ratio = round(max(ratio_vals), 6)
        min_ratio = round(min(ratio_vals), 6)
        motion_range = round(max_ratio - min_ratio, 6)

        aggregated_rows.append([
            video_name,
            sec,
            avg_ratio,
            motion_std,
            max_ratio,
            min_ratio,
            motion_range
        ])

# WRITE FINAL CSV
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow([
        "video",
        "second",
        "avg_motion_ratio",
        "motion_std",
        "max_motion_ratio",
        "min_motion_ratio",
        "motion_range"
    ])

    writer.writerows(aggregated_rows)

print("Motion aggregation complete.")
print(f"Saved to {OUTPUT_FILE}")