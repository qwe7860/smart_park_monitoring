import os
import csv
import statistics
from collections import defaultdict

INPUT_DIR = "data/processed/motion_raw"
OUTPUT_FILE = "data/processed/motion_aggregated.csv"
FIELDNAMES = [
    "video",
    "second",
    "avg_motion_ratio",
    "motion_std",
    "max_motion_ratio",
    "min_motion_ratio",
    "motion_range",
]


def aggregate_motion_csv(motion_csv_path, video_name=None):
    if video_name is None:
        video_name = os.path.basename(motion_csv_path).replace("_motion.csv", "")

    second_data = defaultdict(list)
    with open(motion_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sec = int(float(row["second"]))
            second_data[sec].append(float(row["motion_ratio"]))

    rows = []
    for sec in sorted(second_data):
        ratio_vals = second_data[sec]
        if not ratio_vals:
            continue
        avg_ratio = round(sum(ratio_vals) / len(ratio_vals), 6)
        motion_std = round(statistics.pstdev(ratio_vals), 6) if len(ratio_vals) > 1 else 0
        max_ratio = round(max(ratio_vals), 6)
        min_ratio = round(min(ratio_vals), 6)
        motion_range = round(max_ratio - min_ratio, 6)
        rows.append(
            {
                "video": video_name,
                "second": sec,
                "avg_motion_ratio": avg_ratio,
                "motion_std": motion_std,
                "max_motion_ratio": max_ratio,
                "min_motion_ratio": min_ratio,
                "motion_range": motion_range,
            }
        )
    return rows


def upsert_motion_aggregated(rows, output_file=OUTPUT_FILE):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    existing = {}

    if os.path.exists(output_file):
        with open(output_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing[(row["video"], int(row["second"]))] = row

    for row in rows:
        key = (row["video"], int(row["second"]))
        existing[key] = {k: row[k] for k in FIELDNAMES}

    merged_rows = [existing[key] for key in sorted(existing.keys(), key=lambda x: (x[0], x[1]))]
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(merged_rows)

    return output_file


def aggregate_all_motion(input_dir=INPUT_DIR, output_file=OUTPUT_FILE):
    all_rows = []
    for file_name in os.listdir(input_dir):
        if not file_name.endswith(".csv"):
            continue
        file_path = os.path.join(input_dir, file_name)
        all_rows.extend(aggregate_motion_csv(file_path))
    upsert_motion_aggregated(all_rows, output_file=output_file)
    return output_file


if __name__ == "__main__":
    aggregate_all_motion()
    print("Motion aggregation complete.")
    print(f"Saved to {OUTPUT_FILE}")
