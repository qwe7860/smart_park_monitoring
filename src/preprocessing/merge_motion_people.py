import csv
import os

MOTION_FILE = "data/processed/motion_aggregated.csv"
PEOPLE_DIR = "data/processed/people_per_second"
OUTPUT_FILE = "data/processed/master_dataset.csv"

# -------------------------
# Load Motion Data
# -------------------------
motion_data = {}

with open(MOTION_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row["video"], int(row["second"]))
        motion_data[key] = {
            "avg_motion_ratio": float(row["avg_motion_ratio"]),
            "motion_std": float(row["motion_std"])
        }

# -------------------------
# Load People Data
# -------------------------
people_data = {}

for file_name in os.listdir(PEOPLE_DIR):

    if not file_name.endswith(".csv"):
        continue

    file_path = os.path.join(PEOPLE_DIR, file_name)

    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["video"], int(row["second"]))
            people_data[key] = int(row["people_count"])

# -------------------------
# Merge Using Union of Keys
# -------------------------
all_keys = set(motion_data.keys()) | set(people_data.keys())

# Sort properly by video then second
sorted_keys = sorted(all_keys, key=lambda x: (x[0], x[1]))

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow([
        "video",
        "second",
        "avg_motion_ratio",
        "motion_std",
        "people_count"
    ])

    for video, second in sorted_keys:

        motion_info = motion_data.get((video, second), {})
        avg_motion_ratio = motion_info.get("avg_motion_ratio", 0.0)
        motion_std = motion_info.get("motion_std", 0.0)

        people_count = people_data.get((video, second), 0)

        writer.writerow([
            video,
            second,
            round(avg_motion_ratio, 6),
            round(motion_std, 6),
            people_count
        ])

print("Master dataset created successfully.")
print(f"Saved to {OUTPUT_FILE}")