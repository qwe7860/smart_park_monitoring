import csv
import os

MOTION_FILE = "data/processed/motion_aggregated.csv"
PEOPLE_DIR = "data/processed/people_per_second"
OUTPUT_FILE = "data/processed/master_dataset.csv"
FIELDNAMES = ["video", "second", "avg_motion_ratio", "motion_std", "people_count"]


def _load_motion_data(motion_file=MOTION_FILE):
    motion_data = {}
    with open(motion_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["video"], int(row["second"]))
            motion_data[key] = {
                "avg_motion_ratio": float(row["avg_motion_ratio"]),
                "motion_std": float(row["motion_std"]),
            }
    return motion_data


def _load_people_data(people_dir=PEOPLE_DIR):
    people_data = {}
    for file_name in os.listdir(people_dir):
        if not file_name.endswith(".csv"):
            continue
        file_path = os.path.join(people_dir, file_name)
        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["video"], int(row["second"]))
                people_data[key] = int(row["people_count"])
    return people_data


def _write_master_rows(rows, output_file=OUTPUT_FILE):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def build_master_dataset(
    motion_file=MOTION_FILE,
    people_dir=PEOPLE_DIR,
    output_file=OUTPUT_FILE,
):
    motion_data = _load_motion_data(motion_file)
    people_data = _load_people_data(people_dir)
    all_keys = set(motion_data.keys()) | set(people_data.keys())
    sorted_keys = sorted(all_keys, key=lambda x: (x[0], x[1]))

    rows = []
    for video, second in sorted_keys:
        motion_info = motion_data.get((video, second), {})
        rows.append(
            {
                "video": video,
                "second": second,
                "avg_motion_ratio": round(motion_info.get("avg_motion_ratio", 0.0), 6),
                "motion_std": round(motion_info.get("motion_std", 0.0), 6),
                "people_count": people_data.get((video, second), 0),
            }
        )

    _write_master_rows(rows, output_file=output_file)
    return output_file


def upsert_master_dataset_for_video(
    video_name,
    motion_file=MOTION_FILE,
    people_dir=PEOPLE_DIR,
    output_file=OUTPUT_FILE,
):
    motion_data = _load_motion_data(motion_file)
    people_data = _load_people_data(people_dir)
    existing = {}

    if os.path.exists(output_file):
        with open(output_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["video"], int(row["second"]))
                if row["video"] != video_name:
                    existing[key] = {
                        "video": row["video"],
                        "second": int(row["second"]),
                        "avg_motion_ratio": float(row["avg_motion_ratio"]),
                        "motion_std": float(row["motion_std"]),
                        "people_count": int(row["people_count"]),
                    }

    video_keys = {key for key in set(motion_data) | set(people_data) if key[0] == video_name}
    for video, second in sorted(video_keys, key=lambda x: (x[0], x[1])):
        motion_info = motion_data.get((video, second), {})
        existing[(video, second)] = {
            "video": video,
            "second": second,
            "avg_motion_ratio": round(motion_info.get("avg_motion_ratio", 0.0), 6),
            "motion_std": round(motion_info.get("motion_std", 0.0), 6),
            "people_count": people_data.get((video, second), 0),
        }

    rows = [existing[key] for key in sorted(existing.keys(), key=lambda x: (x[0], x[1]))]
    _write_master_rows(rows, output_file=output_file)
    return output_file


if __name__ == "__main__":
    build_master_dataset()
    print("Master dataset created successfully.")
    print(f"Saved to {OUTPUT_FILE}")
