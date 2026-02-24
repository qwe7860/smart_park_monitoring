import os
import csv

INPUT_DIR = "data/processed/people_per_second"
OUTPUT_FILE = "data/processed/crowd_statistics.csv"

FIELDNAMES = ["video", "avg_people", "max_people", "peak_second"]


def _stats_for_people_csv(file_path, video_name):
    people_counts = []
    peak_second = 0
    max_people = 0

    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sec = int(row["second"])
            people = int(row["people_count"])
            people_counts.append(people)
            if people > max_people:
                max_people = people
                peak_second = sec

    if not people_counts:
        return None

    return {
        "video": video_name,
        "avg_people": round(sum(people_counts) / len(people_counts), 2),
        "max_people": max_people,
        "peak_second": peak_second,
    }


def compute_crowd_statistics(input_dir=INPUT_DIR, output_file=OUTPUT_FILE):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    rows = []
    for file_name in os.listdir(input_dir):
        if not file_name.endswith(".csv"):
            continue
        video_name = file_name.replace("_people.csv", "")
        stat = _stats_for_people_csv(os.path.join(input_dir, file_name), video_name)
        if stat:
            rows.append(stat)

    rows = sorted(rows, key=lambda x: x["video"])
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return output_file


def upsert_crowd_statistics_for_video(
    video_name,
    input_dir=INPUT_DIR,
    output_file=OUTPUT_FILE,
):
    target_file = os.path.join(input_dir, f"{video_name}_people.csv")
    if not os.path.exists(target_file):
        return output_file

    new_stat = _stats_for_people_csv(target_file, video_name)
    existing = []
    if os.path.exists(output_file):
        with open(output_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["video"] != video_name:
                    existing.append(
                        {
                            "video": row["video"],
                            "avg_people": float(row["avg_people"]),
                            "max_people": int(row["max_people"]),
                            "peak_second": int(row["peak_second"]),
                        }
                    )

    if new_stat:
        existing.append(new_stat)
    existing = sorted(existing, key=lambda x: x["video"])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(existing)
    return output_file


if __name__ == "__main__":
    compute_crowd_statistics()
    print("Crowd statistics complete.")
    print(f"Saved to {OUTPUT_FILE}")
