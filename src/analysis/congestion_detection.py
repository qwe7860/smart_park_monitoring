import os
import csv

# =========================
# CONFIG (EDITABLE)
# =========================
PEOPLE_THRESHOLD = 7      # congestion if >= 10 people
DURATION_THRESHOLD = 10    # for at least 10 consecutive seconds

INPUT_DIR = "data/processed/people_per_second"
OUTPUT_FILE = "data/processed/congestion_windows.csv"

FIELDNAMES = ["video", "start_second", "end_second", "duration_seconds", "max_people"]


def detect_congestion_windows_for_video(
    people_csv_path,
    video_name,
    people_threshold=PEOPLE_THRESHOLD,
    duration_threshold=DURATION_THRESHOLD,
):
    seconds_data = {}
    with open(people_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seconds_data[int(row["second"])] = int(row["people_count"])

    sorted_seconds = sorted(seconds_data.keys())
    windows = []
    current_start = None
    consecutive_count = 0
    max_people_in_window = 0

    for sec in sorted_seconds:
        people = seconds_data[sec]
        if people >= people_threshold:
            if current_start is None:
                current_start = sec
                consecutive_count = 1
                max_people_in_window = people
            else:
                consecutive_count += 1
                max_people_in_window = max(max_people_in_window, people)
        else:
            if consecutive_count >= duration_threshold:
                windows.append(
                    {
                        "video": video_name,
                        "start_second": current_start,
                        "end_second": sec - 1,
                        "duration_seconds": consecutive_count,
                        "max_people": max_people_in_window,
                    }
                )
            current_start = None
            consecutive_count = 0
            max_people_in_window = 0

    if sorted_seconds and consecutive_count >= duration_threshold:
        windows.append(
            {
                "video": video_name,
                "start_second": current_start,
                "end_second": sorted_seconds[-1],
                "duration_seconds": consecutive_count,
                "max_people": max_people_in_window,
            }
        )
    return windows


def detect_congestion_all_videos(
    input_dir=INPUT_DIR,
    output_file=OUTPUT_FILE,
    people_threshold=PEOPLE_THRESHOLD,
    duration_threshold=DURATION_THRESHOLD,
):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    all_windows = []
    for file_name in os.listdir(input_dir):
        if not file_name.endswith(".csv"):
            continue
        video_name = file_name.replace("_people.csv", "")
        file_path = os.path.join(input_dir, file_name)
        all_windows.extend(
            detect_congestion_windows_for_video(
                file_path,
                video_name,
                people_threshold=people_threshold,
                duration_threshold=duration_threshold,
            )
        )
    _write_windows(all_windows, output_file)
    return output_file


def upsert_congestion_for_video(
    video_name,
    input_dir=INPUT_DIR,
    output_file=OUTPUT_FILE,
    people_threshold=PEOPLE_THRESHOLD,
    duration_threshold=DURATION_THRESHOLD,
):
    people_csv = os.path.join(input_dir, f"{video_name}_people.csv")
    if not os.path.exists(people_csv):
        return output_file

    existing = []
    if os.path.exists(output_file):
        with open(output_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["video"] != video_name:
                    existing.append(
                        {
                            "video": row["video"],
                            "start_second": int(row["start_second"]),
                            "end_second": int(row["end_second"]),
                            "duration_seconds": int(row["duration_seconds"]),
                            "max_people": int(row["max_people"]),
                        }
                    )

    existing.extend(
        detect_congestion_windows_for_video(
            people_csv,
            video_name,
            people_threshold=people_threshold,
            duration_threshold=duration_threshold,
        )
    )
    _write_windows(existing, output_file)
    return output_file


def _write_windows(rows, output_file):
    rows = sorted(rows, key=lambda x: (x["video"], x["start_second"]))
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    detect_congestion_all_videos()
    print("Congestion detection complete.")
    print(f"Saved to {OUTPUT_FILE}")
