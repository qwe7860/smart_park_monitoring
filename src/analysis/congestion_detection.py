# import csv
# import statistics
# from collections import defaultdict

# INPUT_FILE = "data/processed/motion_aggregated.csv"
# OUTPUT_FILE = "data/processed/congestion_windows.csv"

# video_motion = defaultdict(list)

# with open(INPUT_FILE, newline="", encoding="utf-8") as f:
#     reader = csv.DictReader(f)
#     for row in reader:
#         video_motion[row["video"]].append(
#             (int(row["second"]), float(row["avg_motion_ratio"]))
#         )

# with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
#     writer = csv.writer(f)
#     writer.writerow(["video", "start_sec", "end_sec"])

#     for video, data in video_motion.items():
#         motions = [m for _, m in data]
#         mean = statistics.mean(motions)
#         std = statistics.stdev(motions) if len(motions) > 1 else 0
#         threshold = mean + std

#         peaks = [sec for sec, m in data if m > threshold]

#         # find consecutive peak windows
#         start = None
#         prev = None
#         for sec in peaks:
#             if start is None:
#                 start = sec
#                 prev = sec
#             elif sec == prev + 1:
#                 prev = sec
#             else:
#                 if prev - start >= 2:
#                     writer.writerow([video, start, prev])
#                 start = sec
#                 prev = sec

#         if start is not None and prev - start >= 2:
#             writer.writerow([video, start, prev])

# print("Congestion windows saved.")

import os
import csv

# =========================
# CONFIG (EDITABLE)
# =========================
PEOPLE_THRESHOLD = 10      # congestion if >= 15 people
DURATION_THRESHOLD = 10    # for at least 10 consecutive seconds

INPUT_DIR = "data/processed/people_per_second"
OUTPUT_FILE = "data/processed/congestion_windows.csv"

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

all_windows = []

# =========================
# PROCESS EACH VIDEO
# =========================
for file_name in os.listdir(INPUT_DIR):

    if not file_name.endswith(".csv"):
        continue

    video_name = file_name.replace("_people.csv", "")
    file_path = os.path.join(INPUT_DIR, file_name)

    seconds_data = {}

    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sec = int(row["second"])
            people = int(row["people_count"])
            seconds_data[sec] = people

    sorted_seconds = sorted(seconds_data.keys())

    current_start = None
    consecutive_count = 0
    max_people_in_window = 0

    for sec in sorted_seconds:

        people = seconds_data[sec]

        if people >= PEOPLE_THRESHOLD:
            if current_start is None:
                current_start = sec
                consecutive_count = 1
                max_people_in_window = people
            else:
                consecutive_count += 1
                max_people_in_window = max(max_people_in_window, people)

        else:
            # End of a potential window
            if consecutive_count >= DURATION_THRESHOLD:
                all_windows.append([
                    video_name,
                    current_start,
                    sec - 1,
                    consecutive_count,
                    max_people_in_window
                ])

            # Reset
            current_start = None
            consecutive_count = 0
            max_people_in_window = 0

    # Check if window ended at last second
    if consecutive_count >= DURATION_THRESHOLD:
        all_windows.append([
            video_name,
            current_start,
            sorted_seconds[-1],
            consecutive_count,
            max_people_in_window
        ])

# =========================
# WRITE OUTPUT
# =========================
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow([
        "video",
        "start_second",
        "end_second",
        "duration_seconds",
        "max_people"
    ])

    writer.writerows(all_windows)

print("Congestion detection complete.")
print(f"Saved to {OUTPUT_FILE}")