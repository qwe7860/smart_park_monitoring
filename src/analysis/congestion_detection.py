import csv
import statistics
from collections import defaultdict

INPUT_FILE = "data/processed/motion_aggregated.csv"
OUTPUT_FILE = "data/processed/congestion_windows.csv"

video_motion = defaultdict(list)

with open(INPUT_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        video_motion[row["video"]].append(
            (int(row["second"]), float(row["avg_motion"]))
        )

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["video", "start_sec", "end_sec"])

    for video, data in video_motion.items():
        motions = [m for _, m in data]
        mean = statistics.mean(motions)
        std = statistics.stdev(motions) if len(motions) > 1 else 0
        threshold = mean + std

        peaks = [sec for sec, m in data if m > threshold]

        # find consecutive peak windows
        start = None
        prev = None
        for sec in peaks:
            if start is None:
                start = sec
                prev = sec
            elif sec == prev + 1:
                prev = sec
            else:
                if prev - start >= 2:
                    writer.writerow([video, start, prev])
                start = sec
                prev = sec

        if start is not None and prev - start >= 2:
            writer.writerow([video, start, prev])

print("Congestion windows saved.")