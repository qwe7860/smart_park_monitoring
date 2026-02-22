import os
import csv

INPUT_DIR = "data/processed/people_per_second"
OUTPUT_FILE = "data/processed/crowd_statistics.csv"

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

stats_rows = []

for file_name in os.listdir(INPUT_DIR):

    if not file_name.endswith(".csv"):
        continue

    video_name = file_name.replace("_people.csv", "")
    file_path = os.path.join(INPUT_DIR, file_name)

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

    if len(people_counts) == 0:
        continue

    avg_people = round(sum(people_counts) / len(people_counts), 2)

    stats_rows.append([
        video_name,
        avg_people,
        max_people,
        peak_second
    ])

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow([
        "video",
        "avg_people",
        "max_people",
        "peak_second"
    ])

    writer.writerows(stats_rows)

print("Crowd statistics complete.")
print(f"Saved to {OUTPUT_FILE}")