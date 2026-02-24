# Smart Park Usage & Crowd Behaviour Monitoring System

## Project Overview
AI-powered offline system for monitoring park usage patterns, crowd density, and visitor activities in urban parks of Guwahati.

## Objectives
- Monitor park usage patterns
- Analyze crowd density and movement
- Classify visitor activities (walking, sitting, playing, exercising)
- Identify peak hours, congestion zones, and underutilized areas
- Provide safety, cleanliness, and management insights

## Tech Stack
- Python 3.x
- OpenCV
- TensorFlow/PyTorch
- YOLOv8 (Object Detection)
- Streamlit (Dashboard)

## Installation
```bash
pip install -r requirements.txt
```

## Folder Map
- `data/raw_videos/`: uploaded/source videos
- `data/processed/`: generated CSVs, stats, plots
- `models/`: trained model artifacts
- `src/dashboard/`: Streamlit app
- `src/pipeline/`: end-to-end workflow orchestration
- `src/detection/`: people detection logic
- `src/preprocessing/`: motion/features preprocessing
- `src/ml_pipeline/`: train/predict model modules
- `src/analysis/`: summary statistics and analysis outputs
- `archive/notebooks_old/`: old experiments (archived)
- `archive/reports_old/`: old report files (archived)

## Project Status
In active development.

## Author
Iftikhar Ali Sarkar, Chinmoy Barman - Internship Project 2026

## License
MIT License
