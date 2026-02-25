import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Ensure project root is importable when running via `streamlit run src/dashboard/app.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.video_workflow import (
    run_analysis_for_video,
    save_uploaded_video,
    self_train_after_upload,
)

# =========================
# CONFIG PATHS
# =========================
CROWD_STATS_PATH = "data/processed/crowd_statistics.csv"
CONGESTION_PATH = "data/processed/congestion_windows.csv"
ACTIVITY_DIST_PATH = "data/processed/activity_distribution.csv"
FEATURE_IMPORTANCE_IMG = "data/processed/feature_importance.png"
PEOPLE_DIR = "data/processed/people_per_second"

st.set_page_config(layout="wide")
st.title("Smart Park Monitoring System")
st.caption("Upload a new park video, analyze it, and optionally self-train the model.")


def safe_read_csv(path, required_columns=None):
    if not os.path.exists(path):
        return pd.DataFrame(columns=required_columns or [])
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=required_columns or [])


def rerun_app():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def load_dashboard_data():
    crowd_stats = safe_read_csv(
        CROWD_STATS_PATH,
        required_columns=["video", "avg_people", "max_people", "peak_second"],
    )
    activity_dist = safe_read_csv(
        ACTIVITY_DIST_PATH,
        required_columns=[
            "video",
            "sitting_percent",
            "walking_percent",
            "high_activity_percent",
            "dominant_activity",
        ],
    )
    congestion = safe_read_csv(
        CONGESTION_PATH,
        required_columns=[
            "video",
            "start_second",
            "end_second",
            "duration_seconds",
            "max_people",
        ],
    )
    return crowd_stats, activity_dist, congestion


def upload_and_analyze_ui():
    st.subheader("Upload and Analyze New Video")
    left, center, right = st.columns([1, 2, 1])

    with center:
        uploaded_video = st.file_uploader(
            "Upload video",
            type=["mp4", "mov", "avi", "mkv"],
        )

        col_upload_1, col_upload_2 = st.columns(2)
        with col_upload_1:
            analyze_clicked = st.button("Analyze Uploaded Video", type="primary")
        with col_upload_2:
            self_train_clicked = st.button("Analyze + Self-Train Model")

        if uploaded_video is not None and (analyze_clicked or self_train_clicked):
            try:
                with st.spinner("Saving, analyzing, and updating dataset..."):
                    saved_path, saved_video_name = save_uploaded_video(uploaded_video)
                    analysis_result = run_analysis_for_video(saved_path)
                    st.success(
                        f"Analysis completed for '{saved_video_name}'. "
                        f"Added/updated {analysis_result['dataset_rows']} rows."
                    )
                    st.session_state["selected_video_after_upload"] = saved_video_name

                    if self_train_clicked:
                        train_result = self_train_after_upload(saved_video_name)
                        if train_result["trained"]:
                            st.success(
                                f"Self-training complete. Added {train_result['rows_added']} pseudo-labeled rows."
                            )
                        else:
                            st.warning(f"Self-training skipped: {train_result['reason']}")
                rerun_app()
            except Exception as exc:
                st.error(f"Upload analysis failed: {exc}")


def render_dashboard():
    crowd_stats, activity_dist, congestion = load_dashboard_data()

    if crowd_stats.empty:
        st.warning("No analyzed videos yet. Upload and analyze a video to populate the dashboard.")
        return

    video_list = sorted(crowd_stats["video"].unique())
    default_video = st.session_state.get("selected_video_after_upload", video_list[0])
    default_index = video_list.index(default_video) if default_video in video_list else 0
    selected_video = st.selectbox("Select Video", video_list, index=default_index)

    video_stats = crowd_stats[crowd_stats["video"] == selected_video]
    video_activity = activity_dist[activity_dist["video"] == selected_video]

    if not video_activity.empty:
        sitting = float(video_activity["sitting_percent"].iloc[0])
        walking = float(video_activity["walking_percent"].iloc[0])
        high = float(video_activity["high_activity_percent"].iloc[0])
        # Normalize weighted activity score to 0-1 range.
        activity_score = ((walking * 2 + high * 3 + sitting) / 100) / 3
    else:
        sitting, walking, high = 0.0, 0.0, 0.0
        activity_score = 0

    avg_people = float(video_stats["avg_people"].iloc[0]) if not video_stats.empty else 0.0
    max_people = int(video_stats["max_people"].iloc[0]) if not video_stats.empty else 0
    crowd_score = avg_people / max_people if max_people > 0 else 0
    utilization_score = round((activity_score * 0.6) + (crowd_score * 0.4), 2)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Overview", "Crowd Analysis", "Activity Analysis", "Congestion", "ML Insights"]
    )

    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Average Crowd", avg_people)
        col2.metric("Peak Crowd", max_people)

        if not congestion.empty:
            congestion_count = len(congestion[congestion["video"] == selected_video])
        else:
            congestion_count = 0

        col3.metric("Congestion Events", congestion_count)
        if not video_activity.empty:
            col4.metric("Dominant Activity", video_activity["dominant_activity"].iloc[0])
        else:
            col4.metric("Dominant Activity", "N/A")

        st.metric("Utilization Score (0-1 Scale)", utilization_score)
        st.divider()
        st.subheader("Urban Insights")

        insights = []
        if utilization_score > 0.7:
            insights.append("- Park is highly utilized with strong engagement levels.")
        elif utilization_score > 0.4:
            insights.append("- Park shows moderate utilization with balanced activity.")
        else:
            insights.append("- Park appears underutilized during recorded period.")

        if congestion_count > 0:
            insights.append(
                "- Congestion events detected. Management intervention may be required during peak times."
            )
        else:
            insights.append("- No sustained congestion observed.")

        dominant = video_activity["dominant_activity"].iloc[0] if not video_activity.empty else "N/A"
        if dominant == "sitting":
            insights.append("- Area is primarily used for passive activities (resting/socializing).")
        elif dominant == "walking":
            insights.append("- Area is primarily used for movement-based activities.")
        else:
            insights.append("- Area shows high physical engagement.")

        st.info("\n".join(insights))

    with tab2:
        st.subheader("Crowd Over Time")
        people_file = os.path.join(PEOPLE_DIR, selected_video + "_people.csv")
        if not os.path.exists(people_file):
            st.warning(f"People data not found for '{selected_video}'.")
        else:
            people_df = pd.read_csv(people_file)
            fig, ax = plt.subplots()
            ax.plot(people_df["second"], people_df["people_count"], linewidth=2)
            ax.set_xlabel("Second")
            ax.set_ylabel("People Count")
            ax.set_title("Crowd Density Over Time")
            fig.set_size_inches(10, 3)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with tab3:
        st.subheader("Activity Distribution")
        if not video_activity.empty:
            labels = ["Sitting", "Walking", "High Activity"]
            sizes = [sitting, walking, high]
            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
            ax.set_title("Activity Breakdown")
            fig.set_size_inches(5, 4)
            fig.tight_layout()
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.pyplot(fig)
            plt.close(fig)
        else:
            st.write("No activity data available.")

    with tab4:
        st.subheader("Congestion Events")
        if not congestion.empty:
            video_congestion = congestion[congestion["video"] == selected_video]
            st.dataframe(video_congestion, use_container_width=True)
        else:
            st.write("No congestion detected.")

    with tab5:
        st.subheader("Feature Importance")
        if os.path.exists(FEATURE_IMPORTANCE_IMG):
            st.image(FEATURE_IMPORTANCE_IMG)
        else:
            st.write("Feature importance image not found.")


tab_dashboard, tab_upload = st.tabs(["Dashboard", "Upload & Analyze"])

with tab_dashboard:
    render_dashboard()

with tab_upload:
    upload_and_analyze_ui()
