import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

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

# =========================
# LOAD DATA
# =========================
crowd_stats = pd.read_csv(CROWD_STATS_PATH)
activity_dist = pd.read_csv(ACTIVITY_DIST_PATH)

if os.path.exists(CONGESTION_PATH):
    congestion = pd.read_csv(CONGESTION_PATH)
else:
    congestion = pd.DataFrame()

video_list = crowd_stats["video"].unique()
selected_video = st.selectbox("Select Video", video_list)

video_stats = crowd_stats[crowd_stats["video"] == selected_video]
video_activity = activity_dist[activity_dist["video"] == selected_video]

# =========================
# UTILIZATION SCORE
# =========================
if not video_activity.empty:
    sitting = video_activity["sitting_percent"].values[0]
    walking = video_activity["walking_percent"].values[0]
    high = video_activity["high_activity_percent"].values[0]

    activity_score = (walking * 2 + high * 3 + sitting * 1) / 100
else:
    activity_score = 0

avg_people = float(video_stats["avg_people"])
max_people = int(video_stats["max_people"])

crowd_score = avg_people / max_people if max_people > 0 else 0

utilization_score = round((activity_score * 0.6) + (crowd_score * 0.4), 2)

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Crowd Analysis",
    "Activity Analysis",
    "Congestion",
    "ML Insights"
])

# =========================
# OVERVIEW TAB
# =========================
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
        col4.metric("Dominant Activity", video_activity["dominant_activity"].values[0])
    else:
        col4.metric("Dominant Activity", "N/A")

    st.metric("Utilization Score (0-1 Scale)", utilization_score)

    st.divider()
    st.subheader("Urban Insights")

    insight_text = ""

    if utilization_score > 0.7:
        insight_text += "• Park is highly utilized with strong engagement levels.\n"
    elif utilization_score > 0.4:
        insight_text += "• Park shows moderate utilization with balanced activity.\n"
    else:
        insight_text += "• Park appears underutilized during recorded period.\n"

    if congestion_count > 0:
        insight_text += "• Congestion events detected. Management intervention may be required during peak times.\n"
    else:
        insight_text += "• No sustained congestion observed.\n"

    if video_activity["dominant_activity"].values[0] == "sitting":
        insight_text += "• Area is primarily used for passive activities (resting/socializing).\n"
    elif video_activity["dominant_activity"].values[0] == "walking":
        insight_text += "• Area is primarily used for movement-based activities.\n"
    else:
        insight_text += "• Area shows high physical engagement.\n"

    st.info(insight_text)

# =========================
# CROWD ANALYSIS TAB
# =========================
with tab2:
    st.subheader("Crowd Over Time")

    people_file = os.path.join(
        PEOPLE_DIR,
        selected_video + "_people.csv"
    )

    people_df = pd.read_csv(people_file)

    fig, ax = plt.subplots()

    ax.plot(
        people_df["second"],
        people_df["people_count"],
        linewidth=2
    )

    ax.set_xlabel("Second")
    ax.set_ylabel("People Count")
    ax.set_title("Crowd Density Over Time")

    fig.set_size_inches(10, 3)   # FORCE SHORT HEIGHT
    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# =========================
# ACTIVITY ANALYSIS TAB
# =========================
with tab3:
    st.subheader("Activity Distribution")

    if not video_activity.empty:

        labels = ["Sitting", "Walking", "High Activity"]
        sizes = [sitting, walking, high]

        fig, ax = plt.subplots()

        ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90
        )

        ax.set_title("Activity Breakdown")

        fig.set_size_inches(5, 4)  # Compact size
        fig.tight_layout()

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.pyplot(fig)
        plt.close(fig)

    else:
        st.write("No activity data available.")

# =========================
# CONGESTION TAB
# =========================
with tab4:
    st.subheader("Congestion Events")

    if not congestion.empty:
        video_congestion = congestion[congestion["video"] == selected_video]
        st.dataframe(video_congestion)
    else:
        st.write("No congestion detected.")

# =========================
# ML INSIGHTS TAB
# =========================
with tab5:
    st.subheader("Feature Importance")

    if os.path.exists(FEATURE_IMPORTANCE_IMG):
        st.image(FEATURE_IMPORTANCE_IMG)
    else:
        st.write("Feature importance image not found.")