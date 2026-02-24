import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# CONFIG
# =========================
MODEL_PATH = "models/activity_rf_model.pkl"
DATA_PATH = "data/processed/master_labeled.csv"
OUTPUT_IMAGE = "data/processed/feature_importance.png"

os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)

def generate_feature_importance_plot(
    model_path=MODEL_PATH,
    data_path=DATA_PATH,
    output_image=OUTPUT_IMAGE,
):
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    feature_columns = ["avg_motion_ratio", "motion_std", "people_count"]
    _ = df[feature_columns]

    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": importances,
    }).sort_values(by="importance", ascending=False)

    os.makedirs(os.path.dirname(output_image), exist_ok=True)
    plt.figure()
    plt.bar(importance_df["feature"], importance_df["importance"])
    plt.xticks(rotation=45)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(output_image)
    plt.close()

    return {
        "output_image": output_image,
        "importance_table": importance_df.to_dict(orient="records"),
    }


if __name__ == "__main__":
    result = generate_feature_importance_plot()
    print("\nFeature Importance Ranking:\n")
    print(result["importance_table"])
    print(f"\nFeature importance plot saved to {OUTPUT_IMAGE}")
