# evaluate/shap_analysis.py
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

def run_shap(sample_n: int = 500):
    """
    Uses TreeExplainer (fast, exact for XGBoost/RF).
    Samples test set to keep runtime reasonable.
    """
    # Use XGB explicitly — SHAP TreeExplainer is built for it
    model    = joblib.load("models/xgb_model.pkl")
    features = joblib.load("models/selected_features.pkl")

    test_df  = pd.read_csv("data/test_set.csv")
    X_test   = test_df.drop(columns=["label"])[features]

    # Sample for speed — full set can take a few minutes
    X_sample = X_test.sample(n=min(sample_n, len(X_test)), random_state=42)

    print(f"Running SHAP on {len(X_sample)} samples...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    os.makedirs("evaluate/outputs", exist_ok=True)

    # Beeswarm summary (shows direction of each feature's effect)
    plt.figure()
    shap.summary_plot(shap_values, X_sample,
                      plot_type="dot", show=False)
    plt.tight_layout()
    plt.savefig("evaluate/outputs/shap_beeswarm.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → evaluate/outputs/shap_beeswarm.png")

    # Bar (mean absolute SHAP — global importance)
    plt.figure()
    shap.summary_plot(shap_values, X_sample,
                      plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("evaluate/outputs/shap_bar.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → evaluate/outputs/shap_bar.png")

    # Top-10 feature importance from SHAP
    mean_abs = pd.Series(
        abs(shap_values).mean(axis=0),
        index=features
    ).sort_values(ascending=False)

    print("\n=== Top 10 features by mean |SHAP| ===")
    print(mean_abs.head(10).to_string())

if __name__ == "__main__":
    run_shap()