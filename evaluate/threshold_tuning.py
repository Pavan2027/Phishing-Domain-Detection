# evaluate/threshold_tuning.py
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_auc_score
import os

def tune_threshold(min_precision: float = 0.90):
    """
    Finds the lowest threshold where precision stays >= min_precision.
    Phishing detection prioritises recall (catching more phishing),
    so we relax the threshold as far as possible without tanking precision.
    """
    model    = joblib.load("models/best_model.pkl")
    features = joblib.load("models/selected_features.pkl")
    scaler   = joblib.load("models/scaler.pkl")

    test_df  = pd.read_csv("data/test_set.csv")
    X_test   = test_df.drop(columns=["label"])[features]
    y_test   = test_df["label"].astype(int)

    X_scaled = scaler.transform(X_test)
    probs    = model.predict_proba(X_scaled)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_test, probs)

    # Walk from low→high threshold; pick lowest t where precision >= target
    best_t, best_p, best_r = 0.5, 0.0, 0.0
    for t, p, r in zip(thresholds, precision[:-1], recall[:-1]):
        if p >= min_precision and r > best_r:
            best_t, best_p, best_r = t, p, r

    print(f"\n=== Threshold Tuning (min precision = {min_precision}) ===")
    print(f"Recommended threshold : {best_t:.4f}")
    print(f"Precision at threshold: {best_p:.4f}")
    print(f"Recall    at threshold: {best_r:.4f}")
    print(f"(Default 0.5 recall   : "
          f"{recall[np.searchsorted(thresholds, 0.5)]:.4f})")

    os.makedirs("evaluate/outputs", exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(thresholds, precision[:-1], label="Precision", color="steelblue")
    plt.plot(thresholds, recall[:-1],    label="Recall",    color="darkorange")
    plt.axvline(best_t, color="red", linestyle="--",
                label=f"Best threshold ({best_t:.2f})")
    plt.axvline(0.5,    color="gray", linestyle=":",
                label="Default 0.5")
    plt.xlabel("Decision threshold")
    plt.ylabel("Score")
    plt.title("Precision & Recall vs Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig("evaluate/outputs/threshold_tuning.png", dpi=150)
    plt.close()
    print("Saved → evaluate/outputs/threshold_tuning.png")

    return best_t

if __name__ == "__main__":
    tune_threshold()