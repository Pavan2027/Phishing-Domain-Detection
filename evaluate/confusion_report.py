# evaluate/confusion_report.py
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
import os

def full_report(threshold: float = 0.5):
    model    = joblib.load("models/best_model.pkl")
    features = joblib.load("models/selected_features.pkl")
    scaler   = joblib.load("models/scaler.pkl")

    test_df = pd.read_csv("data/test_set.csv")
    X_test  = test_df.drop(columns=["label"])[features]
    y_test  = test_df["label"].astype(int)

    X_scaled = scaler.transform(X_test)
    probs    = model.predict_proba(X_scaled)[:, 1]
    preds    = (probs >= threshold).astype(int)

    os.makedirs("evaluate/outputs", exist_ok=True)

    print(f"\n=== Classification Report (threshold={threshold}) ===")
    print(classification_report(y_test, preds,
                                target_names=["legit", "phishing"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, probs):.4f}")

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["legit", "phishing"],
                yticklabels=["legit", "phishing"])
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title(f"Confusion Matrix  (threshold={threshold})")
    plt.tight_layout()
    plt.savefig("evaluate/outputs/confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved → evaluate/outputs/confusion_matrix.png")

    # --- ROC curve ---
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — XGBoost")
    plt.legend()
    plt.tight_layout()
    plt.savefig("evaluate/outputs/roc_curve.png", dpi=150)
    plt.close()
    print("Saved → evaluate/outputs/roc_curve.png")

if __name__ == "__main__":
    full_report()