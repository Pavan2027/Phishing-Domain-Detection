# evaluate/lime_explain.py
import lime
import lime.lime_tabular
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from features.lexical import extract_lexical
import os

def explain_url(url: str, num_features: int = 10):
    model    = joblib.load("models/best_model.pkl")
    features = joblib.load("models/selected_features.pkl")
    scaler   = joblib.load("models/scaler.pkl")

    # LIME needs training-distribution background data
    test_df  = pd.read_csv("data/test_set.csv")
    X_bg     = test_df.drop(columns=["label"])[features].values

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data  = X_bg,
        feature_names  = features,
        class_names    = ["legit", "phishing"],
        mode           = "classification",
        discretize_continuous = True,
        random_state   = 42
    )

    # Build feature row for URL
    feats = extract_lexical(url)
    row   = pd.DataFrame([feats])
    for f in features:
        if f not in row.columns:
            row[f] = -1
    row       = row[features].fillna(-1)
    row_scaled = scaler.transform(row)

    exp = explainer.explain_instance(
        row_scaled[0],
        model.predict_proba,
        num_features = num_features
    )

    prob = model.predict_proba(row_scaled)[0][1]
    label = "PHISHING" if prob >= 0.5 else "legit"
    print(f"\nURL     : {url}")
    print(f"Decision: {label}  (confidence {prob:.1%})")
    print("\nTop contributing features:")
    for feat, weight in exp.as_list():
        direction = "↑ phishing" if weight > 0 else "↓ legit"
        print(f"  {feat:<45}  {weight:+.4f}  {direction}")

    os.makedirs("evaluate/outputs", exist_ok=True)
    safe_name = url[:40].replace("/", "_").replace(":", "").replace(".", "_")
    out_path  = f"evaluate/outputs/lime_{safe_name}.png"
    fig = exp.as_pyplot_figure()
    plt.title(f"LIME — {url[:60]}", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")
    return exp

if __name__ == "__main__":
    # Test on one known phishing + one known legit
    explain_url("http://paypal-secure-login.xyz/verify/account?id=1234")
    explain_url("https://github.com/login")