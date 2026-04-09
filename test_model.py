# test_model.py
import joblib
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from features.lexical import extract_lexical

model    = joblib.load("models/rf_model.pkl")
features = joblib.load("models/selected_features.pkl")
scaler   = joblib.load("models/scaler.pkl")

TEST_URLS = [
    # obvious phishing
    "http://paypal-secure-login.xyz/verify/account?id=1234",
    "http://amazon-account-update.tk/login.php",
    "http://192.168.1.1/bank/login",

    # legitimate
    "https://www.google.com",
    "https://www.amazon.com/products",
    "https://github.com/login",

    # tricky edge cases
    "https://paypal.com/signin",           # real paypal
    "http://paypal.com.evil-site.xyz",     # fake paypal
]

print(f"{'URL':<55} {'Prediction':>12} {'Confidence':>12}")
print("-" * 80)

for url in TEST_URLS:
    feats = extract_lexical(url)
    row   = pd.DataFrame([feats])

    # align to selected features, fill missing with -1
    for f in features:
        if f not in row.columns:
            row[f] = -1
    row = row[features].fillna(-1)

    row_scaled = pd.DataFrame(scaler.transform(row), columns=row.columns)
    pred       = model.predict(row_scaled)[0]
    prob       = model.predict_proba(row_scaled)[0][1]
    label      = "PHISHING" if pred == 1 else "legit"

    # truncate URL for display
    display_url = url[:52] + "..." if len(url) > 55 else url
    print(f"{display_url:<55} {label:>12} {prob:>11.1%}")