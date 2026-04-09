import joblib
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from features.lexical import extract_lexical


class PhishingPredictor:
    def __init__(self):
        self.model = joblib.load("models/best_model.pkl")
        self.features = joblib.load("models/selected_features.pkl")
        self.scaler = joblib.load("models/scaler.pkl")
        self.threshold = 0.5

    def predict(self, url: str):
        feats = extract_lexical(url)

        if not feats:
            return {
                "url": url,
                "prediction": "error",
                "confidence": 0.0,
                "is_phishing": False
            }

        row = pd.DataFrame([feats])

        for col in self.features:
            if col not in row.columns:
                row[col] = -1

        row = row[self.features].fillna(-1)
        row_scaled = self.scaler.transform(row)

        prob = float(self.model.predict_proba(row_scaled)[0][1])
        is_phishing = prob >= self.threshold

        return {
            "url": url,
            "prediction": "phishing" if is_phishing else "legit",
            "confidence": round(prob, 4),
            "is_phishing": is_phishing
        }

    def predict_batch(self, urls):
        return [self.predict(url) for url in urls]