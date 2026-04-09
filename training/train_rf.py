# training/train_rf.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (classification_report,
                              roc_auc_score,
                              confusion_matrix)
from training.split import load_and_split

def train_random_forest():
    X_train, X_test, y_train, y_test = load_and_split()

    print("\n[RF] Starting GridSearchCV...")
    param_grid = {
        "n_estimators": [100, 300],
        "max_depth":    [10, 20, None],
        "min_samples_split": [2, 5],
    }

    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ),
        param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)

    best   = grid.best_estimator_
    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)

    print("\n=== Random Forest Results ===")
    print(f"Best params : {grid.best_params_}")
    print(f"ROC-AUC     : {auc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred,
                                 target_names=["legit", "phishing"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(best, "models/rf_model.pkl")
    print("Saved → models/rf_model.pkl")

    return best, auc

if __name__ == "__main__":
    train_random_forest()