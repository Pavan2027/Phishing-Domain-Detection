# training/train_svm.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (classification_report,
                              roc_auc_score,
                              confusion_matrix)
from training.split import load_and_split

def train_svm():
    X_train, X_test, y_train, y_test = load_and_split()

    # SVM is O(n²) — subsample train set to keep it fast
    MAX_TRAIN = 10000
    if len(X_train) > MAX_TRAIN:
        print(f"[SVM] Subsampling train to {MAX_TRAIN:,} rows for speed...")

        # Stratified subsample: 5k per class using index alignment
        # Avoids pandas groupby().apply() dropping the groupby column in newer versions
        per_class = MAX_TRAIN // 2
        idx_0 = y_train[y_train == 0].sample(n=min(per_class, (y_train == 0).sum()), random_state=42).index
        idx_1 = y_train[y_train == 1].sample(n=min(per_class, (y_train == 1).sum()), random_state=42).index
        idx   = idx_0.append(idx_1)

        X_train = X_train.loc[idx]
        y_train = y_train.loc[idx]
        print(f"[SVM] Subsample class distribution: {y_train.value_counts().to_dict()}")

    print("\n[SVM] Starting GridSearchCV...")
    param_grid = {
        "C":      [0.1, 1, 10],
        "kernel": ["rbf", "linear"],
        "gamma":  ["scale", "auto"],
    }

    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        SVC(probability=True, random_state=42),
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

    print("\n=== SVM Results ===")
    print(f"Best params : {grid.best_params_}")
    print(f"ROC-AUC     : {auc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred,
                                 target_names=["legit", "phishing"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(best, "models/svm_model.pkl")
    print("Saved → models/svm_model.pkl")

    return best, auc

if __name__ == "__main__":
    train_svm()