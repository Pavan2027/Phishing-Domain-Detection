# training/split.py
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_and_split(path: str = "data/processed_features.csv"):
    df = pd.read_csv(path)

    X = df.drop(columns=["label"])
    y = df["label"].astype(int)

    print(f"Full dataset: {X.shape[0]:,} rows, {X.shape[1]} features")
    print(f"Label distribution:\n{y.value_counts().to_string()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")

    counts   = y_train.value_counts()
    majority = counts.max()
    minority = counts.min()
    ratio    = majority / minority

    # Threshold raised to 4.0 — intentional augmentation creates a 2:1 ratio
    # which is acceptable for tree-based models. SMOTE at 2:1 generates
    # large volumes of synthetic phishing samples that poison the boundary.
    # Only apply SMOTE for severe imbalance (original data without augmentation
    # would trigger this if needed).
    if ratio > 4.0:
        print(f"Imbalance ratio {ratio:.2f} — applying SMOTE...")
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(f"After SMOTE: {pd.Series(y_train).value_counts().to_string()}")
    else:
        print(f"Ratio {ratio:.2f}:1 — balance acceptable, skipping SMOTE")

    # Save test set for evaluation
    test_df = X_test.copy()
    test_df["label"] = y_test.values
    test_df.to_csv("data/test_set.csv", index=False)
    print("Test set saved → data/test_set.csv")

    return X_train, X_test, y_train, y_test