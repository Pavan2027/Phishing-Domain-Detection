# features/preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import joblib, os

# columns that are strings / identifiers — not useful as ML features
DROP_COLS = ["url", "domain", "tld", "registrar",
             "ssl_issuer", "country", "rdap_error"]

# features/preprocess.py  — add this list at the top
LEXICAL_ONLY = [
    "url_length", "domain_length", "path_length",
    "num_dots", "num_hyphens", "num_underscores",
    "num_slashes", "num_at", "num_question", "num_equals",
    "num_ampersand", "num_digits", "digit_ratio",
    "subdomain_count", "has_ip", "has_port",
    "has_at_symbol", "double_slash_path", "has_redirect",
    "url_entropy", "domain_entropy", "suspicious_tld",
    "brand_in_subdomain", "brand_in_domain",
    "num_tokens", "longest_token", "is_trusted_domain",
    "suspicious_word_count", "suspicious_short_domain",
    "brand_typo_detected", "random_domain_flag",
    "trusted_with_redirect"
]

def preprocess(
    matrix_path: str = "data/feature_matrix.csv",
    out_path:    str = "data/processed_features.csv",
    top_n:       int = 35
):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import mutual_info_classif
    import joblib, os

    df = pd.read_csv(matrix_path)
    print(f"Loaded feature matrix: {df.shape}")

    y = df["label"].astype(int)

    # ── use lexical features only ─────────────────────────────────────────
    available = [f for f in LEXICAL_ONLY if f in df.columns]
    X = df[available].copy()
    print(f"Using {len(available)} lexical features only")

    # fill missing
    X = X.fillna(-1)

    # drop zero-variance columns
    before = X.shape[1]
    X = X.loc[:, X.nunique() > 1]
    print(f"Dropped {before - X.shape[1]} constant columns")

    # mutual information
    print("Computing mutual information scores...")
    mi     = mutual_info_classif(X, y, random_state=42)
    mi_ser = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    print(f"\nTop features:")
    print(mi_ser.to_string())

    top_features = mi_ser.head(top_n).index.tolist()
    X = X[top_features]

    # normalize
    scaler   = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), columns=X.columns
    )

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler,       "models/scaler.pkl")
    joblib.dump(top_features, "models/selected_features.pkl")

    out = X_scaled.copy()
    out["label"] = y.values
    out.to_csv(out_path, index=False)
    print(f"\nProcessed features saved → {out_path}")
    print(f"Final shape: {out.shape}")
    return out

if __name__ == "__main__":
    preprocess()