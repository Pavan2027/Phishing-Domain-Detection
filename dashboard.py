# dashboard.py  — run with: streamlit run dashboard.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import (
    confusion_matrix, roc_curve,
    roc_auc_score, classification_report
)
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from features.lexical import extract_lexical

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "PhishGuard Dashboard",
    page_icon  = "🎣",
    layout     = "wide"
)

# ── Load model once ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load("models/best_model.pkl")
    features = joblib.load("models/selected_features.pkl")
    scaler   = joblib.load("models/scaler.pkl")
    return model, features, scaler

@st.cache_data
def load_test_data():
    return pd.read_csv("data/test_set.csv")

model, features, scaler = load_model()
test_df = load_test_data()

X_test  = test_df.drop(columns=["label"])[features]
y_test  = test_df["label"].astype(int)
X_scaled = scaler.transform(X_test)
probs    = model.predict_proba(X_scaled)[:, 1]

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.image("extension/icons/icon128.png", width=60)
st.sidebar.title("PhishGuard")
st.sidebar.caption("ML Phishing Detection System")
st.sidebar.markdown("---")

threshold = st.sidebar.slider(
    "Decision threshold",
    min_value=0.1, max_value=0.9,
    value=0.5, step=0.01,
    help="Lower = catch more phishing (more false positives). "
         "Higher = more conservative."
)
preds = (probs >= threshold).astype(int)

page = st.sidebar.radio(
    "View",
    ["Live Detector", "Model Performance", "Feature Importance", "Batch Check"]
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Model: XGBoost  |  Features: {len(features)}")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Live Detector
# ════════════════════════════════════════════════════════════════════════════
if page == "Live Detector":
    st.title("🎣 PhishGuard — Live URL Detector")
    st.caption("Enter any URL to check it with the trained XGBoost model.")

    col1, col2 = st.columns([3, 1])
    with col1:
        url_input = st.text_input(
            "URL",
            placeholder="https://example.com/login",
            label_visibility="collapsed"
        )
    with col2:
        check_btn = st.button("Check URL", use_container_width=True, type="primary")

    if check_btn and url_input:
        if not url_input.startswith(("http://", "https://")):
            st.warning("URL must start with http:// or https://")
        else:
            feats = extract_lexical(url_input)
            if not feats:
                st.error("Could not extract features from this URL.")
            else:
                row = pd.DataFrame([feats])
                for f in features:
                    if f not in row.columns:
                        row[f] = -1
                row       = row[features].fillna(-1)
                row_sc    = scaler.transform(row)
                prob      = float(model.predict_proba(row_sc)[0][1])
                phishing  = prob >= threshold

                # Verdict card
                if phishing:
                    st.error(f"⚠️ **PHISHING** — {prob:.1%} confidence")
                else:
                    st.success(f"✅ **Safe** — {prob:.1%} phishing risk")

                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode  = "gauge+number",
                    value = round(prob * 100, 1),
                    title = {"text": "Phishing Risk %"},
                    gauge = {
                        "axis": {"range": [0, 100]},
                        "bar":  {"color": "#dc3545" if phishing else "#28a745"},
                        "steps": [
                            {"range": [0,  40], "color": "#d4edda"},
                            {"range": [40, 70], "color": "#fff3cd"},
                            {"range": [70, 100], "color": "#f8d7da"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 2},
                            "thickness": 0.75,
                            "value": threshold * 100
                        }
                    }
                ))
                fig.update_layout(height=260, margin=dict(t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)

                # Feature breakdown
                with st.expander("Feature values used for prediction"):
                    feat_df = pd.DataFrame(feats, index=["value"]).T
                    feat_df = feat_df[feat_df.index.isin(features)]
                    st.dataframe(feat_df, use_container_width=True)

    # Quick test URLs
    st.markdown("---")
    st.subheader("Quick test URLs")
    demos = {
        "✅ GitHub login":        "https://github.com/login",
        "✅ Google search":       "https://www.google.com/search?q=phishing",
        "✅ Amazon product":      "https://www.amazon.com/dp/B08N5WRWNW",
        "⚠️ PayPal phishing":    "http://paypal-secure-login.xyz/verify/account?id=1234",
        "⚠️ Fake Amazon":        "http://amazon-account-update.tk/login.php",
        "⚠️ IP-based login":     "http://192.168.1.1/bank/login",
        "⚠️ Brand in subdomain": "http://paypal.com.evil-site.xyz/signin",
    }
    rows = []
    for label, url in demos.items():
        feats = extract_lexical(url)
        if feats:
            row = pd.DataFrame([feats])
            for f in features:
                if f not in row.columns: row[f] = -1
            row   = row[features].fillna(-1)
            prob  = float(model.predict_proba(scaler.transform(row))[0][1])
            rows.append({
                "Label":      label,
                "URL":        url,
                "Prediction": "PHISHING" if prob >= threshold else "legit",
                "Confidence": f"{prob:.1%}"
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Model Performance
# ════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.title("📊 Model Performance")
    st.caption(f"Evaluated on {len(y_test):,} held-out test rows  |  Threshold: {threshold:.2f}")

    # Metrics row
    from sklearn.metrics import precision_score, recall_score, f1_score
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    auc = roc_auc_score(y_test, probs)
    m_col1.metric("ROC-AUC",   f"{auc:.4f}")
    m_col2.metric("Precision", f"{precision_score(y_test, preds):.4f}")
    m_col3.metric("Recall",    f"{recall_score(y_test, preds):.4f}")
    m_col4.metric("F1",        f"{f1_score(y_test, preds):.4f}")

    col1, col2 = st.columns(2)

    # Confusion matrix
    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, preds)
        fig = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicted", y="Actual",
                        color="Count"),
            x=["legit", "phishing"],
            y=["legit", "phishing"],
            color_continuous_scale="Blues"
        )
        fig.update_layout(height=350, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        tn, fp, fn, tp = cm.ravel()
        st.caption(
            f"True Positives: {tp} | False Positives: {fp} | "
            f"False Negatives: {fn} | True Negatives: {tn}"
        )

    # ROC curve
    with col2:
        st.subheader("ROC Curve")
        fpr, tpr, roc_thresh = roc_curve(y_test, probs)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"XGBoost (AUC={auc:.4f})",
            line=dict(color="#1a1a2e", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(color="gray", dash="dash"),
            showlegend=False
        ))
        fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=350,
            margin=dict(t=10, b=10),
            legend=dict(x=0.6, y=0.1)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Precision-Recall vs Threshold
    st.subheader("Precision & Recall vs Decision Threshold")
    from sklearn.metrics import precision_recall_curve
    prec_arr, rec_arr, thr_arr = precision_recall_curve(y_test, probs)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=thr_arr, y=prec_arr[:-1],
        name="Precision", line=dict(color="#007bff")
    ))
    fig.add_trace(go.Scatter(
        x=thr_arr, y=rec_arr[:-1],
        name="Recall", line=dict(color="#fd7e14")
    ))
    fig.add_vline(
        x=threshold, line_dash="dash", line_color="red",
        annotation_text=f"Current: {threshold:.2f}"
    )
    fig.update_layout(
        xaxis_title="Threshold", yaxis_title="Score",
        height=300, margin=dict(t=10, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Drag the threshold slider in the sidebar to see how metrics change.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Feature Importance
# ════════════════════════════════════════════════════════════════════════════
elif page == "Feature Importance":
    st.title("🔍 Feature Importance")

    col1, col2 = st.columns(2)

    # XGBoost built-in importance
    with col1:
        st.subheader("XGBoost feature importance")
        importance = model.feature_importances_
        imp_df = pd.DataFrame({
            "Feature":    features,
            "Importance": importance
        }).sort_values("Importance", ascending=True)

        fig = px.bar(
            imp_df, x="Importance", y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues"
        )
        fig.update_layout(
            height=500, margin=dict(t=10, b=10),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)

    # SHAP (if outputs exist)
    with col2:
        st.subheader("SHAP analysis")
        shap_path = "evaluate/outputs/shap_bar.png"
        beeswarm_path = "evaluate/outputs/shap_beeswarm.png"
        if os.path.exists(shap_path):
            st.image(shap_path, caption="Mean |SHAP| per feature")
        if os.path.exists(beeswarm_path):
            st.image(beeswarm_path, caption="SHAP beeswarm (direction of effect)")
        if not os.path.exists(shap_path):
            st.info(
                "SHAP plots not found. Run Phase 5 first:\n\n"
                "```bash\ncd evaluate && python run_phase5.py\n```"
            )

    # Feature distribution: phishing vs legit
    st.markdown("---")
    st.subheader("Feature distribution — phishing vs legit")
    feat_choice = st.selectbox("Select feature", options=features)

    feat_df = test_df[["label", feat_choice]].copy()
    feat_df["Class"] = feat_df["label"].map({0: "legit", 1: "phishing"})
    fig = px.histogram(
        feat_df, x=feat_choice, color="Class",
        barmode="overlay", opacity=0.7,
        color_discrete_map={"legit": "#28a745", "phishing": "#dc3545"},
        nbins=40
    )
    fig.update_layout(height=300, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Batch Check
# ════════════════════════════════════════════════════════════════════════════
elif page == "Batch Check":
    st.title("📋 Batch URL Checker")
    st.caption("Paste multiple URLs (one per line) or upload a .txt / .csv file.")

    tab1, tab2 = st.tabs(["Paste URLs", "Upload file"])

    def predict_urls(url_list):
        results = []
        for url in url_list:
            url = url.strip()
            if not url: continue
            feats = extract_lexical(url)
            if not feats:
                results.append({"URL": url, "Prediction": "error",
                                 "Confidence": "—", "Risk": 0})
                continue
            row = pd.DataFrame([feats])
            for f in features:
                if f not in row.columns: row[f] = -1
            row  = row[features].fillna(-1)
            prob = float(model.predict_proba(scaler.transform(row))[0][1])
            results.append({
                "URL":        url,
                "Prediction": "PHISHING" if prob >= threshold else "legit",
                "Confidence": f"{prob:.1%}",
                "Risk":       round(prob, 4)
            })
        return pd.DataFrame(results)

    with tab1:
        text_input = st.text_area(
            "URLs (one per line)",
            height=180,
            placeholder="https://github.com/login\nhttp://paypal-secure-login.xyz/verify"
        )
        if st.button("Check all", type="primary", key="batch_text"):
            urls = [u for u in text_input.strip().splitlines() if u.strip()]
            if urls:
                result_df = predict_urls(urls)
                phish_count = (result_df["Prediction"] == "PHISHING").sum()
                c1, c2, c3 = st.columns(3)
                c1.metric("Total checked", len(result_df))
                c2.metric("Phishing",      phish_count)
                c3.metric("Safe",          len(result_df) - phish_count)
                st.dataframe(
                    result_df.drop(columns=["Risk"])
                              .style.applymap(
                                  lambda v: "color: #dc3545; font-weight:600"
                                  if v == "PHISHING" else "",
                                  subset=["Prediction"]
                              ),
                    use_container_width=True, hide_index=True
                )
                csv = result_df.to_csv(index=False).encode()
                st.download_button("Download results CSV", csv,
                                   "phishguard_results.csv", "text/csv")

    with tab2:
        uploaded = st.file_uploader(
            "Upload .txt or .csv (must have a 'url' column for CSV)",
            type=["txt", "csv"]
        )
        if uploaded:
            if uploaded.name.endswith(".txt"):
                urls = uploaded.read().decode().splitlines()
            else:
                df_up = pd.read_csv(uploaded)
                if "url" not in df_up.columns:
                    st.error("CSV must have a column named 'url'")
                    urls = []
                else:
                    urls = df_up["url"].tolist()

            if urls and st.button("Check uploaded URLs", type="primary"):
                result_df = predict_urls(urls)
                phish_count = (result_df["Prediction"] == "PHISHING").sum()
                c1, c2, c3 = st.columns(3)
                c1.metric("Total checked", len(result_df))
                c2.metric("Phishing",      phish_count)
                c3.metric("Safe",          len(result_df) - phish_count)
                st.dataframe(result_df.drop(columns=["Risk"]),
                             use_container_width=True, hide_index=True)
                csv = result_df.to_csv(index=False).encode()
                st.download_button("Download results CSV", csv,
                                   "phishguard_results.csv", "text/csv")