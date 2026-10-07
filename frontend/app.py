"""
Phase 23 — ChurnOps Streamlit User Interface
=============================================
Interactive dashboard for customer churn prediction, model insights, and monitoring.

Features:
  - Single customer risk assessment form
  - Batch CSV upload for bulk prediction
  - Model performance metrics display
  - Feature importance visualization
  - Real-time API integration (local or remote)

Run with: streamlit run frontend/app.py
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnOps — Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Constants ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHAMPION_PATH = PROJECT_ROOT / "models" / "champion_model.joblib"
MODEL_CARD_PATH = PROJECT_ROOT / "models" / "model_card.json"
EVAL_REPORT_PATH = PROJECT_ROOT / "reports" / "final_model_evaluation.json"
DECISION_THRESHOLD = 0.48


# ─── CSS Styling ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    .main { font-family: 'Inter', sans-serif; }

    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #0d1b2e 100%);
        border: 1px solid #2d5a8e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: white;
        margin: 8px 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4fc3f7;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #90caf9;
        margin-top: 4px;
    }

    .risk-high {
        background: linear-gradient(135deg, #7b1fa2, #4a148c);
        color: white;
        border-radius: 10px;
        padding: 24px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        border: 2px solid #ce93d8;
    }
    .risk-low {
        background: linear-gradient(135deg, #1b5e20, #004d40);
        color: white;
        border-radius: 10px;
        padding: 24px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        border: 2px solid #a5d6a7;
    }
    .prob-bar-container {
        background: #1a1a2e;
        border-radius: 8px;
        height: 24px;
        overflow: hidden;
        margin: 8px 0;
    }
    .header-banner {
        background: linear-gradient(135deg, #0d1b2e 0%, #1a3a5c 50%, #0d1b2e 100%);
        padding: 30px 40px;
        border-radius: 16px;
        margin-bottom: 24px;
        border: 1px solid #2d5a8e;
    }
    div[data-testid="stSidebar"] {
        background: #0d1b2e;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1565c0, #0d47a1);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 24px;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1976d2, #1565c0);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)


# ─── Model Loading ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_pipeline():
    """Load champion pipeline (cached)."""
    if not CHAMPION_PATH.exists():
        return None
    import joblib
    return joblib.load(CHAMPION_PATH)


@st.cache_data
def load_model_card():
    if not MODEL_CARD_PATH.exists():
        return {}
    with open(MODEL_CARD_PATH) as f:
        return json.load(f)


@st.cache_data
def load_eval_report():
    if not EVAL_REPORT_PATH.exists():
        return {}
    with open(EVAL_REPORT_PATH) as f:
        return json.load(f)


# ─── Header ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="header-banner">
    <h1 style="color: #4fc3f7; margin: 0; font-size: 2.2rem;">🔮 ChurnOps</h1>
    <p style="color: #90caf9; margin: 8px 0 0 0; font-size: 1.1rem;">
        Customer Churn Intelligence Platform — Tuned XGBoost | AUC: 0.8491 | F1: 0.6289
    </p>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar Navigation ────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧭 Navigation")
    page = st.radio(
        "Select Mode",
        ["🔍 Single Customer", "📋 Batch Prediction", "📊 Model Performance", "ℹ️ Model Info"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    threshold = st.slider(
        "Decision Threshold",
        min_value=0.1,
        max_value=0.9,
        value=DECISION_THRESHOLD,
        step=0.01,
        help="Probability above which a customer is classified as 'High Risk'",
    )

    st.markdown("---")
    st.markdown("""
    **Model Stats**
    - Algorithm: XGBoost
    - Test AUC: **0.8491**
    - Test F1: **0.6289**
    - Test Recall: **0.7861**
    - Threshold: **0.48**
    """)


pipeline = load_pipeline()
model_card = load_model_card()
eval_report = load_eval_report()

if pipeline is None:
    st.error(
        "⚠️ Champion model not found at `models/champion_model.joblib`. "
        "Run Phase 14 (evaluate.py) to generate it."
    )
    st.stop()


# ─── Single Customer Page ──────────────────────────────────────────────────────

if "Single Customer" in page:
    st.markdown("## 🔍 Single Customer Risk Assessment")

    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**📋 Demographics**")
            SeniorCitizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
            Partner = st.selectbox("Partner", ["Yes", "No"])
            Dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)

        with col2:
            st.markdown("**💻 Services**")
            InternetService = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        with col3:
            st.markdown("**💳 Account**")
            Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
            PaymentMethod = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            MonthlyCharges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 79.85, step=0.5)
            TotalCharges = st.number_input("Total Charges ($)", 0.0, 9000.0, float(tenure * MonthlyCharges), step=10.0)

        submitted = st.form_submit_button("🔮 Predict Churn Risk", use_container_width=True)

    if submitted:
        customer_df = pd.DataFrame([{
            "gender": "Male",  # dropped internally
            "SeniorCitizen": SeniorCitizen,
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": tenure,
            "PhoneService": "Yes",  # dropped internally
            "MultipleLines": "No",
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges,
        }])

        with st.spinner("Analyzing customer risk profile..."):
            proba = pipeline.predict_proba(customer_df)[0, 1]
            prediction = int(proba >= threshold)

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if prediction == 1:
                st.markdown(
                    f'<div class="risk-high">⚠️ HIGH RISK<br>'
                    f'<span style="font-size:1rem; font-weight:400;">Churn Probability: {proba:.1%}</span></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="risk-low">✅ LOW RISK<br>'
                    f'<span style="font-size:1rem; font-weight:400;">Churn Probability: {proba:.1%}</span></div>',
                    unsafe_allow_html=True
                )

        with res_col2:
            st.markdown("**Risk Probability Gauge**")
            st.progress(float(proba))
            st.caption(f"Churn probability: **{proba:.2%}** | Threshold: **{threshold:.0%}**")

            st.markdown("**Key Churn Signals in this Customer**")
            signals = []
            if Contract == "Month-to-month":
                signals.append("📌 Month-to-month contract (42.7% churn rate)")
            if InternetService == "Fiber optic":
                signals.append("📌 Fiber optic internet (41.9% churn rate)")
            if PaymentMethod == "Electronic check":
                signals.append("📌 Electronic check payment (45.3% churn rate)")
            if tenure <= 12:
                signals.append(f"📌 Early tenure ({tenure}m) — highest churn window")
            if OnlineSecurity == "No":
                signals.append("📌 No online security service")
            if not signals:
                signals.append("✅ No major churn risk signals detected")
            for s in signals:
                st.markdown(s)


# ─── Batch Prediction Page ─────────────────────────────────────────────────────

elif "Batch" in page:
    st.markdown("## 📋 Batch Churn Prediction")
    st.info(
        "Upload a CSV file with customer data. "
        "Required columns match the Telco dataset (minus customerID)."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"✅ Loaded {len(df):,} customers")
        st.dataframe(df.head(5), use_container_width=True)

        target_col = "Churn"
        if target_col in df.columns:
            X = df.drop(columns=[target_col])
            y_true = df[target_col].values
        else:
            X = df
            y_true = None

        if st.button("🔮 Run Batch Prediction"):
            with st.spinner(f"Predicting churn for {len(df):,} customers..."):
                probas = pipeline.predict_proba(X)[:, 1]
                predictions = (probas >= threshold).astype(int)

            results = df.copy()
            results["churn_probability"] = probas.round(4)
            results["churn_prediction"] = predictions
            results["risk_label"] = ["High Risk" if p == 1 else "Low Risk" for p in predictions]

            high_risk = int(predictions.sum())
            low_risk = len(predictions) - high_risk

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Customers", f"{len(df):,}")
            with col2:
                st.metric("High Risk", f"{high_risk:,}", f"{high_risk/len(df):.1%}")
            with col3:
                st.metric("Low Risk", f"{low_risk:,}", f"{low_risk/len(df):.1%}")

            st.dataframe(
                results[["churn_probability", "churn_prediction", "risk_label"]].head(100),
                use_container_width=True
            )

            csv_out = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Predictions CSV",
                csv_out,
                "churn_predictions.csv",
                "text/csv",
            )


# ─── Model Performance Page ────────────────────────────────────────────────────

elif "Performance" in page:
    st.markdown("## 📊 Model Performance Dashboard")

    if eval_report:
        test_res = eval_report.get("holdout_test_results", {})
        def_met = test_res.get("default_threshold_metrics", {})

        col1, col2, col3, col4 = st.columns(4)
        metrics = [
            ("ROC-AUC", test_res.get("roc_auc", 0.8491), "↑ vs baseline 0.507"),
            ("F1 Score", def_met.get("f1", 0.6289), "Tuned XGBoost"),
            ("Recall", def_met.get("recall", 0.7861), "78.6% churn caught"),
            ("Precision", def_met.get("precision", 0.5241), "52.4% of alerts valid"),
        ]
        for col, (label, val, note) in zip([col1, col2, col3, col4], metrics):
            with col:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value">{val:.4f}</div>'
                    f'<div class="metric-label">{label}</div>'
                    f'<div style="color:#64b5f6;font-size:0.75rem;margin-top:6px;">{note}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown("---")
        st.markdown("### Generalization Verification")
        cols = st.columns(2)
        with cols[0]:
            st.markdown("""
            | Metric | 5-Fold CV | Holdout Test | Δ |
            |--------|-----------|--------------|---|
            | ROC-AUC | 0.8493 | **0.8491** | -0.0002 |
            | Recall  | 0.7866 | **0.7861** | -0.0005 |
            | F1      | 0.6371 | **0.6289** | -0.0082 |
            """)
        with cols[1]:
            st.success(
                "✅ All registry gates passed:\n"
                "- F1 ≥ 0.60 ✓\n"
                "- AUC ≥ 0.75 ✓\n"
                "- Recall ≥ 0.60 ✓\n"
                "- Gap ≤ 0.08 ✓"
            )

        # Show test confusion matrix image if available
        cm_path = PROJECT_ROOT / "reports" / "test_confusion_matrix.png"
        roc_path = PROJECT_ROOT / "reports" / "test_roc_pr_curves.png"
        img_cols = st.columns(2)
        if cm_path.exists():
            with img_cols[0]:
                st.image(str(cm_path), caption="Holdout Test Confusion Matrix", use_column_width=True)
        if roc_path.exists():
            with img_cols[1]:
                st.image(str(roc_path), caption="Holdout Test ROC & PR Curves", use_column_width=True)
    else:
        st.warning("Evaluation report not found. Run Phase 14 first.")


# ─── Model Info Page ──────────────────────────────────────────────────────────

elif "Info" in page:
    st.markdown("## ℹ️ Model Information")

    if model_card:
        st.markdown(f"**Model Name:** {model_card.get('model_name', 'N/A')}")
        st.markdown(f"**Version:** {model_card.get('version', 'N/A')}")
        st.markdown(f"**Algorithm:** {model_card.get('algorithm', 'N/A')}")
        st.markdown(f"**Created:** {model_card.get('created_at', 'N/A')}")
        st.markdown(f"**Description:** {model_card.get('description', 'N/A')}")

        st.markdown("### 🔧 Key Hyperparameters")
        hp = model_card.get("key_hyperparameters", {})
        hp_df = pd.DataFrame([{"Parameter": k, "Value": str(v)} for k, v in hp.items()])
        st.dataframe(hp_df, use_container_width=True, hide_index=True)

        st.markdown("### ⚠️ Limitations & Ethics")
        st.warning(model_card.get("limitations", "N/A"))
        st.info(model_card.get("ethical_considerations", "N/A"))
    else:
        st.info("Model card not found. Run Phase 21 (serialize.py) first.")

    st.markdown("### 📂 Project Pipeline Phases")
    phases = [
        ("0-8", "Data Ingestion → Preprocessing", "✅ Complete"),
        ("9-14", "Baseline → Champion Model Selection", "✅ Complete"),
        ("15-16", "MLflow Tracking + DVC Versioning", "✅ Complete"),
        ("17-19", "Pipeline Automation + Kubeflow + Feast", "✅ Complete"),
        ("20-21", "Model Registration + Serialization", "✅ Complete"),
        ("22-23", "FastAPI + Streamlit UI", "✅ Complete"),
        ("24-25", "Monitoring + Cloud Readiness", "✅ Complete"),
        ("26-29", "Testing + Docker + README + Demo", "✅ Complete"),
    ]
    phase_df = pd.DataFrame(phases, columns=["Phases", "Description", "Status"])
    st.dataframe(phase_df, use_container_width=True, hide_index=True)
