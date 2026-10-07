# ChurnOps — Demo Guide

> Step-by-step walkthrough to showcase the full ChurnOps MLOps pipeline to stakeholders,  
> interviewers, or reviewers. This guide runs in ~15 minutes end-to-end.

---

## 🎯 Demo Overview

ChurnOps demonstrates a **production-grade MLOps pipeline** that predicts telecom customer churn.  
The demo covers: data pipeline → model training → REST API → UI dashboard → monitoring.

| Milestone | Time |
|-----------|------|
| Environment setup | ~2 min |
| Run trained pipeline demo | ~3 min |
| API prediction demo | ~3 min |
| Streamlit dashboard | ~3 min |
| MLflow tracking UI | ~2 min |
| Monitoring demo | ~2 min |

---

## ✅ Prerequisites

```bash
cd C:\Users\Abhishek\OneDrive\Pictures\Desktop\ChurnOps
.venv\Scripts\activate
```

Verify environment:
```bash
python -c "import mlflow, xgboost, fastapi, streamlit; print('All deps OK')"
```

---

## 📍 Step 1: Show Pipeline Status

Display the project structure and phase completion status:

```bash
# Print all 30 pipeline phases
python -c "
from pathlib import Path
print(open('PROGRESS.md').read()[:3000])
"
```

Show the DVC pipeline DAG:
```bash
dvc dag
```

Expected DAG output shows 14 connected stages from `ingest` → `evaluate`.

---

## 📍 Step 2: Show Trained Champion Model

Verify the champion model and its metrics:

```bash
python -c "
import json
with open('reports/final_model_evaluation.json') as f:
    r = json.load(f)
print('Champion:', r['champion_model'])
test = r['holdout_test_results']
dm = test['default_threshold_metrics']
print(f'Test ROC-AUC: {test[\"roc_auc\"]:.4f}')
print(f'Test F1:      {dm[\"f1\"]:.4f}')
print(f'Test Recall:  {dm[\"recall\"]:.4f}')
print(f'Threshold:    {r[\"optimal_threshold\"]}')
print('All registry gates:', r['registry_gate_summary'])
"
```

---

## 📍 Step 3: Live API Prediction

### Start the API server:
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

### In a second terminal, run predictions:

**Single High-Risk Customer (Month-to-Month + Fiber + Electronic Check):**
```bash
curl -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"customer_id\":\"HIGH-RISK-001\",\"tenure\":3,\"Contract\":\"Month-to-month\",\"InternetService\":\"Fiber optic\",\"MonthlyCharges\":89.5,\"TotalCharges\":268.5,\"PaymentMethod\":\"Electronic check\",\"SeniorCitizen\":0,\"Partner\":\"No\",\"Dependents\":\"No\",\"PaperlessBilling\":\"Yes\",\"OnlineSecurity\":\"No\",\"TechSupport\":\"No\",\"gender\":\"Male\",\"PhoneService\":\"Yes\",\"MultipleLines\":\"Yes\",\"OnlineBackup\":\"No\",\"DeviceProtection\":\"No\",\"StreamingTV\":\"Yes\",\"StreamingMovies\":\"Yes\"}"
```

Expected: `"risk_label": "High Risk"`, probability > 0.70

**Single Low-Risk Customer (2-Year Contract + Bank Transfer):**
```bash
curl -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"customer_id\":\"LOW-RISK-001\",\"tenure\":60,\"Contract\":\"Two year\",\"InternetService\":\"DSL\",\"MonthlyCharges\":55.0,\"TotalCharges\":3300.0,\"PaymentMethod\":\"Bank transfer (automatic)\",\"SeniorCitizen\":0,\"Partner\":\"Yes\",\"Dependents\":\"Yes\",\"PaperlessBilling\":\"No\",\"OnlineSecurity\":\"Yes\",\"TechSupport\":\"Yes\",\"gender\":\"Female\",\"PhoneService\":\"Yes\",\"MultipleLines\":\"No\",\"OnlineBackup\":\"Yes\",\"DeviceProtection\":\"Yes\",\"StreamingTV\":\"No\",\"StreamingMovies\":\"No\"}"
```

Expected: `"risk_label": "Low Risk"`, probability < 0.20

**Health + Model Info:**
```bash
curl http://localhost:8000/health
curl http://localhost:8000/model/info
```

**Interactive API Docs:**  
Open: http://localhost:8000/docs

---

## 📍 Step 4: Streamlit Dashboard

```bash
# Stop the API server (or use a new terminal)
streamlit run frontend/app.py
```

Open: http://localhost:8501

**Demo flow:**
1. **Single Customer tab** → Fill in: tenure=6, Contract=Month-to-month, InternetService=Fiber optic, MonthlyCharges=85 → Click "Predict Churn Risk"
2. **Batch Prediction tab** → Upload `data/processed/test.csv` → Run predictions → Download CSV
3. **Model Performance tab** → Show confusion matrix and ROC curve
4. **Model Info tab** → Show hyperparameters, ethics, limitations

---

## 📍 Step 5: MLflow Tracking UI

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
```

Open: http://localhost:5000

**What to show:**
1. **Experiments** → `ChurnOps_Model_Comparison` → 10 runs
2. **Runs** → Filter by tag `is_champion = true` → Shows `Champion_Tuned_XGBoost`
3. **Compare** → Select Baseline vs Tuned XGBoost → show F1 improvement (0.578 → 0.637)
4. **Model Registry** → `ChurnOps_Champion` → v2 with alias `champion`
5. **Artifacts** → Confusion matrix, ROC curve attached to champion run

---

## 📍 Step 6: Show Test Suite

```bash
python -m pytest tests/test_pipeline.py -v --tb=short
```

Expected: **33 passed** in ~3 seconds.

Key tests to highlight:
- `TestRegistryGates` — all production quality gates verified automatically
- `TestAPIContract` — full FastAPI endpoint contract tested
- `TestPreprocessingPipeline::test_inference_latency` — < 100ms verified

---

## 📍 Step 7: Monitoring Demo

**Drift monitoring:**
```bash
python src/monitoring/drift.py
```
Shows PSI analysis across 9 features — all **STABLE** between train/test distributions.

**Quality monitoring:**
```bash
python src/monitoring/quality.py
```
Shows AUC=0.8491, F1=0.6267, Brier=0.1572 — within acceptable bounds.

---

## 📍 Step 8: Docker (Optional)

```bash
# Build image
docker build -t churnops-api:1.0.0 .

# Run
docker run -p 8000:8000 -v "%cd%\models:/app/models" churnops-api:1.0.0

# Test health
curl http://localhost:8000/health
```

---

## 📍 Step 9: DVC Pipeline Demo

```bash
# Show current pipeline state
dvc status

# Show what dvc repro would do
dvc repro --dry

# Show pipeline DAG
dvc dag --md
```

---

## 🗝️ Key Talking Points

### 1. No Data Leakage
- Preprocessing parameters (mean, std, OHE vocab) fitted **only on train.csv**
- `test.csv` touched exactly **once** in Phase 14
- Feature engineering inside Scikit-Learn pipeline prevents train-serve skew

### 2. Overfitting Eradicated
- RandomForest baseline: gap = **+0.4494** (severe overfitting)
- After tuning: gap = **+0.0576** (within 0.08 gate)
- XGBoost baseline gap: **+0.3788** → Tuned: **+0.0194**

### 3. Registry Gate System
- 4 automated quality gates enforced by code (F1, AUC, Recall, Gap)
- Champion only promoted when all 4 pass on unseen test data
- Logged to MLflow + registry manifest JSON

### 4. Production Readiness
- FastAPI with health check, batch prediction, model info endpoints
- Multi-stage Dockerfile (< 300MB image)
- Drift + quality monitoring for post-deployment surveillance
- Cloud deployment configs for Vertex AI, SageMaker, Kubernetes

### 5. Reproducibility
- Every random seed controlled via `params.yaml`
- All pipeline stages versioned in DVC
- 5-fold CV indices stored in `cv_folds.json` (same folds used across all phases)

---

## 📊 Key Numbers to Remember

| Metric | Value |
|--------|-------|
| Training samples | 5,634 |
| Test samples | 1,409 |
| Features (raw) | 19 |
| Features (after engineering) | 26 |
| CV folds | 5 (stratified) |
| Baseline best F1 (LR) | 0.6258 |
| Tuned XGBoost F1 | 0.6371 |
| Holdout Test AUC | **0.8491** |
| Inference latency | < 1ms/sample |
| Test suite | 33/33 ✅ |
| MLflow runs | 10 |
| Pipeline stages (DVC) | 14 |
| Total phases | 30 (0–29) |
