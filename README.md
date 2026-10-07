# ChurnOps — Customer Churn Prediction MLOps Platform

> **Production-grade MLOps pipeline** for telecom customer churn prediction.  
> End-to-end: Data Ingestion → EDA → Feature Engineering → Model Training → Hyperparameter Optimization → FastAPI Deployment → Monitoring.

---

## 📊 Champion Model Performance

| Metric | Cross-Validation | Holdout Test | Registry Gate | Status |
|--------|-----------------|--------------|---------------|--------|
| **ROC-AUC** | 0.8493 | **0.8491** | ≥ 0.75 | ✅ PASS |
| **F1 Score** | 0.6371 | **0.6289** | ≥ 0.60 | ✅ PASS |
| **Recall** | 0.7866 | **0.7861** | ≥ 0.60 | ✅ PASS |
| Generalization Gap | — | **0.0002** | ≤ 0.08 | ✅ PASS |

**Algorithm:** Tuned XGBoost (max_depth=2, lr=0.08, reg_alpha=1.0, reg_lambda=20.0)  
**Training data:** IBM Telco Customer Churn — 7,043 customers, 5,634 training samples  
**Test suite:** 33/33 tests passing ✅

---

## 🏗️ Project Architecture

```
ChurnOps/
├── data/
│   ├── raw/              # Immutable raw CSV
│   ├── interim/          # Cleaned + engineered data
│   └── processed/        # train.csv, test.csv, cv_folds.json
├── src/
│   ├── data/             # ingestion, validation, EDA, cleaning, split, preprocessing
│   ├── features/         # engineering, selection
│   ├── models/           # train, cross_validate, diagnose, tune, improve, evaluate
│   │                     # track_mlflow, register, serialize
│   ├── monitoring/       # drift.py, quality.py
│   ├── pipelines/        # kubeflow_pipeline.py
│   └── deploy/           # cloud_readiness.py
├── api/
│   ├── app.py            # FastAPI inference server
│   └── schemas.py        # Pydantic request/response models
├── frontend/
│   └── app.py            # Streamlit dashboard
├── feature_repo/
│   └── feature_store_def.py  # Feast feature store definitions
├── notebooks/            # 13 executed Jupyter notebooks (one per phase)
├── models/               # Serialized model artifacts + model card
├── reports/              # All metrics, charts, and JSON reports
├── tests/                # pytest suite (33 tests)
├── scripts/              # run_pipeline.py, setup_dvc_versioning.py
├── deploy/               # K8s manifests, cloud configs
├── dvc.yaml              # 14-stage DVC pipeline
├── params.yaml           # Central configuration (all hyperparameters)
├── Dockerfile            # Multi-stage production Docker build
├── docker-compose.yml    # FastAPI + MLflow services
└── PROGRESS.md           # Phase-by-phase decision log
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Git + DVC
- Docker (for containerized deployment)

### 1. Clone & Setup

```bash
git clone https://github.com/your-org/ChurnOps.git
cd ChurnOps

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
# Option A: DVC-managed pipeline (recommended)
dvc repro                        # Reproduce all 14 DVC stages

# Option B: Python automation script
python scripts/run_pipeline.py   # Runs all 19 pipeline stages end-to-end
python scripts/run_pipeline.py --dry-run   # Preview without executing
python scripts/run_pipeline.py --list      # List all stages

# Option C: Individual stages
dvc repro ingest validate eda clean
dvc repro engineer_features select_features split preprocess
dvc repro baseline_train cv_analysis diagnose tune improve evaluate
```

### 3. Start the API Server

```bash
# Local development
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

# Docker
docker build -t churnops-api .
docker run -p 8000:8000 churnops-api

# Docker Compose (API + MLflow UI)
docker-compose up
```

**API Docs:** http://localhost:8000/docs  
**MLflow UI:** http://localhost:5000 (docker-compose)

### 4. Make a Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "DEMO-001",
    "tenure": 12,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "MonthlyCharges": 79.85,
    "TotalCharges": 958.2,
    "PaymentMethod": "Electronic check",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "PaperlessBilling": "Yes",
    "OnlineSecurity": "No",
    "TechSupport": "No",
    "gender": "Male",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No"
  }'
```

Expected response:
```json
{
  "customer_id": "DEMO-001",
  "churn_probability": 0.74,
  "churn_prediction": 1,
  "risk_label": "High Risk",
  "decision_threshold": 0.48,
  "latency_ms": 0.8,
  "model_version": "latest",
  "timestamp": "2026-10-07T..."
}
```

### 5. Launch the Streamlit Dashboard

```bash
streamlit run frontend/app.py
```

Access at: http://localhost:8501

---

## 📋 Pipeline Phases (30 Phases, 0–29)

| Phase | Name | Key Artifact |
|-------|------|-------------|
| 0 | Project Initialization | `params.yaml`, `requirements.txt` |
| 1 | Data Ingestion | `reports/dataset_metadata.json` |
| 2 | Data Validation | `reports/validation_report.json` |
| 3 | Exploratory Data Analysis | `reports/eda/*.png` |
| 4 | Data Cleaning | `data/interim/telco_clean.csv` |
| 5 | Feature Engineering | `data/interim/telco_engineered.csv` |
| 6 | Feature Selection | `reports/feature_selection_report.json` |
| 7 | Train/Test Split | `data/processed/train.csv`, `test.csv` |
| 8 | Preprocessing Pipeline | `models/preprocessor.joblib` |
| 9 | Baseline Models | `reports/baseline_summary.json` |
| 10 | Cross-Validation Deep-Dive | `reports/cv_analysis_report.json` |
| 11 | Overfitting/Underfitting Diagnosis | `reports/diagnosis_report.json` |
| 12 | Hyperparameter Optimization | `models/tuned_xgboost.joblib` |
| 13 | Iterative Improvement (Ensembles) | `models/ensemble_voting.joblib` |
| 14 | Final Model Selection + Test Eval | `models/champion_model.joblib` |
| 15 | MLflow Tracking | `reports/mlflow_runs_summary.json` |
| 16 | DVC Dataset Versioning | `reports/dvc_versioning_manifest.json` |
| 17 | Pipeline Automation | `scripts/run_pipeline.py` |
| 18 | Kubeflow Pipeline Design | `src/pipelines/kubeflow_pipeline.py` |
| 19 | Feast Feature Store | `feature_repo/feature_store_def.py` |
| 20 | Model Registration | `reports/model_registry_manifest.json` |
| 21 | Model Serialization | `models/model_card.json`, versioned joblib |
| 22 | FastAPI Inference Server | `api/app.py`, `/predict` endpoint |
| 23 | Streamlit User Interface | `frontend/app.py` |
| 24 | Monitoring (Drift + Quality) | `reports/drift_report.json` |
| 25 | Cloud Deployment Readiness | `reports/cloud_readiness_report.json` |
| 26 | Testing | `tests/test_pipeline.py` — 33/33 ✅ |
| 27 | Docker | `Dockerfile`, `docker-compose.yml` |
| 28 | Final README | `README.md` |
| 29 | Demo Guide | `DEMO_GUIDE.md` |

---

## 🧪 Running Tests

```bash
# Full test suite
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=src --cov=api --cov-report=html

# Specific test class
python -m pytest tests/test_pipeline.py::TestRegistryGates -v
```

**Test Coverage:**
- `TestDataIntegrity` (9 tests): Schema, types, null checks, stratification
- `TestPreprocessingPipeline` (6 tests): Prediction correctness, latency, unknown categories
- `TestRegistryGates` (6 tests): F1, AUC, Recall, Gap — all champion gates verified
- `TestAPIContract` (7 tests): `/health`, `/model/info`, `/predict`, `/predict/batch`

---

## 🔭 MLflow Experiment Tracking

```bash
# View MLflow UI
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

# Access at http://localhost:5000
```

**Tracked runs:** 10 (4 baselines, 3 tuned, 2 ensembles, 1 champion)  
**Champion run:** `Champion_Tuned_XGBoost` — tagged `is_champion=true`  
**Registered model:** `ChurnOps_Champion` v2 (alias: `champion`)

---

## 🔍 Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| XGBoost as champion | Highest val F1 (0.6371), lowest gap (0.019), fastest inference (<1ms/sample) |
| `max_depth=2` | Phase 11 showed depths ≥4 cause severe overfitting; depth 2-3 is optimal |
| Threshold = 0.48 | Optimal F1 vs Recall trade-off for imbalanced classes |
| `class_weight` via `scale_pos_weight` | No leakage; computed per training fold |
| `gender` dropped | Zero mutual information (0.000); enforces algorithmic fairness |
| `PhoneService` dropped | 100% redundant with `MultipleLines` |
| DomainFeatureEngineer in pipeline | Prevents train-serve skew for real-time inference |

---

## 📁 Key Metrics Files

| File | Contents |
|------|----------|
| `reports/final_model_evaluation.json` | Holdout test metrics (AUC, F1, Recall, Precision) |
| `reports/tuning_report.json` | HPO results across 3 models |
| `reports/model_registry_manifest.json` | Registry gate audit |
| `models/model_card.json` | Full model card (schema, performance, ethics) |
| `models/inference_signature.json` | Input/output contract for API |

---

## 🐳 Docker

```bash
# Build
docker build -t churnops-api:1.0.0 .

# Run API only
docker run -p 8000:8000 -v $(pwd)/models:/app/models churnops-api:1.0.0

# Full stack (API + MLflow)
docker-compose up --build

# Health check
curl http://localhost:8000/health
```

---

## ☁️ Cloud Deployment

| Platform | Command |
|----------|---------|
| **Vertex AI** | `gcloud ai endpoints deploy-model ...` (see `reports/cloud_readiness_report.json`) |
| **AWS SageMaker** | `python deploy/sagemaker_deploy.py` |
| **Kubernetes** | `kubectl apply -f deploy/k8s_deployment.yaml -f deploy/k8s_service.yaml` |

---

## 📦 Tech Stack

| Layer | Technology |
|-------|------------|
| **ML** | Scikit-Learn 1.4, XGBoost 2.x, SHAP |
| **Tracking** | MLflow 2.14 (SQLite backend) |
| **Versioning** | DVC 3.x (local/S3/GCS remote) |
| **Feature Store** | Feast 0.38 (SQLite online store) |
| **Pipelines** | KFP v2 SDK (Kubeflow / Vertex AI) |
| **API** | FastAPI + Uvicorn |
| **UI** | Streamlit |
| **Testing** | pytest 8.x |
| **Containerization** | Docker (multi-stage) + docker-compose |
| **Language** | Python 3.11 |

---

## 📄 License

MIT License — see `LICENSE` for details.
