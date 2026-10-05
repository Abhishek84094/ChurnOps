# ChurnOps — PROGRESS LOG
Track phase status, key decisions, open questions, and findings.
Updated after every phase.

---

## Phase Status

| Phase | Name                          | Status      | Date       |
|-------|-------------------------------|-------------|------------|
| 0     | Project Initialization        | ✅ Complete | 2026-10-05 |
| 1     | Data Ingestion                | ✅ Complete | 2026-10-05 |
| 2     | Data Validation               | ⏳ Pending  |            |
| 3     | Exploratory Data Analysis     | ⏳ Pending  |            |
| 4     | Data Cleaning                 | ⏳ Pending  |            |
| 5     | Feature Engineering           | ⏳ Pending  |            |
| 6     | Feature Selection             | ⏳ Pending  |            |
| 7     | Train/Validation/Test Split   | ⏳ Pending  |            |
| 8     | Preprocessing Pipeline        | ⏳ Pending  |            |
| 9     | Baseline Models               | ⏳ Pending  |            |
| 10    | Cross-Validation              | ⏳ Pending  |            |
| 11    | Overfitting/Underfitting Diag | ⏳ Pending  |            |
| 12    | Hyperparameter Optimization   | ⏳ Pending  |            |
| 13    | Iterative Improvement         | ⏳ Pending  |            |
| 14    | Final Model Selection         | ⏳ Pending  |            |
| 15    | MLflow Tracking               | ⏳ Pending  |            |
| 16    | DVC Dataset Versioning        | ⏳ Pending  |            |
| 17    | Pipeline Automation           | ⏳ Pending  |            |
| 18    | Kubeflow Pipeline Design      | ⏳ Pending  |            |
| 19    | Feast Feature Store           | ⏳ Pending  |            |
| 20    | Model Registration            | ⏳ Pending  |            |
| 21    | Model Serialization           | ⏳ Pending  |            |
| 22    | FastAPI                       | ⏳ Pending  |            |
| 23    | User Interface                | ⏳ Pending  |            |
| 24    | Monitoring                    | ⏳ Pending  |            |
| 25    | Cloud Deployment Readiness    | ⏳ Pending  |            |
| 26    | Testing                       | ⏳ Pending  |            |
| 27    | Docker                        | ⏳ Pending  |            |
| 28    | Final README                  | ⏳ Pending  |            |
| 29    | Demo Guide                    | ⏳ Pending  |            |

---

## Phase 0 — Project Initialization

**Status:** ✅ Complete  
**Date:** 2026-10-05

### What was done
- Created full directory structure (`data/`, `src/`, `notebooks/`, etc.)
- Moved raw dataset to `data/raw/` (immutable from this point)
- Created `params.yaml`, `requirements.txt`, `.gitignore`, `PROGRESS.md`
- Created all `__init__.py` stubs and placeholder source files
- Created `src/utils/config.py` (YAML loader, path helpers)
- Initialized git repository; made initial commit
- Created and activated Python virtual environment `.venv`
- Installed all pinned dependencies

### Key Decisions
| Decision | Reasoning |
|----------|-----------|
| Python 3.11 | Best compat with kfp 2.x, feast 0.38, mlflow 2.14, xgboost 2.x |
| `random_state: 42` in params.yaml | All randomness controlled centrally |
| Test size 0.20, Stratified | 80% train keeps ample data; stratify preserves class ratio |
| CV folds: 5 | Standard; balanced bias-variance for ~5.6k training rows |
| Imbalance strategy: class_weight (default) | No leakage risk; evaluated in Phase 9 |
| Primary tuning metric: F1 | Balances precision/recall; better than accuracy for imbalanced |
| `feast==0.38.1` | Latest stable with local SQLite online store support |
| `kfp==2.7.0` | KFP v2 SDK; compatible with Vertex AI Pipelines |

### Open Questions (resolved in future phases)
- [ ] Exact class distribution → Phase 1/2
- [ ] Whether TotalCharges has blank strings → Phase 2
- [ ] Whether the extended IBM version is used (leakage columns) → Phase 2
- [ ] Final primary metric weight (recall vs F1) → Phase 14

### Honesty Status
- `Implemented locally`: environment setup, directory structure, config
- Not yet: any model training, MLflow runs, DVC tracking

---

## Phase 1 — Data Ingestion

**Status:** Complete  
**Date:** 2026-10-05

### What was done
- Implemented `src/data/ingestion.py` with integrity checks, SHA-256 hashing, and full summary statistics.
- Ran ingestion on the real raw CSV and verified all numbers from code output (R1).
- Produced `reports/dataset_metadata.json` with real numbers.
- Added `ingest` stage to `dvc.yaml`.

### Real numbers (verified from code execution, not fabricated)
| Metric                  | Value                              |
|-------------------------|-------------------------------------|
| File                    | WA_Fn-UseC_-Telco-Customer-Churn.csv |
| File size               | 954.59 KB (977,501 bytes)           |
| SHA-256 (first 16 hex)  | 88be4b93fbe0cc83...                 |
| Shape                   | 7,043 rows x 21 columns             |
| Duplicate rows          | 0                                   |
| Duplicate customerIDs   | 0                                   |
| Churn=No                | 5,174 (73.46%)                      |
| Churn=Yes               | 1,869 (26.54%)                      |
| Missing values (NaN)    | 0 columns                           |
| TotalCharges blanks     | 11 rows (blank string, not NaN)     |

### Columns (21 total)
customerID, gender, SeniorCitizen, Partner, Dependents, tenure,
PhoneService, MultipleLines, InternetService, OnlineSecurity,
OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
MonthlyCharges, TotalCharges, Churn

### Key findings
- Confirmed standard IBM Telco (NOT extended version): no leakage columns detected.
- TotalCharges: stored as object dtype, 11 blank strings (likely where tenure=0). Will investigate in Phase 2 and fix in Phase 4.
- SeniorCitizen: stored as "0"/"1" strings (all others are "Yes"/"No"). Noted for cleaning.
- Class imbalance: ~73.5% No vs ~26.5% Yes — moderate imbalance. Accuracy will be misleading; must use Recall/F1/PR-AUC.
- No NaN missing values detected at pandas level; the "missing" TotalCharges data is disguised as blank strings.

### Honesty Status
- Implemented locally: All numbers from actual code execution.
