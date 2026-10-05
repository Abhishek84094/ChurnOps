# ChurnOps — PROGRESS LOG
Track phase status, key decisions, open questions, and findings.
Updated after every phase.

---

## Phase Status

| Phase | Name                          | Status      | Date       |
|-------|-------------------------------|-------------|------------|
| 0     | Project Initialization        | ✅ Complete | 2026-10-05 |
| 1     | Data Ingestion                | ⏳ Pending  |            |
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
