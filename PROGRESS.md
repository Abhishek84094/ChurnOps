# ChurnOps — PROGRESS LOG
Track phase status, key decisions, open questions, and findings.
Updated after every phase.

---

## Phase Status

| Phase | Name                          | Status      | Date       |
|-------|-------------------------------|-------------|------------|
| 0     | Project Initialization        | ✅ Complete | 2026-10-05 |
| 1     | Data Ingestion                | ✅ Complete | 2026-10-05 |
| 2     | Data Validation               | Complete | 2026-10-05 |
| 3     | Exploratory Data Analysis     | ✅ Complete | 2026-10-05 |
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

## Phase 2 - Data Validation

**Status:** Complete
**Date:** 2026-10-05

### Validation Result: PASSED (0 critical errors, 1 warning)

### Real numbers from validation_report.json
| Check                        | Result                                      |
|------------------------------|---------------------------------------------|
| Schema (21 expected cols)    | PASS - all present, 0 missing, 0 extra      |
| Leakage columns              | PASS - none detected                        |
| Duplicate rows               | PASS - 0                                    |
| Duplicate customerIDs        | PASS - 0                                    |
| Target Churn classes         | PASS - ["No", "Yes"] only                   |
| Target null values           | PASS - 0                                    |
| Imbalance ratio (No:Yes)     | 2.7683                                      |
| tenure range [0, 72]         | PASS - mean=32.37, std=24.56                |
| MonthlyCharges range [18.25, 118.75] | PASS - mean=64.76, std=30.09      |
| TotalCharges range [18.8, 8684.8] | PASS (11 blank strings, WARN)          |
| SeniorCitizen range [0, 1]   | PASS - mean=0.1621 (16.2% senior)           |
| All categorical columns      | PASS - all values within expected sets      |
| All binary Yes/No columns    | PASS                                        |

### Only Warning (non-critical)
- TotalCharges: 11 rows are blank strings (not NaN). These correspond to tenure=0
  customers with no charges yet. Will be addressed in Phase 4 (deterministic fix).
  Detected here, NOT silently fixed.

### Honesty Status
- Implemented locally: all findings from real code execution.

---

## Phase 3 — Exploratory Data Analysis

**Status:** ✅ Complete  
**Date:** 2026-10-05

### What was done
- Implemented `src/data/eda.py` module computing univariate statistics, bivariate associations, Spearman rank correlation, and bias-corrected **Cramér's V** for categorical features.
- Generated publication-quality figures saved into `reports/eda/`:
  - `01_target_distribution.png`: Count barplot + Imbalance donut chart.
  - `02_numeric_distributions.png`: Histograms with KDE + Churn-stratified boxplots for `tenure`, `MonthlyCharges`, `TotalCharges`.
  - `03_cramers_v_ranking.png`: Ranking of all categorical features by Cramér's V association with Churn.
  - `04_bivariate_key_drivers.png`: Churn rate breakdowns for Contract, InternetService, PaymentMethod, and SeniorCitizen.
  - `05_numeric_correlation_matrix.png`: Pearson and Spearman correlation heatmaps.
- Created and executed `notebooks/02_eda.ipynb` with interactive cell-by-cell walkthrough, inline figures, code commentary, and domain takeaways.
- Produced `reports/eda/eda_summary.json` containing complete quantitative metadata.
- Added `eda` stage to `dvc.yaml`.

### Real Numbers from EDA Execution
| Feature / Metric | Value |
|------------------|-------|
| Target Count (No) | 5,174 (73.46%) |
| Target Count (Yes) | 1,869 (26.54%) |
| Class Imbalance Ratio | 2.768 : 1 (No : Yes) |
| Top 1 Categorical Driver | `Contract` (Cramér's V = 0.4098) |
| Top 2 Categorical Driver | `OnlineSecurity` (Cramér's V = 0.3470) |
| Top 3 Categorical Driver | `TechSupport` (Cramér's V = 0.3425) |
| Top 4 Categorical Driver | `InternetService` (Cramér's V = 0.3220) |
| Top 5 Categorical Driver | `PaymentMethod` (Cramér's V = 0.3027) |
| Tenure Churners vs Non-Churners | Churn heavily concentrated in months 0-12 (median churn tenure ~10 mos vs ~38 mos retained) |
| Month-to-Month Contract Churn Rate | 42.71% (vs 11.27% 1-yr, 2.83% 2-yr) |
| Fiber Optic Internet Churn Rate | 41.89% (vs 18.96% DSL, 7.40% No internet) |
| Electronic Check Churn Rate | 45.29% (vs 15.24% Credit card auto, 16.71% Bank transfer auto) |
| TotalCharges Blanks Root Cause | Exactly 11 rows, 100% have `tenure = 0` (brand new signups, 0 full billing cycles) |

### Key Decisions & Downstream Directives
1. **Handling Class Imbalance:** Naive accuracy yields 73.46% with 0% churn recall. Downstream phases will enforce:
   - Stratified train/test splits (Phase 7).
   - Cost-sensitive weighting (`class_weight='balanced'`, `scale_pos_weight=2.77`) in Phase 9.
   - Primary optimization on Recall, F1-Score, and PR-AUC (Phase 10, 12, 14).
2. **Outlier Assessment:** IQR and range checks on `tenure` (0-72m), `MonthlyCharges` ($18.25-$118.75), and `TotalCharges` ($18.80-$8684.80) confirm all values represent valid telecommunication customers. Zero records will be deleted as "outliers" (retaining natural domain variation).
3. **Data Cleaning Plan (Phase 4):**
   - Deterministic imputation of `0.0` for `TotalCharges` where `tenure = 0`, then cast to `float64`.
   - Drop `customerID` (unique identifier with no generalizable signal).
   - Map `Churn` to integer (Yes -> 1, No -> 0).
4. **Feature Engineering Candidates (Phase 5):**
   - `tenure_cohort` (binned lifecycle cohorts).
   - `is_auto_pay` (Credit Card / Bank Transfer automatic payment flag).
   - `total_services` (count of add-on security/support services).
   - `charges_per_tenure` (ratio of TotalCharges to tenure+1).
   - `has_fiber_optic` (high-risk service flag).

### Honesty Status
- Implemented locally: All metrics, plots, and figures computed from actual code execution on the Telco CSV.

