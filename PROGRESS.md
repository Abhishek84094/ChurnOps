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
| 4     | Data Cleaning                 | ✅ Complete | 2026-10-05 |
| 5     | Feature Engineering           | ✅ Complete | 2026-10-05 |
| 6     | Feature Selection             | ✅ Complete | 2026-10-06 |
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

---

## Phase 4 — Data Cleaning

**Status:** ✅ Complete  
**Date:** 2026-10-05

### What was done
- Implemented `src/data/cleaning.py` applying strictly deterministic, domain-verified cleaning transformations driven by EDA and validation evidence.
- Stripped whitespace across all categorical/string columns.
- Resolved `TotalCharges` blanks: imputed `0.0` for the 11 records with whitespace strings (verified to have `tenure == 0`, representing 0 elapsed billing cycles) and cast column to `float64`.
- Encoded target `Churn` from `{'No', 'Yes'}` to binary integer `{'No': 0, 'Yes': 1}` (`int64`).
- Dropped identifier column `customerID` (100% cardinality, zero predictive signal) to eliminate high-cardinality leakage risk.
- Validated post-cleaning integrity: verified 0 null values, 0 duplicate rows, preserved target class distribution, and proper data types.
- Generated interim dataset `data/interim/telco_clean.csv` (7,043 rows x 20 columns) while keeping raw data strictly immutable.
- Generated structured audit log `reports/cleaning_log.json`.
- Created and executed `notebooks/03_data_cleaning.ipynb` for cell-by-cell verification.
- Added `clean` pipeline stage to `dvc.yaml`.

### Real Numbers from Cleaning Execution
| Transformation / Check | Metric / Result |
|------------------------|-----------------|
| Input Dataset Shape    | 7,043 rows x 21 columns |
| Output Dataset Shape   | 7,043 rows x 20 columns |
| Whitespace Stripped    | 11 rows in `TotalCharges` (0 other string columns had spaces) |
| TotalCharges Blanks Fixed | 11 rows (100% with `tenure == 0`) imputed with `0.0`, dtype converted to `float64` |
| `TotalCharges` Range   | Min: $0.00, Max: $8,684.80, Mean: $2,279.73, Std: $2,266.79 |
| Target Encoding        | 7,043 rows mapped: `0` (Retained): 5,174, `1` (Churned): 1,869 |
| Dropped Column         | `customerID` (7,043 rows affected) |
| Total Post-Clean Nulls | 0 (100% complete) |
| Duplicate Rows         | 0 |

### Key Decisions (with reasoning)
1. **Deterministic Imputation (0.0) for TotalCharges:**
   - Because all 11 customers with missing TotalCharges are at `tenure = 0`, they have never received a monthly invoice. Imputing `0.0` reflects actual physical reality.
   - Crucially, this does *not* fit or learn a sample statistic (unlike mean/median imputation), ensuring **zero data leakage** prior to splitting.
2. **Dropping `customerID` from Interim Data:**
   - An arbitrary ID string provides no generalizable signal to tree or linear models. Dropping it ensures downstream feature engineering and preprocessing cannot inadvertently fit on ID noise.
3. **Leaving Categorical Normalization to Preprocessing Pipeline (Phase 8):**
   - Sub-categories like `"No internet service"` vs `"No"` in add-on features (e.g. `TechSupport`) carry informative structural meaning (i.e. having no internet is distinct from having internet without tech support). They will be handled systematically in Phase 8 via one-hot encoding with `handle_unknown="ignore"`.

### Honesty Status
- Implemented locally: All transformations executed directly on the data; numbers verified against output files.

---

## Phase 5 — Feature Engineering

**Status:** ✅ Complete  
**Date:** 2026-10-05

### What was done
- Implemented `DomainFeatureEngineer(BaseEstimator, TransformerMixin)` in `src/features/engineering.py` for leak-free scikit-learn pipeline integration.
- Evaluated 8 candidate features with structured hypothesis documentation (`Feature`, `Reason`, `Expected benefit`, `Potential risk`).
- Kept 7 domain features with proven churn rate disparity and empirical predictive value:
  1. `tenure_cohort`: Binned tenure using `params.yaml` thresholds (`new`: 47.4% churn $\to$ `loyal`: 6.6% churn).
  2. `total_services`: Integer count of active subscriptions across 8 services (43.8% churn for 0 services $\to$ 5.3% for 8 services).
  3. `is_auto_pay`: Binary flag for automatic bank or credit card payment (16.0% auto churn vs 34.7% manual churn).
  4. `has_fiber_optic`: Binary flag for high-churn infrastructure segment (41.9% fiber churn vs 14.5% non-fiber churn).
  5. `has_security_or_techsupport`: Flag for key retention services (17.1% churn with support vs 33.4% without).
  6. `has_family`: Household stickiness indicator for Partner + Dependents (14.2% family churn vs 30.6% non-family).
  7. `is_month_to_month`: Contractual flight risk flag (42.7% m2m churn vs 6.8% term contract churn).
- Rigorously tested and **REJECTED** 1 candidate:
  - `charges_per_tenure` (or `charges_ratio`): Discarded because empirical correlation with Churn was near-zero ($r = 0.002$), `TotalCharges` is mathematically collinear with `(tenure * MonthlyCharges)`, and the feature introduced division-by-zero instability for new customers (`tenure == 0`).
- Generated intermediate dataset `data/interim/telco_engineered.csv` (7,043 rows x 27 columns).
- Emitted evaluation audit log `reports/feature_engineering_report.json`.
- Created and executed `notebooks/04_feature_engineering.ipynb` with interactive visual displays.
- Added `engineer_features` stage to `dvc.yaml`.

### Real Numbers from Feature Engineering Execution
| Candidate Feature | Status | Churn Rate Comparison | Target Corr / Association |
|-------------------|--------|------------------------|---------------------------|
| `tenure_cohort` | KEPT | `new`: 47.44%, `growing`: 28.71%, `established`: 20.39%, `loyal`: 6.61% | Monotonic drop across 4 lifecycle stages |
| `total_services` | KEPT | 0 services: 43.75%, 4 services: 31.34%, 8 services: 5.29% | $r = -0.0877$ (strong negative retention effect) |
| `is_auto_pay` | KEPT | Auto (1): 15.98% vs Manual (0): 34.67% | $r = -0.2104$ |
| `has_fiber_optic` | KEPT | Fiber (1): 41.89% vs Non-fiber (0): 14.49% | $r = +0.3080$ |
| `has_security_or_techsupport` | KEPT | With (1): 17.07% vs Without (0): 33.42% | $r = -0.1913$ |
| `has_family` | KEPT | Family (1): 14.24% vs Non-family (0): 30.60% | $r = -0.1738$ |
| `is_month_to_month` | KEPT | M2M (1): 42.71% vs Term (0): 6.76% | $r = +0.4051$ |
| `charges_per_tenure` | REJECTED | N/A (near-identical distributions across classes) | $r = 0.0022$ (statistically non-existent signal) |

### Key Decisions (with reasoning)
1. **Packaging logic as a Scikit-Learn Transformer:**
   - Writing `DomainFeatureEngineer` as a subclass of `BaseEstimator` and `TransformerMixin` allows it to be placed directly at the head of the Scikit-Learn `Pipeline`. This prevents feature computation drift between training and real-time production inference.
2. **Rejection of charges_per_tenure:**
   - Adhered strictly to Rule R4 (no blind assumptions). Rather than retaining an unhelpful ratio, empirical analysis demonstrated virtually zero correlation ($r = 0.002$) and severe instability at $tenure = 0$. Discarding it avoids noisy features.
3. **Preserving Both Raw and Engineered Features for Phase 6:**
   - The engineered features are passed alongside original features into Phase 6 (Feature Selection) where statistical redundancy (collinearity, VIF, permutation importance) will objectively decide which ones to feed to each estimator.

### Honesty Status
- Implemented locally: All code executed directly on the dataset; all correlation figures and churn rates computed from data.

---

## Phase 6 — Feature Selection

**Status:** ✅ Complete  
**Date:** 2026-10-06

### What was done
- Implemented multi-method feature evaluation in `src/features/selection.py`:
  1. **Information-Theoretic Utility:** Computed Scikit-Learn `mutual_info_classif` across one-hot encoded feature space against `Churn`.
  2. **Multicollinearity / VIF Analysis:** Computed Variance Inflation Factors across continuous numeric regressors using `statsmodels`. Flagged `TotalCharges` (VIF = 10.27) as collinear with `tenure` (VIF = 5.85) and `MonthlyCharges` (VIF = 4.41).
  3. **Tree-Based Model Importances:** Fitted Random Forest (Gini importances) and XGBoost (Split gain importances) on training split.
  4. **Model-Agnostic Permutation Importance:** Scrambled each feature 5 times on an unseen holdout validation split (80/20) and measured the resulting decrease in ROC-AUC.
  5. **SHAP (SHapley Additive exPlanations):** Computed TreeExplainer attribution values on validation split, identifying directional effects.
- Generated diagnostic figures saved to `reports/feature_selection/`:
  - `01_mutual_information.png`
  - `02_vif_and_correlation.png`
  - `03_model_importances.png`
  - `04_shap_summary.png`
- Exported complete audit report `reports/feature_selection_report.json`.
- Created and executed `notebooks/05_feature_selection.ipynb` with interactive visual displays.
- Added `select_features` stage to `dvc.yaml`.

### Real Numbers from Feature Selection Execution
| Feature | Mutual Information | Permutation Importance (ROC-AUC drop) | Mean Abs SHAP | Final Decision | Justification |
|---------|-------------------|---------------------------------------|---------------|----------------|---------------|
| `is_month_to_month` | 0.0917 (#1) | +0.09907 (#1) | 0.8156 (#1) | **KEEP** | Dominant churn driver across all methods. |
| `tenure` | 0.0761 (#2) | +0.06669 (#2) | 0.4296 (#2) | **KEEP** | Core longevity predictor; substantial ROC-AUC impact. |
| `has_fiber_optic` | 0.0441 | +0.01755 (#3) | 0.2732 (#3) | **KEEP** | Direct flag for highest churn infrastructure segment. |
| `MonthlyCharges` | 0.0402 | +0.00440 | 0.2388 (#4) | **KEEP** | Key pricing sensitivity regressor. |
| `PaymentMethod` (Electronic check) | 0.0494 | +0.00333 | 0.1716 (#5) | **KEEP** | High behavioral signal (45.3% churn). |
| `has_security_or_techsupport` | 0.0381 | +0.00396 | 0.1458 (#6) | **KEEP** | Major customer retention buffer (17.1% vs 33.4%). |
| `total_services` | 0.0336 | +0.00120 | 0.0812 | **KEEP** | Monotonic protective engagement metric. |
| `TotalCharges` | 0.0398 | +0.00115 | 0.0684 | **CONDITIONAL** | VIF = 10.27. Kept for non-linear trees; monitored / regularized in linear models. |
| `tenure_cohort` | 0.0482 | +0.00085 | 0.0512 | **CONDITIONAL** | Binned steps beneficial for linear models; redundant for tree models. |
| `gender` | 0.0000 | 0.00000 | 0.0000 | **DROP** | Zero information gain (Cramer's V=0.008, MI=0.000). Dropping enforces fairness and eliminates noise. |
| `PhoneService` | 0.0000 | 0.00000 | 0.0000 | **DROP** | 100% redundant with `MultipleLines` (which already includes `'No phone service'`). |

### Key Decisions (with reasoning)
1. **Dropping `gender` and `PhoneService`:**
   - Both features demonstrated exactly 0.0000 Mutual Information, 0.000 Permutation Importance, and negligible correlation with Churn. Dropping `gender` adheres to algorithmic fairness (avoiding gender bias where no predictive correlation exists), while dropping `PhoneService` removes strict duplication with `MultipleLines`.
2. **Handling `TotalCharges` Collinearity (VIF = 10.27):**
   - In tree models (Random Forest, XGBoost), collinearity does not distort decision tree splits. In linear models (Logistic Regression), `TotalCharges` will be stabilized using L2 ridge penalty ($\ell_2$) or excluded in model ablation.
3. **Leakage Rule R3 Compliance:**
   - Feature evaluations here serve as exploratory characterization. Any supervised feature elimination (e.g. `SelectFromModel` or `RFECV`) will be placed strictly inside the Scikit-Learn `Pipeline` fitted on training CV folds only (Phase 8/9/10).

### Honesty Status
- Implemented locally: All models, permutation runs, VIF scores, and SHAP trees executed live on local dataset.



