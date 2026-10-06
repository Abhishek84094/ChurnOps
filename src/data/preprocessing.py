"""
Phase 8 — Preprocessing Pipeline Module
=======================================
Constructs modular, reusable Scikit-Learn Preprocessing Pipelines using
ColumnTransformer, DomainFeatureEngineer, StandardScaler, and OneHotEncoder.

Features & Transformations:
  1. Feature Engineering: DomainFeatureEngineer() (from Phase 5).
  2. Feature Dropping: Drops 'gender' and 'PhoneService' (justified in Phase 6).
  3. Numeric Regressors: Median imputation + StandardScaler (scale_numeric=True)
     or Passthrough (scale_numeric=False for tree models).
  4. Categorical Variables: Most-frequent imputation + OneHotEncoder(handle_unknown="ignore").
  5. Binary Indicators: Direct passthrough (SeniorCitizen, is_auto_pay, etc.).

Outputs:
  - models/preprocessor.joblib (fitted on train.csv only)
  - reports/preprocessing_report.json (schema, feature names, fitted statistics)

Leakage Rule (R3):
  Fitted exclusively on data/processed/train.csv.
  The locked test set (data/processed/test.csv) is never accessed or fitted.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow running as a script OR imported as a module
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.engineering import DomainFeatureEngineer
from src.utils.config import PARAMS, get_path
from src.utils.logging import get_logger

log = get_logger("preprocessing")

# Column grouping definitions
DROP_COLUMNS = ["gender", "PhoneService"]
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges", "total_services"]
PASSTHROUGH_BINARY_COLUMNS = [
    "SeniorCitizen",
    "is_auto_pay",
    "has_fiber_optic",
    "has_security_or_techsupport",
    "has_family",
    "is_month_to_month",
]
CATEGORICAL_COLUMNS = [
    "Partner",
    "Dependents",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_cohort",
]


def build_column_transformer(
    scale_numeric: bool = True,
    numeric_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
    pass_cols: Optional[List[str]] = None,
    drop_cols: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer for feature preprocessing.

    Args:
        scale_numeric: If True, applies StandardScaler (for linear models).
                       If False, passes numeric values unscaled (for tree models).
        numeric_cols: List of numeric column names.
        cat_cols: List of categorical column names.
        pass_cols: List of binary/passthrough column names.
        drop_cols: List of columns to drop.

    Returns:
        Configured ColumnTransformer instance.
    """
    num_cols = numeric_cols if numeric_cols is not None else NUMERIC_COLUMNS
    categoricals = cat_cols if cat_cols is not None else CATEGORICAL_COLUMNS
    passthroughs = pass_cols if pass_cols is not None else PASSTHROUGH_BINARY_COLUMNS
    drops = drop_cols if drop_cols is not None else DROP_COLUMNS

    # Numeric sub-pipeline
    num_steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(num_steps)

    # Categorical sub-pipeline
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = [
        ("drop", "drop", drops),
        ("num", numeric_pipe, num_cols),
        ("cat", cat_pipe, categoricals),
        ("pass", "passthrough", passthroughs),
    ]

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_full_pipeline(
    estimator: Optional[Any] = None,
    scale_numeric: bool = True,
) -> Pipeline:
    """
    Builds an end-to-end Pipeline:
      DomainFeatureEngineer -> ColumnTransformer -> Optional[Estimator]

    The resulting pipeline accepts raw/clean customer records (19 features)
    and handles all downstream engineering and transformation seamlessly.
    """
    steps: List[Tuple[str, Any]] = [
        ("feature_engineer", DomainFeatureEngineer()),
        ("preprocessor", build_column_transformer(scale_numeric=scale_numeric)),
    ]
    if estimator is not None:
        steps.append(("model", estimator))

    return Pipeline(steps)


def fit_and_audit_preprocessing() -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Fits the preprocessing pipeline on train.csv ONLY and generates an audit report.
    """
    train_path = Path("data/processed/train.csv")
    if not train_path.exists():
        raise FileNotFoundError(f"Train dataset not found at: {train_path}")

    log.info("Loading training dataset from: %s", train_path)
    train_df = pd.read_csv(train_path)

    target_col = PARAMS["dataset"]["target_column"]
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    # Build and fit pipeline
    full_pipeline = build_full_pipeline(scale_numeric=True)
    full_pipeline.fit(X_train, y_train)
    log.info("Preprocessing pipeline fitted successfully on %d training rows", len(X_train))

    # Transform training data to inspect outputs
    X_train_proc = full_pipeline.transform(X_train)
    col_transformer: ColumnTransformer = full_pipeline.named_steps["preprocessor"]
    output_feature_names = col_transformer.get_feature_names_out().tolist()

    # Extract fitted scaler parameters for transparency
    num_pipeline = col_transformer.named_transformers_["num"]
    scaler = num_pipeline.named_steps.get("scaler")
    scaler_stats = {}
    if scaler is not None:
        for col, mean_val, var_val in zip(NUMERIC_COLUMNS, scaler.mean_, scaler.var_):
            scaler_stats[col] = {
                "mean": round(float(mean_val), 3),
                "std": round(float(np.sqrt(var_val)), 3),
            }

    audit_report: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_features_count": X_train.shape[1],
        "output_features_count": len(output_feature_names),
        "engineered_features_added": [
            "tenure_cohort",
            "total_services",
            "is_auto_pay",
            "has_fiber_optic",
            "has_security_or_techsupport",
            "has_family",
            "is_month_to_month",
        ],
        "dropped_features": DROP_COLUMNS,
        "numeric_features_scaled": NUMERIC_COLUMNS,
        "scaler_fitted_parameters": scaler_stats,
        "categorical_features_encoded": CATEGORICAL_COLUMNS,
        "passthrough_binary_features": PASSTHROUGH_BINARY_COLUMNS,
        "output_feature_names": output_feature_names,
        "leakage_boundary": {
            "fitted_on": "data/processed/train.csv (5,634 rows)",
            "test_set_access": "ZERO (test.csv untouched, locked)",
            "ohe_unknown_policy": "ignore (guarantees runtime inference robustness)",
        },
    }

    return full_pipeline, audit_report


def run_preprocessing_pipeline() -> Dict[str, Any]:
    """Orchestrates Phase 8 execution and artifact export."""
    log.info("Starting Phase 8 — Preprocessing Pipeline")

    pipeline, audit_report = fit_and_audit_preprocessing()

    # 1. Save serialized fitted pipeline
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "preprocessor.joblib"
    joblib.dump(pipeline, model_path)
    log.info("Saved serialized preprocessor -> %s", model_path)

    # 2. Save audit report
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "preprocessing_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)
    log.info("Saved preprocessing report -> %s", report_path)

    return audit_report


if __name__ == "__main__":
    rep = run_preprocessing_pipeline()
    print("\n" + "=" * 60)
    print("PHASE 8 — PREPROCESSING PIPELINE COMPLETED")
    print("=" * 60)
    print(f"Input Features : {rep['input_features_count']} columns")
    print(f"Output Features: {rep['output_features_count']} columns")
    print(f"Dropped        : {rep['dropped_features']}")
    print(f"Numeric Scaled : {rep['numeric_features_scaled']}")
    print("\nFitted Scaler Means & Stds (Train Set Only):")
    for col, stats in rep["scaler_fitted_parameters"].items():
        print(f"  - {col:16s}: mean={stats['mean']:8.2f}, std={stats['std']:8.2f}")
    print(f"\nSerialized Pipeline: models/preprocessor.joblib")
    print(f"Leakage Status: {rep['leakage_boundary']['test_set_access']}")
    print("=" * 60)
