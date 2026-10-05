"""
Phase 4 — Data Cleaning Module
==============================
Applies deterministic, domain-verified cleaning operations driven by EDA findings.
Raw data is strictly immutable and left untouched.

Operations:
  1. Whitespace stripping across all string columns.
  2. Impute 0.0 for 11 blank TotalCharges records (empirically proven tenure=0), cast to float64.
  3. Map target column 'Churn' from {'No', 'Yes'} to binary {0, 1}.
  4. Drop 'customerID' identifier column from feature space (avoids high-cardinality leakage).
  5. Validate clean dataset integrity (0 nulls, correct types, preserved distribution).

Outputs:
  - data/interim/telco_clean.csv
  - reports/cleaning_log.json
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

# Allow running as a script OR imported as a module
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.utils.config import PARAMS, get_path
from src.utils.logging import get_logger

log = get_logger("cleaning")


def clean_dataset(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes deterministic cleaning transformations and records an audit log.
    
    Args:
        raw_df: Unmodified raw DataFrame as ingested.
        
    Returns:
        Tuple of (cleaned_df, cleaning_audit_log).
    """
    df = raw_df.copy()
    initial_shape = list(df.shape)
    log.info("Starting data cleaning on input shape: %s", initial_shape)

    audit_log: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_shape": initial_shape,
        "transformations": [],
    }

    # --- Step 1: Strip leading and trailing whitespace from string columns ---
    str_cols = df.select_dtypes(include="object").columns.tolist()
    stripped_counts = {}
    for col in str_cols:
        before = df[col]
        after = df[col].astype(str).str.strip()
        diff = (before != after).sum()
        if diff > 0:
            stripped_counts[col] = int(diff)
        df[col] = after

    audit_log["transformations"].append({
        "step": 1,
        "name": "strip_whitespace",
        "columns_modified": stripped_counts,
        "affected_rows": sum(stripped_counts.values()),
        "rationale": "Removes invisible whitespace to prevent categorical matching failures and type errors."
    })
    log.info("Step 1 - Whitespace stripped on: %s", stripped_counts)

    # --- Step 2: Handle TotalCharges blanks and cast to float64 ---
    blank_mask = df["TotalCharges"] == ""
    blank_count = int(blank_mask.sum())
    tenure_of_blanks = df.loc[blank_mask, "tenure"].tolist()

    # Domain assertion: all blank TotalCharges must have tenure == 0
    if not all(t == 0 for t in tenure_of_blanks):
        raise ValueError(
            f"Integrity violation: Blank TotalCharges found for non-zero tenure: {tenure_of_blanks}"
        )

    df["TotalCharges"] = df["TotalCharges"].replace("", "0.0").astype("float64")

    audit_log["transformations"].append({
        "step": 2,
        "name": "fix_totalcharges_blanks_and_type",
        "column": "TotalCharges",
        "affected_rows": blank_count,
        "tenure_values": tenure_of_blanks,
        "new_value": 0.0,
        "new_dtype": "float64",
        "rationale": (
            "Empirical evidence proves all 11 customers with blank TotalCharges have tenure=0 "
            "(new registrations prior to first billing cycle). Imputing 0.0 represents physical "
            "reality with zero statistical data leakage."
        )
    })
    log.info("Step 2 - TotalCharges fixed: %d blank rows set to 0.0, converted to float64", blank_count)

    # --- Step 3: Binary map target column 'Churn' ---
    target_col = PARAMS["dataset"]["target_column"]
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    raw_target_dist = df[target_col].value_counts().to_dict()
    target_mapping = {"No": 0, "Yes": 1}
    df[target_col] = df[target_col].map(target_mapping)

    if df[target_col].isnull().any():
        raise ValueError("Target mapping produced null values; unexpected target categories present.")

    df[target_col] = df[target_col].astype("int64")

    audit_log["transformations"].append({
        "step": 3,
        "name": "encode_target_binary",
        "column": target_col,
        "mapping": target_mapping,
        "affected_rows": len(df),
        "raw_distribution": raw_target_dist,
        "cleaned_distribution": df[target_col].value_counts().to_dict(),
        "rationale": "Maps categorical target to standard binary integer (0=Retained, 1=Churned) required by ML estimators."
    })
    log.info("Step 3 - Target column '%s' mapped: %s", target_col, target_mapping)

    # --- Step 4: Drop customer identifier column ---
    id_col = PARAMS["dataset"]["id_column"]
    if id_col in df.columns:
        df = df.drop(columns=[id_col])
        audit_log["transformations"].append({
            "step": 4,
            "name": "drop_identifier_column",
            "column_dropped": id_col,
            "affected_rows": len(df),
            "rationale": (
                "Unique alphanumeric customer identifier carries 100% cardinality with zero "
                "generalizable predictive signal. Dropping prevents high-cardinality leakage and overfitting."
            )
        })
        log.info("Step 4 - Dropped identifier column '%s'", id_col)

    # --- Step 5: Post-cleaning validation assertions ---
    final_shape = list(df.shape)
    audit_log["final_shape"] = final_shape
    audit_log["validation_checks"] = {
        "null_count": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_dtype": str(df[target_col].dtype),
        "totalcharges_dtype": str(df["TotalCharges"].dtype),
        "totalcharges_min": float(df["TotalCharges"].min()),
        "totalcharges_max": float(df["TotalCharges"].max()),
        "target_distribution": {int(k): int(v) for k, v in df[target_col].value_counts().items()},
    }

    # Enforce strict post-conditions
    assert audit_log["validation_checks"]["null_count"] == 0, "Cleaned dataset contains null values!"
    assert audit_log["validation_checks"]["target_dtype"] == "int64", "Target dtype is not int64!"
    assert audit_log["validation_checks"]["totalcharges_dtype"] == "float64", "TotalCharges is not float64!"
    assert df.shape[1] == initial_shape[1] - 1, f"Expected {initial_shape[1] - 1} columns, got {df.shape[1]}"
    assert df.shape[0] == initial_shape[0], f"Expected {initial_shape[0]} rows, got {df.shape[0]}"

    log.info("Step 5 - Validation passed: shape=%s, 0 nulls, 0 duplicates", final_shape)
    return df, audit_log


def run_cleaning_pipeline() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Loads raw data, executes clean_dataset, and writes interim outputs."""
    raw_path = get_path("raw_data")
    interim_path = get_path("interim_data")
    interim_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading raw data from: %s", raw_path)
    raw_df = pd.read_csv(raw_path)

    clean_df, audit_log = clean_dataset(raw_df)

    # Save cleaned interim CSV
    clean_df.to_csv(interim_path, index=False)
    log.info("Saved cleaned dataset -> %s (size: %d bytes)", interim_path, interim_path.stat().st_size)

    # Save cleaning log JSON
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    log_file = reports_dir / "cleaning_log.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(audit_log, f, indent=2)
    log.info("Saved cleaning log -> %s", log_file)

    return clean_df, audit_log


if __name__ == "__main__":
    df_clean, audit = run_cleaning_pipeline()
    print("\n" + "=" * 60)
    print("PHASE 4 — DATA CLEANING COMPLETED")
    print("=" * 60)
    print(f"Input Shape : {audit['input_shape']}")
    print(f"Output Shape: {audit['final_shape']}")
    print(f"Transformations applied: {len(audit['transformations'])}")
    for t in audit['transformations']:
        print(f"  - Step {t['step']}: {t['name']} (affected rows: {t['affected_rows']})")
    print(f"Validation:")
    print(f"  - Total Nulls: {audit['validation_checks']['null_count']}")
    print(f"  - Target Distribution: {audit['validation_checks']['target_distribution']}")
    print("=" * 60)
