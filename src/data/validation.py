"""
Phase 2 - Data Validation
=========================
Produces: reports/validation_report.json + console summary
Exits non-zero on CRITICAL failures (missing required cols, invalid target).
Warns (but continues) on non-critical issues.
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np

_HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_HERE))

from src.utils.config import PARAMS, get_path
from src.utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
EXPECTED_COLS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"
]

NUMERIC_COLS   = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
BINARY_YES_NO  = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
CATEGORICAL    = {
    "gender":          {"Female", "Male"},
    "MultipleLines":   {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity":  {"Yes", "No", "No internet service"},
    "OnlineBackup":    {"Yes", "No", "No internet service"},
    "DeviceProtection":{"Yes", "No", "No internet service"},
    "TechSupport":     {"Yes", "No", "No internet service"},
    "StreamingTV":     {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract":        {"Month-to-month", "One year", "Two year"},
    "PaymentMethod":   {
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    },
}
NUMERIC_RANGES = {
    "tenure":         (0, 200),
    "MonthlyCharges": (0, 200),
    "SeniorCitizen":  (0, 1),
}
LEAKAGE_COLS = ["Churn Label", "Churn Score", "Churn Reason", "CLTV"]

CRITICAL   = []
WARNINGS   = []

def _crit(msg):
    CRITICAL.append(msg)
    log.error("CRITICAL: %s", msg)

def _warn(msg):
    WARNINGS.append(msg)
    log.warning("WARN: %s", msg)

def _ok(msg):
    log.info("OK: %s", msg)


def run_validation() -> dict:
    raw_path    = get_path("raw_data")
    reports_dir = get_path("reports_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    target_col  = PARAMS["dataset"]["target_column"]
    id_col      = PARAMS["dataset"]["id_column"]

    df_raw = pd.read_csv(raw_path, dtype=str)
    n_rows, n_cols = df_raw.shape
    log.info("Loaded %d rows x %d cols", n_rows, n_cols)

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "shape": {"rows": n_rows, "cols": n_cols},
        "schema": {},
        "missing": {},
        "duplicates": {},
        "numeric": {},
        "categorical": {},
        "target": {},
        "leakage_check": {},
        "criticals": [],
        "warnings": [],
    }

    # ---- 1. SCHEMA -------------------------------------------------------
    actual_cols  = set(df_raw.columns)
    expected_set = set(EXPECTED_COLS)
    missing_cols = expected_set - actual_cols
    extra_cols   = actual_cols - expected_set

    if missing_cols:
        _crit(f"Required columns missing: {sorted(missing_cols)}")
    else:
        _ok("All expected columns present")

    if extra_cols:
        _warn(f"Unexpected extra columns (check for leakage): {sorted(extra_cols)}")

    results["schema"] = {
        "expected_cols":   EXPECTED_COLS,
        "actual_cols":     df_raw.columns.tolist(),
        "missing_cols":    sorted(missing_cols),
        "extra_cols":      sorted(extra_cols),
    }

    # ---- 2. LEAKAGE COLUMNS ----------------------------------------------
    found_leakage = [c for c in LEAKAGE_COLS if c in df_raw.columns]
    if found_leakage:
        _crit(f"Target-leakage columns present: {found_leakage}")
    else:
        _ok("No leakage columns detected")
    results["leakage_check"] = {"found": found_leakage}

    # ---- 3. MISSING VALUES -----------------------------------------------
    missing_report = {}
    for col in df_raw.columns:
        n_nan    = int(df_raw[col].isna().sum())
        n_blank  = int((df_raw[col].astype(str).str.strip() == "").sum())
        n_total  = n_nan + n_blank
        if n_total > 0:
            missing_report[col] = {
                "nan_count":   n_nan,
                "blank_count": n_blank,
                "total":       n_total,
                "pct":         round(100 * n_total / n_rows, 4),
            }
            if col == "TotalCharges" and n_blank > 0:
                _warn(f"TotalCharges: {n_blank} blank strings (not NaN) - stored as text, needs conversion")
            elif n_total > 0:
                _warn(f"Column '{col}' has {n_total} missing/blank values ({round(100*n_total/n_rows,2)}%)")

    if not missing_report:
        _ok("No NaN or blank-string missing values beyond TotalCharges")
    results["missing"] = missing_report

    # ---- 4. DUPLICATES ---------------------------------------------------
    n_dup_rows = int(df_raw.duplicated().sum())
    n_dup_ids  = int(df_raw[id_col].duplicated().sum()) if id_col in df_raw.columns else "N/A"
    if n_dup_rows > 0:
        _warn(f"{n_dup_rows} duplicate rows found")
    else:
        _ok("No duplicate rows")
    if n_dup_ids and n_dup_ids > 0:
        _warn(f"{n_dup_ids} duplicate customer IDs found")
    results["duplicates"] = {"duplicate_rows": n_dup_rows, "duplicate_ids": n_dup_ids}

    # ---- 5. NUMERIC CHECKS -----------------------------------------------
    numeric_results = {}
    for col in NUMERIC_COLS:
        if col not in df_raw.columns:
            continue
        raw_series = df_raw[col].copy()

        # TotalCharges: blank strings must be excluded before numeric conversion
        if col == "TotalCharges":
            clean = raw_series[raw_series.str.strip() != ""]
        else:
            clean = raw_series

        try:
            numeric = pd.to_numeric(clean, errors="raise")
            mn, mx = float(numeric.min()), float(numeric.max())
            mean, med, std = float(numeric.mean()), float(numeric.median()), float(numeric.std())
            lo, hi = NUMERIC_RANGES.get(col, (None, None))
            out_of_range = int((numeric < lo).sum() + (numeric > hi).sum()) if lo is not None else 0
            if out_of_range:
                _warn(f"{col}: {out_of_range} values outside expected range [{lo}, {hi}]")
            else:
                if lo is not None:
                    _ok(f"{col}: range [{mn:.2f}, {mx:.2f}] within bounds [{lo}, {hi}]")
            non_numeric_count = int(pd.to_numeric(raw_series, errors="coerce").isna().sum()) - int(raw_series.isna().sum())
            numeric_results[col] = {
                "min": round(mn, 4), "max": round(mx, 4),
                "mean": round(mean, 4), "median": round(med, 4), "std": round(std, 4),
                "out_of_range": out_of_range,
                "non_numeric_strings": max(0, non_numeric_count),
            }
        except Exception as e:
            _warn(f"{col}: could not convert to numeric - {e}")
            numeric_results[col] = {"error": str(e)}

    results["numeric"] = numeric_results

    # ---- 6. CATEGORICAL CHECKS -------------------------------------------
    cat_results = {}
    for col, expected_vals in CATEGORICAL.items():
        if col not in df_raw.columns:
            continue
        actual_vals = set(df_raw[col].dropna().unique())
        unexpected  = actual_vals - expected_vals
        missing_exp = expected_vals - actual_vals
        value_counts = df_raw[col].value_counts(dropna=False).to_dict()
        value_counts = {str(k): int(v) for k, v in value_counts.items()}
        if unexpected:
            _warn(f"Column '{col}': unexpected categories: {unexpected}")
        else:
            _ok(f"Column '{col}': all {len(actual_vals)} categories valid")
        cat_results[col] = {
            "expected_categories": sorted(expected_vals),
            "actual_categories":   sorted(actual_vals),
            "unexpected":          sorted(unexpected),
            "missing_expected":    sorted(missing_exp),
            "value_counts":        value_counts,
        }

    # Binary Yes/No columns
    for col in BINARY_YES_NO:
        if col not in df_raw.columns:
            continue
        actual_vals   = set(df_raw[col].dropna().unique())
        unexpected    = actual_vals - {"Yes", "No"}
        value_counts  = df_raw[col].value_counts(dropna=False).to_dict()
        value_counts  = {str(k): int(v) for k, v in value_counts.items()}
        if unexpected:
            _warn(f"Column '{col}': unexpected values (expected Yes/No): {unexpected}")
        else:
            _ok(f"Column '{col}': valid Yes/No categories")
        cat_results[col] = {
            "expected_categories": ["No", "Yes"],
            "actual_categories":   sorted(actual_vals),
            "unexpected":          sorted(unexpected),
            "value_counts":        value_counts,
        }

    results["categorical"] = cat_results

    # ---- 7. TARGET VALIDATION --------------------------------------------
    target_vals  = df_raw[target_col].dropna().unique().tolist() if target_col in df_raw.columns else []
    invalid_tgt  = [v for v in target_vals if v not in ("Yes", "No")]
    dist         = df_raw[target_col].value_counts(dropna=False).to_dict()
    dist         = {str(k): int(v) for k, v in dist.items()}
    null_tgt     = int(df_raw[target_col].isna().sum()) if target_col in df_raw.columns else 0

    if target_col not in df_raw.columns:
        _crit(f"Target column '{target_col}' not found")
    elif invalid_tgt:
        _crit(f"Target column contains invalid values: {invalid_tgt}")
    elif null_tgt > 0:
        _crit(f"Target column has {null_tgt} null values")
    else:
        _ok(f"Target '{target_col}' valid: classes={target_vals}")

    results["target"] = {
        "column": target_col,
        "unique_values": target_vals,
        "invalid_values": invalid_tgt,
        "null_count": null_tgt,
        "distribution": dist,
        "imbalance_ratio": round(dist.get("No", 0) / dist.get("Yes", 1), 4),
    }

    # ---- 8. Finalize -------------------------------------------------------
    results["criticals"] = CRITICAL
    results["warnings"]  = WARNINGS
    results["passed"]    = len(CRITICAL) == 0

    out_path = reports_dir / "validation_report.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    log.info("Validation report -> %s", out_path)

    return results


if __name__ == "__main__":
    r = run_validation()

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    status = "PASSED" if r["passed"] else "FAILED"
    print(f"Status        : {status}")
    print(f"Shape         : {r['shape']['rows']} rows x {r['shape']['cols']} cols")
    print(f"Schema        : {len(r['schema']['missing_cols'])} missing cols, {len(r['schema']['extra_cols'])} extra cols")
    print(f"Duplicates    : {r['duplicates']['duplicate_rows']} rows, {r['duplicates']['duplicate_ids']} IDs")
    print(f"Target        : {r['target']['unique_values']}  |  imbalance ratio (No:Yes): {r['target']['imbalance_ratio']}")
    print(f"\nNumeric ranges:")
    for col, s in r["numeric"].items():
        if "min" in s:
            print(f"  {col:<20}: [{s['min']}, {s['max']}]  mean={s['mean']}  std={s['std']}")
    print(f"\nMissing / blank-string issues:")
    if r["missing"]:
        for col, info in r["missing"].items():
            print(f"  {col}: nan={info['nan_count']}, blank={info['blank_count']}")
    else:
        print("  None (TotalCharges blanks noted separately)")
    print(f"\nCritical errors  : {len(r['criticals'])}")
    for c in r["criticals"]:
        print(f"  [CRITICAL] {c}")
    print(f"Warnings         : {len(r['warnings'])}")
    for w in r["warnings"]:
        print(f"  [WARN] {w}")
    print("=" * 60)

    if not r["passed"]:
        sys.exit(1)
