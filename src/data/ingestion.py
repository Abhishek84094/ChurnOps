"""
Phase 1 — Data Ingestion
========================
Responsibilities:
  1. Locate raw CSV (path from params.yaml, never modified).
  2. Run integrity checks: exists, non-empty, correct extension, hash computed.
  3. Load into pandas and compute summary statistics.
  4. Write reports/dataset_metadata.json (real numbers only — R1).
  5. Exit with non-zero code on critical failure; warn on non-critical.

Usage (called by DVC stage or directly):
    .venv/Scripts/python src/data/ingestion.py
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Allow running as a script OR imported as a module
_HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_HERE))

from src.utils.config import PARAMS, get_path
from src.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    """Compute SHA-256 hash of a file in 1-MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while buf := fh.read(chunk):
            h.update(buf)
    return h.hexdigest()


def _integrity_checks(path: Path) -> None:
    """
    Critical checks — exit loudly on failure (R2 integrity guarantee).
    """
    if not path.exists():
        log.error("CRITICAL: Raw data file not found: %s", path)
        sys.exit(1)
    if path.stat().st_size == 0:
        log.error("CRITICAL: Raw data file is empty: %s", path)
        sys.exit(1)
    if path.suffix.lower() != ".csv":
        log.error("CRITICAL: Expected .csv extension, got: %s", path.suffix)
        sys.exit(1)
    log.info("Integrity checks passed: %s (%.1f KB)", path.name, path.stat().st_size / 1024)


def _dtype_summary(df: pd.DataFrame) -> dict:
    """Return dict of column -> dtype string."""
    return {col: str(df[col].dtype) for col in df.columns}


def _missing_summary(df: pd.DataFrame) -> dict:
    """Return dict of column -> {count, pct} for columns with any missing."""
    total = len(df)
    missing = {}
    for col in df.columns:
        n = int(df[col].isna().sum())
        if n > 0:
            missing[col] = {"count": n, "pct": round(100 * n / total, 4)}
    return missing


def _target_distribution(df: pd.DataFrame, target: str) -> dict:
    """Return value counts and percentages for the target column."""
    if target not in df.columns:
        return {"error": f"Column '{target}' not found"}
    counts = df[target].value_counts(dropna=False)
    total = len(df)
    return {
        str(k): {"count": int(v), "pct": round(100 * v / total, 4)}
        for k, v in counts.items()
    }


def _numeric_summary(df: pd.DataFrame) -> dict:
    """Descriptive stats for numeric columns."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    summary = {}
    for col in num_cols:
        s = df[col]
        summary[col] = {
            "min":    round(float(s.min()), 4),
            "max":    round(float(s.max()), 4),
            "mean":   round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "std":    round(float(s.std()), 4),
        }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_ingestion() -> dict:
    """
    Full ingestion pipeline. Returns the metadata dict.
    Writes reports/dataset_metadata.json.
    """
    raw_path    = get_path("raw_data")
    reports_dir = get_path("reports_dir")
    target_col  = PARAMS["dataset"]["target_column"]
    id_col      = PARAMS["dataset"]["id_column"]

    reports_dir.mkdir(parents=True, exist_ok=True)

    # -- 1. Integrity checks --------------------------------------------------
    _integrity_checks(raw_path)

    # -- 2. Compute file hash BEFORE loading (R8: never mutate raw) -----------
    file_hash = _sha256(raw_path)
    file_size = raw_path.stat().st_size
    log.info("SHA-256: %s", file_hash)

    # -- 3. Load dataframe ----------------------------------------------------
    df = pd.read_csv(raw_path, dtype=str)  # dtype=str avoids silent coercions
    log.info("Loaded: %d rows × %d columns", len(df), len(df.columns))

    # -- 4. Expected-size checks (warn, not fatal) ----------------------------
    min_rows = PARAMS["dataset"]["expected_min_rows"]
    min_cols = PARAMS["dataset"]["expected_min_cols"]
    if len(df) < min_rows:
        log.warning("Row count %d below expected minimum %d", len(df), min_rows)
    if len(df.columns) < min_cols:
        log.warning("Column count %d below expected minimum %d", len(df.columns), min_cols)

    # -- 5. Duplicate check ---------------------------------------------------
    n_dup_rows = int(df.duplicated().sum())
    n_dup_ids  = int(df[id_col].duplicated().sum()) if id_col in df.columns else "N/A"
    if n_dup_rows > 0:
        log.warning("Found %d duplicate rows", n_dup_rows)
    else:
        log.info("No duplicate rows found")

    # -- 6. Target presence check ---------------------------------------------
    if target_col not in df.columns:
        log.error("CRITICAL: Target column '%s' not in dataset", target_col)
        sys.exit(1)
    unique_targets = df[target_col].dropna().unique().tolist()
    log.info("Target '%s' unique values: %s", target_col, unique_targets)

    # -- 7. Detect TotalCharges dtype issue (known dataset quirk) -------------
    if "TotalCharges" in df.columns:
        blank_tc = int((df["TotalCharges"].str.strip() == "").sum())
        log.info(
            "TotalCharges column: dtype=object, blank-string count=%d "
            "(Note: stored as text; conversion and handling in Phase 4)", blank_tc
        )
    else:
        blank_tc = None

    # -- 8. Detect extended IBM columns (leakage risk) -----------------------
    leakage_risk_cols = [
        c for c in ["Churn Label", "Churn Score", "Churn Reason", "CLTV"]
        if c in df.columns
    ]
    if leakage_risk_cols:
        log.warning(
            "LEAKAGE RISK: Extended IBM columns detected: %s. "
            "These will be excluded from features.", leakage_risk_cols
        )

    # -- 9. Assemble metadata dict --------------------------------------------
    metadata = {
        "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset":   "IBM Telco Customer Churn",
            "filename":  raw_path.name,
            "filepath":  str(raw_path),
            "url_note":  "Kaggle: blastchar/telco-customer-churn (also IBM GitHub sample-data)",
        },
        "file": {
            "size_bytes": file_size,
            "size_kb":    round(file_size / 1024, 2),
            "sha256":     file_hash,
            "extension":  raw_path.suffix,
        },
        "shape": {
            "rows":    len(df),
            "columns": len(df.columns),
        },
        "columns": df.columns.tolist(),
        "dtypes_as_read": _dtype_summary(df),
        "missing_values": _missing_summary(df),
        "duplicates": {
            "duplicate_rows":        n_dup_rows,
            "duplicate_customer_ids": n_dup_ids,
        },
        "target": {
            "column":       target_col,
            "distribution": _target_distribution(df, target_col),
        },
        "id_column": id_col,
        "known_quirks": {
            "TotalCharges_blank_strings": blank_tc,
            "SeniorCitizen_is_01_not_YesNo": "SeniorCitizen" in df.columns,
            "leakage_risk_columns_present": leakage_risk_cols,
        },
    }

    # -- 10. Write report -----------------------------------------------------
    out_path = reports_dir / "dataset_metadata.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    log.info("Metadata written → %s", out_path)

    return metadata


if __name__ == "__main__":
    meta = run_ingestion()

    # Print human-readable summary to stdout
    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"File         : {meta['source']['filename']}")
    print(f"Size         : {meta['file']['size_kb']} KB")
    print(f"SHA-256      : {meta['file']['sha256'][:16]}…")
    print(f"Shape        : {meta['shape']['rows']} rows × {meta['shape']['columns']} columns")
    print(f"Duplicate rows: {meta['duplicates']['duplicate_rows']}")
    print(f"Dup IDs      : {meta['duplicates']['duplicate_customer_ids']}")
    print(f"\nTarget '{meta['target']['column']}' distribution:")
    for val, info in meta["target"]["distribution"].items():
        print(f"  {val:<10} : {info['count']:>5}  ({info['pct']}%)")
    print(f"\nMissing values: {len(meta['missing_values'])} column(s) affected")
    for col, info in meta["missing_values"].items():
        print(f"  {col}: {info['count']} ({info['pct']}%)")
    print(f"\nKnown quirks:")
    for k, v in meta["known_quirks"].items():
        print(f"  {k}: {v}")
    print("=" * 60)
