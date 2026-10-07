"""
Phase 17 — Pipeline Automation
================================
Single entry-point script to execute the full ChurnOps MLOps pipeline
end-to-end. Runs each DVC stage in dependency order, logs progress, and
generates a run manifest.

Usage:
    python scripts/run_pipeline.py                    # Full pipeline
    python scripts/run_pipeline.py --stages ingest clean split   # Selected stages
    python scripts/run_pipeline.py --dry-run          # Print plan without executing
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import get_logger

log = get_logger("pipeline_runner")

# ─── Ordered pipeline stages ──────────────────────────────────────────────────
PIPELINE_STAGES = [
    # DVC-managed stages (run via dvc repro)
    {"name": "ingest",            "type": "dvc",  "desc": "Phase 1: Data Ingestion"},
    {"name": "validate",          "type": "dvc",  "desc": "Phase 2: Data Validation"},
    {"name": "eda",               "type": "dvc",  "desc": "Phase 3: Exploratory Data Analysis"},
    {"name": "clean",             "type": "dvc",  "desc": "Phase 4: Data Cleaning"},
    {"name": "engineer_features", "type": "dvc",  "desc": "Phase 5: Feature Engineering"},
    {"name": "select_features",   "type": "dvc",  "desc": "Phase 6: Feature Selection"},
    {"name": "split",             "type": "dvc",  "desc": "Phase 7: Train/Test Split"},
    {"name": "preprocess",        "type": "dvc",  "desc": "Phase 8: Preprocessing Pipeline"},
    {"name": "baseline_train",    "type": "dvc",  "desc": "Phase 9: Baseline Models"},
    {"name": "cv_analysis",       "type": "dvc",  "desc": "Phase 10: Cross-Validation"},
    {"name": "diagnose",          "type": "dvc",  "desc": "Phase 11: Overfitting Diagnosis"},
    {"name": "tune",              "type": "dvc",  "desc": "Phase 12: Hyperparameter Optimization"},
    {"name": "improve",           "type": "dvc",  "desc": "Phase 13: Iterative Improvement"},
    {"name": "evaluate",          "type": "dvc",  "desc": "Phase 14: Final Model Evaluation"},
    # Python scripts (not DVC-managed)
    {"name": "track_mlflow",      "type": "python", "script": "src/models/track_mlflow.py",  "desc": "Phase 15: MLflow Tracking"},
    {"name": "serialize",         "type": "python", "script": "src/models/serialize.py",     "desc": "Phase 21: Model Serialization"},
    {"name": "register",          "type": "python", "script": "src/models/register.py",      "desc": "Phase 20: Model Registration"},
    {"name": "drift_monitor",     "type": "python", "script": "src/monitoring/drift.py",     "desc": "Phase 24: Drift Monitoring"},
    {"name": "quality_monitor",   "type": "python", "script": "src/monitoring/quality.py",   "desc": "Phase 24: Quality Monitoring"},
]

PYTHON = str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")
DVC = str(PROJECT_ROOT / ".venv" / "Scripts" / "dvc.exe")


def run_dvc_stage(stage_name: str, dry_run: bool = False) -> dict:
    """Run a single DVC stage via `dvc repro <stage>`."""
    cmd = [DVC, "repro", stage_name, "--allow-missing"]
    if dry_run:
        log.info("[DRY-RUN] Would run: %s", " ".join(cmd))
        return {"status": "dry_run", "stage": stage_name, "elapsed_s": 0.0}

    t0 = time.perf_counter()
    log.info("Running DVC stage: %s", stage_name)
    result = subprocess.run(
        cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True
    )
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        log.error("DVC stage '%s' FAILED:\n%s", stage_name, result.stderr[-2000:])
        return {
            "status": "FAILED",
            "stage": stage_name,
            "elapsed_s": round(elapsed, 2),
            "stderr": result.stderr[-500:],
        }
    log.info("DVC stage '%s' completed in %.1fs", stage_name, elapsed)
    return {
        "status": "SUCCESS",
        "stage": stage_name,
        "elapsed_s": round(elapsed, 2),
        "stdout": result.stdout[-200:],
    }


def run_python_script(script_path: str, dry_run: bool = False) -> dict:
    """Run a standalone Python script."""
    cmd = [PYTHON, script_path]
    if dry_run:
        log.info("[DRY-RUN] Would run: %s", " ".join(cmd))
        return {"status": "dry_run", "script": script_path, "elapsed_s": 0.0}

    t0 = time.perf_counter()
    log.info("Running script: %s", script_path)
    result = subprocess.run(
        cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True
    )
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        log.error("Script '%s' FAILED:\n%s", script_path, result.stderr[-2000:])
        return {
            "status": "FAILED",
            "script": script_path,
            "elapsed_s": round(elapsed, 2),
            "stderr": result.stderr[-500:],
        }
    log.info("Script '%s' completed in %.1fs", script_path, elapsed)
    return {
        "status": "SUCCESS",
        "script": script_path,
        "elapsed_s": round(elapsed, 2),
    }


def run_pipeline(
    stages: list[str] | None = None,
    dry_run: bool = False,
) -> dict:
    """Run the full or partial pipeline."""
    pipeline_start = time.perf_counter()
    all_names = [s["name"] for s in PIPELINE_STAGES]

    if stages:
        invalid = set(stages) - set(all_names)
        if invalid:
            raise ValueError(f"Unknown stages: {invalid}. Valid: {all_names}")
        selected = [s for s in PIPELINE_STAGES if s["name"] in stages]
    else:
        selected = PIPELINE_STAGES

    log.info("=" * 70)
    log.info("ChurnOps Pipeline Automation — %d stages selected", len(selected))
    if dry_run:
        log.info("MODE: DRY-RUN (no commands will be executed)")
    log.info("=" * 70)

    results = []
    failed = []

    for stage in selected:
        log.info("\n[%s] %s", stage["name"].upper(), stage["desc"])

        if stage["type"] == "dvc":
            result = run_dvc_stage(stage["name"], dry_run=dry_run)
        else:
            result = run_python_script(stage["script"], dry_run=dry_run)

        result["name"] = stage["name"]
        result["description"] = stage["desc"]
        results.append(result)

        if result["status"] == "FAILED":
            failed.append(stage["name"])
            log.error("Stage '%s' failed. Stopping pipeline.", stage["name"])
            break

    total_elapsed = time.perf_counter() - pipeline_start
    success_count = sum(1 for r in results if r["status"] == "SUCCESS")
    dry_count = sum(1 for r in results if r["status"] == "dry_run")

    manifest = {
        "pipeline_run_id": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "dry_run" if dry_run else "live",
        "stages_planned": len(selected),
        "stages_executed": len(results),
        "stages_succeeded": success_count,
        "stages_dry_run": dry_count,
        "stages_failed": len(failed),
        "failed_stages": failed,
        "total_elapsed_s": round(total_elapsed, 2),
        "overall_status": "FAILED" if failed else ("DRY_RUN" if dry_run else "SUCCESS"),
        "stage_results": results,
    }

    manifest_path = PROJECT_ROOT / "reports" / "pipeline_run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log.info("Pipeline manifest saved -> %s", manifest_path)

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="ChurnOps Pipeline Automation — run selected or all pipeline stages"
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        help="Specific stage names to run (default: all)",
        default=None,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print pipeline plan without executing",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available stages and exit",
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable pipeline stages:")
        for s in PIPELINE_STAGES:
            print(f"  {s['name']:<25} ({s['type']:<7}) — {s['desc']}")
        return

    manifest = run_pipeline(stages=args.stages, dry_run=args.dry_run)

    print("\n" + "=" * 80)
    print(f"PIPELINE {'[DRY-RUN] ' if args.dry_run else ''}COMPLETE — {manifest['overall_status']}")
    print("=" * 80)
    print(f"  Run ID           : {manifest['pipeline_run_id']}")
    print(f"  Stages executed  : {manifest['stages_executed']}/{manifest['stages_planned']}")
    print(f"  Succeeded        : {manifest['stages_succeeded']}")
    print(f"  Failed           : {manifest['stages_failed']}")
    print(f"  Total time       : {manifest['total_elapsed_s']:.1f}s")
    print(f"  Manifest         : reports/pipeline_run_manifest.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
