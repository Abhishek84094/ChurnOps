"""
Phase 16 — DVC Dataset Versioning
====================================
Configures DVC remote storage, adds all tracked data and model artifacts
to DVC version control, and generates a versioning manifest.

Designed to work with a local remote (for portability) that can be
swapped for S3/GCS in production via params.yaml.

Usage:
    python scripts/setup_dvc_versioning.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import get_logger

log = get_logger("dvc_versioning")

DVC = str(PROJECT_ROOT / ".venv" / "Scripts" / "dvc.exe")
REPORTS_DIR = PROJECT_ROOT / "reports"

# Files to track with DVC (in addition to dvc.yaml outs)
ARTIFACTS_TO_TRACK = [
    "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "data/interim/telco_clean.csv",
    "data/interim/telco_engineered.csv",
    "data/processed/train.csv",
    "data/processed/test.csv",
    "data/processed/cv_folds.json",
    "models/preprocessor.joblib",
    "models/tuned_xgboost.joblib",
    "models/champion_model.joblib",
    "models/champion_metadata.json",
    "models/model_card.json",
]


def run_cmd(cmd: list, description: str) -> dict:
    """Run a shell command and capture output."""
    log.info("Running: %s", description)
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    success = result.returncode == 0
    if not success:
        log.warning("Command failed (%s): %s", description, result.stderr[:300])
    else:
        log.info("Success: %s", description)
    return {
        "description": description,
        "success": success,
        "returncode": result.returncode,
        "stdout": result.stdout[:200],
        "stderr": result.stderr[:200] if not success else "",
    }


def setup_versioning() -> dict:
    """Configure DVC remote and add artifacts."""
    results = []

    # 1. Configure local DVC remote
    remote_path = str(PROJECT_ROOT / "dvc_remote_storage")
    Path(remote_path).mkdir(parents=True, exist_ok=True)

    r = run_cmd(
        [DVC, "remote", "add", "--default", "--force", "local_remote", remote_path],
        "Configure DVC local remote",
    )
    results.append(r)

    # 2. Add artifacts to DVC tracking (skip if already tracked)
    added = []
    skipped = []
    for artifact_rel in ARTIFACTS_TO_TRACK:
        artifact_path = PROJECT_ROOT / artifact_rel
        if not artifact_path.exists():
            log.info("Skipping (not found): %s", artifact_rel)
            skipped.append(artifact_rel)
            continue

        # Check if already tracked
        dvc_file = Path(str(artifact_path) + ".dvc")
        if dvc_file.exists():
            log.info("Already tracked: %s", artifact_rel)
            added.append({"artifact": artifact_rel, "status": "already_tracked"})
            continue

        r = run_cmd([DVC, "add", artifact_rel], f"DVC add: {artifact_rel}")
        results.append(r)
        added.append({"artifact": artifact_rel, "status": "added" if r["success"] else "failed"})

    # 3. Push to remote
    r = run_cmd([DVC, "push"], "DVC push to local remote")
    results.append(r)

    # 4. Generate .dvc/config documentation
    config_content = """[core]
    remote = local_remote
['remote "local_remote"']
    url = dvc_remote_storage
# ─── Production alternatives ─────────────────────────────
# AWS S3:     url = s3://your-bucket/churnops-dvc
# GCS:        url = gs://your-bucket/churnops-dvc
# Azure:      url = azure://your-container/churnops-dvc
# SSH remote: url = ssh://user@host/path/to/remote
"""
    dvc_config_path = PROJECT_ROOT / ".dvc" / "config"
    with open(dvc_config_path, "w", encoding="utf-8") as f:
        f.write(config_content)
    log.info("DVC config written -> %s", dvc_config_path)

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": 16,
        "dvc_remote": remote_path,
        "artifacts_tracked": len(added),
        "artifacts_skipped": len(skipped),
        "skipped": skipped,
        "tracked": added,
        "commands_run": len(results),
        "commands_successful": sum(1 for r in results if r["success"]),
        "push_status": "SUCCESS" if results and results[-1]["success"] else "PARTIAL",
        "notes": [
            "DVC pipeline stages tracked in dvc.yaml (15 stages)",
            "Raw data and model artifacts tracked as .dvc files",
            "Swap remote URL in .dvc/config for S3/GCS production deployment",
        ],
    }

    out_path = REPORTS_DIR / "dvc_versioning_manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    log.info("DVC versioning manifest saved -> %s", out_path)

    return manifest


def main():
    log.info("=" * 70)
    log.info("PHASE 16 — DVC DATASET VERSIONING")
    log.info("=" * 70)

    manifest = setup_versioning()

    print("\n" + "=" * 80)
    print("PHASE 16 — DVC VERSIONING COMPLETE")
    print("=" * 80)
    print(f"  DVC Remote       : {manifest['dvc_remote']}")
    print(f"  Artifacts tracked: {manifest['artifacts_tracked']}")
    print(f"  Artifacts skipped: {manifest['artifacts_skipped']}")
    print(f"  Push status      : {manifest['push_status']}")
    print("=" * 80)
    print("\nUseful DVC commands:")
    print("  dvc status          — check which stages are stale")
    print("  dvc repro           — re-run all stale stages")
    print("  dvc push            — push artifacts to remote")
    print("  dvc pull            — pull artifacts from remote")
    print("  dvc dag             — visualize pipeline DAG")


if __name__ == "__main__":
    main()
