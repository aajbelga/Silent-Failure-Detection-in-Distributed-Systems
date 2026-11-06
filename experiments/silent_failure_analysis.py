"""
experiments/silent_failure_analysis.py
======================================
COMPONENT B -- silent-failure / degradation analysis.

Standalone experiment: build the degradation index, test it against the
independent explicit-failure signal, profile machines, and run the temporal
degradation analysis.

    python -m experiments.silent_failure_analysis

Requires ``data/processed/feature_dataset.csv``.

Reminder of the scientific position, restated here because this is the file a
reader is most likely to open first: **there is no silent-failure ground truth
in this trace.** Nothing below claims to detect silent failures. What is built
is a transparent degradation index plus an honest measurement of how much
information it carries.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config, target_engineering, temporal_analysis
from src.data_loader import load_processed
from src.utils import PipelineError, banner, info, save_json, section


def main() -> int:
    config.ensure_dirs()
    banner("COMPONENT B -- SILENT-FAILURE / DEGRADATION ANALYSIS")
    print("NOTE: this trace has NO silent-failure ground truth. What follows is a")
    print("      transparent degradation index and an honest test of its value.\n")

    try:
        df = load_processed("feature")
    except PipelineError as exc:
        print(f"\n{exc}\n")
        return 1

    info(f"loaded {len(df):,} rows x {df.shape[1]} columns")

    df, machines, meta = target_engineering.run(df)
    temporal = temporal_analysis.run(df)

    save_json({k: v for k, v in meta.items() if k != "provenance"},
              config.METRICS_DIR / "silent_failure_summary.json")

    section("Headline numbers")
    val = meta.get("validation", {})
    print(f"components built            : {meta['n_components']} of 6")
    print(f"weighting-scheme agreement  : min Spearman rho = {meta.get('min_agreement')}")
    if "auc_score_vs_failed" in val:
        print(f"score-as-ranker ROC-AUC     : {val['auc_score_vs_failed']:.4f} "
              f"(base rate {val['base_rate']:.4f})")
        print(f"top/bottom decile lift      : {val.get('lift_top_vs_bottom')}")
    print(f"machines profiled           : {len(machines):,}")
    cands = temporal.get("degradation_candidates")
    if cands is not None and not cands.empty:
        print(f"degradation candidates      : {len(cands):,} collections with "
              f">=2 worsening indicators")
    print()
    print(val.get("verdict", ""))
    pre = temporal.get("pre_failure", {})
    if pre.get("conclusion"):
        print()
        print(pre["conclusion"])

    # keep the scored dataset so the dashboard and Component C can reuse it
    df.to_csv(config.MODEL_CSV, index=False)
    info(f"scored dataset -> {config.MODEL_CSV.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
