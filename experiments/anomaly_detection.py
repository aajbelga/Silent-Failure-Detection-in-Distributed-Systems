"""
experiments/anomaly_detection.py
================================
COMPONENT C -- unsupervised anomaly detection.

    python -m experiments.anomaly_detection

Requires ``data/processed/feature_dataset.csv``. Prefers
``data/processed/model_dataset.csv`` if Component B has already run, so that
anomalies can be cross-tabulated against the degradation bands.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import anomaly_detection, config
from src import feature_engineering as fe
from src.data_loader import load_processed
from src.utils import PipelineError, banner, info, save_json, section


def main() -> int:
    config.ensure_dirs()
    banner("COMPONENT C -- UNSUPERVISED ANOMALY DETECTION")

    which = "model" if config.MODEL_CSV.exists() else "feature"
    try:
        df = load_processed(which)
    except PipelineError as exc:
        print(f"\n{exc}\n")
        return 1
    info(f"loaded '{which}' dataset: {len(df):,} rows x {df.shape[1]} columns")

    feature_cols = fe.model_feature_columns(df)
    if not feature_cols:
        print("No usable features; cannot run anomaly detection.")
        return 1

    df, res = anomaly_detection.run(df, feature_cols)

    save_json({k: v for k, v in res.items() if k != "features"},
              config.METRICS_DIR / "anomaly_detection_summary.json")

    if "risk_band" in df.columns:
        section("Anomalies vs degradation bands")
        ct = pd.crosstab(df["risk_band"], df["anomaly_iforest"], normalize="index")
        print(ct.to_string(float_format=lambda v: f"{v:.4f}"))
        ct.to_csv(config.TABLE_DIR / "anomaly_by_risk_band.csv")
        print("\nThe degradation index and the anomaly detectors were built")
        print("independently, so agreement between them is evidence, not")
        print("circularity.")

    section("Summary")
    for name, d in res["detectors"].items():
        cc = d.get("cross_check") or {}
        print(f"{name:20s} {d.get('n_anomalies', 0):>8,} anomalies   "
              f"enrichment vs failure = {cc.get('failure_enrichment', 'n/a')}")
    print(f"{'consensus (>=2)':20s} "
          f"{res['consensus']['n_consensus_anomalies']:>8,} anomalies")

    keep = [c for c in ["machine_id", "collection_id", "cluster",
                        "anomaly_iforest", "anomaly_score_iforest", "anomaly_lof",
                        "anomaly_ocsvm", "anomaly_consensus", "risk_band",
                        "silent_failure_risk_score", config.TARGET]
            if c in df.columns]
    df[keep].to_csv(config.PRED_DIR / "anomaly_predictions.csv", index=False)
    df.to_csv(config.MODEL_CSV, index=False)
    info("anomaly flags merged into model_dataset.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
