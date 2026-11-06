"""
experiments/explicit_failure.py
===============================
COMPONENT A -- explicit failure prediction.

Standalone experiment: load the engineered dataset, train every classifier
under all three split protocols, evaluate, explain, and write the artefacts.

    python -m experiments.explicit_failure
    python experiments/explicit_failure.py

Requires ``data/processed/feature_dataset.csv`` (produced by run_project.py
stages 1-3). Fails with a clear message if it is absent rather than a traceback.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from src import config, evaluation, explainability
from src import feature_engineering as fe
from src import models as models_mod
from src.data_loader import load_processed
from src.utils import PipelineError, banner, info, save_json, section, warn


def main() -> int:
    config.ensure_dirs()
    banner("COMPONENT A -- EXPLICIT FAILURE PREDICTION")

    try:
        df = load_processed("feature")
    except PipelineError as exc:
        print(f"\n{exc}\n")
        return 1

    info(f"loaded engineered dataset: {len(df):,} rows x {df.shape[1]} columns")

    feature_cols = fe.model_feature_columns(df)
    info(f"{len(feature_cols)} model-safe features after the leakage whitelist")
    if not feature_cols:
        print("No model-safe features survived the whitelist; cannot train.")
        return 1

    out = models_mod.run(df, feature_cols)

    section("Cross-validation of the best honest model")
    name, model = models_mod.best_model(out)
    cv = None
    if model is not None:
        import numpy as np
        X = np.nan_to_num(df[feature_cols].to_numpy(dtype="float32"), nan=0.0,
                          posinf=0.0, neginf=0.0)
        y = df[config.TARGET].to_numpy(dtype="int8")
        groups = df["collection_id"].to_numpy()
        cv = evaluation.cross_validate_best(model, X, y, groups)
        info(f"grouped CV: {cv}")

    evaluation.write_evaluation_report(out, cv)
    if name:
        evaluation.save_predictions(out, df, name)

    section("Explainability")
    expl = {}
    if model is not None:
        expl = explainability.run(out, df, name, model)
        joblib.dump(model, config.MODEL_DIR / "best_explicit_failure_model.joblib")
        info(f"saved best model ({name}) to outputs/models/")

    save_json({"best_model": name, "cross_validation": cv,
               "splits": out["splits"], "n_features": len(feature_cols),
               "positive_rate": out["positive_rate"]},
              config.METRICS_DIR / "explicit_failure_summary.json")

    res = out.get("comparison")
    if res is not None and not res.empty:
        print()
        print(res.to_string(index=False))
    print(f"\nBest honest model: {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
