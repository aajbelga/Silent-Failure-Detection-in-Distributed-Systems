"""
run_project.py
==============
One command runs the entire study.

    python run_project.py                     # full dataset
    SFD_SAMPLE_ROWS=50000 python run_project.py   # fast smoke run
    python run_project.py --skip-heavy        # skip SVM/KNN and One-Class SVM
    python run_project.py --from feature      # reuse cached processed data

Ten stages, each printed as [n/10]. Every number this prints is computed from
the actual data -- nothing is hard-coded.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src import (anomaly_detection, config, data_loader, data_validation,
                 evaluation, explainability, models, preprocessing, reporting,
                 target_engineering, temporal_analysis, visualization)
from src import feature_engineering as fe
from src.utils import PipelineError, banner, info, save_json, timer, warn

N_STAGES = 10


def step(i: int, title: str) -> float:
    print()
    print("=" * 78)
    print(f"[{i}/{N_STAGES}] {title}")
    print("=" * 78)
    return time.perf_counter()


def done(t0: float, i: int) -> None:
    print(f"--- [{i}/{N_STAGES}] complete in {time.perf_counter() - t0:.1f}s")


def parse_args():
    p = argparse.ArgumentParser(description="Silent failure detection pipeline")
    p.add_argument("--from", dest="start_from", default="raw",
                   choices=["raw", "cleaned", "feature"],
                   help="reuse a cached processed dataset instead of starting over")
    p.add_argument("--skip-heavy", action="store_true",
                   help="shrink the quadratic-cost models (SVM, KNN, Gradient "
                        "Boosting, One-Class SVM) to tiny subsamples instead of "
                        "dropping them, so the comparison table stays complete")
    p.add_argument("--sample", type=int, default=None,
                   help="row subsample (overrides SFD_SAMPLE_ROWS)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample:
        config.SAMPLE_ROWS = args.sample
    if args.skip_heavy:
        config.HEAVY_MODELS = ("SVM (RBF)", "KNN", "Gradient Boosting")
        config.OCSVM_SAMPLE_SIZE = 5_000
        config.HEAVY_MODEL_SAMPLE = 4_000
        config.LOF_SAMPLE_SIZE = 20_000

    t_start = time.perf_counter()
    config.ensure_dirs()

    banner("SILENT FAILURE DETECTION IN DISTRIBUTED SYSTEMS\n"
           "Google Borg Cluster Traces -- full pipeline")
    print("Research question:")
    print("  Can resource-utilisation, machine-level and temporal behaviour in")
    print("  Google Borg cluster traces be used to identify patterns associated")
    print("  with silent system degradation before an explicit failure occurs?")
    print()
    print("Position on ground truth: the trace contains NO silent-failure label.")
    print("'failed' is an EXPLICIT failure flag. Silent degradation is therefore")
    print("modelled unsupervised and validated indirectly. Nothing is fabricated.")
    if config.SAMPLE_ROWS:
        print(f"\n*** SAMPLING ACTIVE: {config.SAMPLE_ROWS:,} rows. "
              f"Results are subsample estimates. ***")

    ctx: dict = {}
    profile = None

    # ---------------------------------------------------------------- 1
    t0 = step(1, "LOADING THE RAW DATASET")
    if args.start_from == "raw":
        data_loader.check_raw_exists()
        phys = data_loader.physical_line_count()
        df = data_loader.load_raw()
        data_loader.assert_minimum_schema(df)
        print(f"physical lines in file : {phys:,}")
        print(f"logical records parsed : {len(df):,}")
        print(f"columns                : {df.shape[1]}")
        print("(the gap is real: NumPy array reprs wrap across physical lines "
              "inside quoted fields)")
        ctx["physical_lines"] = phys
    else:
        which = "cleaned" if args.start_from == "cleaned" else "feature"
        df = data_loader.load_processed(which)
        print(f"reusing cached '{which}' dataset: {len(df):,} rows x {df.shape[1]} cols")
        ctx["physical_lines"] = None
    ctx["raw_rows"] = len(df)
    ctx["raw_cols"] = df.shape[1]
    done(t0, 1)

    # ---------------------------------------------------------------- 2
    t0 = step(2, "DATA VALIDATION, PROFILING AND LEAKAGE AUDIT")
    if args.start_from == "raw":
        profile = data_validation.profile_dataset(df, ctx.get("physical_lines"))
        ctx["profile"] = profile
        mv = profile.get("missing")
        ctx["raw_missing"] = mv
        leak = profile.get("leakage")
        if leak is not None and not leak.empty:
            flagged = leak[leak.get("flagged", False)] if "flagged" in leak else leak
            print(f"\nLEAKAGE AUDIT: {len(flagged)} column(s) flagged as "
                  f"target re-encodings or near-deterministic predictors.")
            cols = [c for c in ["column", "target_purity", "verdict"]
                    if c in leak.columns]
            print(leak[cols].head(10).to_string(index=False))
    else:
        print("skipped (cached dataset already validated in an earlier run)")
    done(t0, 2)

    # ---------------------------------------------------------------- 3
    t0 = step(3, "PREPROCESSING")
    if args.start_from == "raw":
        df, prep = preprocessing.clean(df, profile)
        preprocessing.write_report(prep)
        ctx["prep"] = prep
        print()
        print(f"Original rows      : {prep['original_rows']:,}")
        print(f"Rows removed       : {prep['rows_removed']:,}")
        print(f"Duplicate rows     : {prep['duplicate_rows_found']:,}")
        print(f"Missing values     : {prep['missing_values_before']:,} -> "
              f"{prep['missing_values_after']:,}")
        print(f"Rows after cleaning: {prep['final_rows']:,}")
        print(f"Columns before     : {prep['original_columns']}")
        print(f"Columns after      : {prep['final_columns']}")
        print(f"Memory             : {prep['memory_mb']:.0f} MB")
        with timer("write cleaned_dataset.csv"):
            df.to_csv(config.CLEANED_CSV, index=False)
    else:
        print("skipped (using cached data)")
    done(t0, 3)

    # ---------------------------------------------------------------- 4
    t0 = step(4, "FEATURE ENGINEERING")
    if args.start_from in ("raw", "cleaned"):
        df, registry = fe.build_features(df)
        registry.to_csv(config.TABLE_DIR / "feature_registry.csv", index=False)
        with timer("write feature_dataset.csv"):
            df.to_csv(config.FEATURE_CSV, index=False)
        ctx["registry"] = registry
        print(f"\nfeatures registered : {len(registry)}")
        print(registry.groupby("group").size().to_string())
    else:
        ctx["registry"] = (pd.read_csv(config.TABLE_DIR / "feature_registry.csv")
                           if (config.TABLE_DIR / "feature_registry.csv").exists()
                           else pd.DataFrame())
        print("skipped (using cached feature_dataset.csv)")

    feature_cols = fe.model_feature_columns(df)
    ctx["feature_cols"] = feature_cols
    print(f"model-safe features : {len(feature_cols)} "
          f"(after removing target re-encodings, identifiers and keys)")
    done(t0, 4)

    # ---------------------------------------------------------------- 5
    t0 = step(5, "SILENT-FAILURE DEGRADATION SCORING  [core research component]")
    df, machines, risk_meta = target_engineering.run(df)
    ctx["machines"] = machines
    ctx["risk_meta"] = risk_meta
    val = risk_meta.get("validation", {})
    print()
    print(f"components built           : {risk_meta['n_components']} of 6")
    print(f"weighting agreement (min r): {risk_meta.get('min_agreement')}")
    if "auc_score_vs_failed" in val:
        print(f"score-as-ranker ROC-AUC    : {val['auc_score_vs_failed']:.4f} "
              f"(base rate {val['base_rate']:.4f})")
        print(f"top/bottom decile lift     : {val.get('lift_top_vs_bottom')}")
    for k, v in risk_meta.get("band_counts", {}).items():
        print(f"  {k:14s} {v:>10,}")
    print()
    print(val.get("verdict", ""))
    done(t0, 5)

    # ---------------------------------------------------------------- 6
    t0 = step(6, "ANOMALY DETECTION")
    df, anom = anomaly_detection.run(df, feature_cols)
    ctx["anomaly"] = anom
    print()
    for name, d in anom["detectors"].items():
        cc = d.get("cross_check") or {}
        print(f"{name:20s} {d.get('n_anomalies', 0):>8,} anomalies  "
              f"failure enrichment = {cc.get('failure_enrichment', 'n/a')}")
    print(f"{'consensus (>=2)':20s} "
          f"{anom['consensus']['n_consensus_anomalies']:>8,} anomalies")
    done(t0, 6)

    # ---------------------------------------------------------------- 7
    t0 = step(7, "EXPLICIT-FAILURE MODELLING  [3 split protocols]")
    model_out = models.run(df, feature_cols)
    ctx["model_out"] = model_out
    ctx["model_comparison"] = model_out.get("comparison")
    res = model_out.get("comparison")
    if res is not None and not res.empty:
        show = [c for c in ["protocol", "model", "accuracy", "precision", "recall",
                            "f1", "roc_auc", "pr_auc"] if c in res.columns]
        print()
        print(res[show].to_string(index=False, na_rep="n/a"))
    best_name, best_model = models.best_model(model_out)
    ctx["best_model_name"] = best_name

    cv = None
    if best_model is not None:
        X = np.nan_to_num(df[feature_cols].to_numpy(dtype="float32"), nan=0.0,
                          posinf=0.0, neginf=0.0)
        y = df[config.TARGET].to_numpy(dtype="int8")
        groups = df["collection_id"].to_numpy()
        cv = evaluation.cross_validate_best(best_model, X, y, groups)
        print(f"\ngrouped {config.CV_FOLDS}-fold CV of {best_name}: {cv}")
        import joblib
        joblib.dump(best_model, config.MODEL_DIR / "best_explicit_failure_model.joblib")
        joblib.dump({"feature_cols": feature_cols, "protocol": config.HONEST_PROTOCOL},
                    config.MODEL_DIR / "model_metadata.joblib")
    ctx["cv"] = cv
    evaluation.write_evaluation_report(model_out, cv)
    if best_name:
        evaluation.save_predictions(model_out, df, best_name)
    done(t0, 7)

    # ---------------------------------------------------------------- 8
    t0 = step(8, "EXPLAINABILITY  (importance / permutation / SHAP)")
    expl = {}
    if best_model is not None:
        expl = explainability.run(model_out, df, best_name, best_model)
        if "permutation" in expl:
            print()
            print(expl["permutation"].head(10).to_string(index=False))
        print(f"\nSHAP status: {expl.get('shap_meta', {}).get('status')}")
    else:
        warn("no fitted model available; explainability skipped")
    ctx["explain"] = expl
    done(t0, 8)

    # ---------------------------------------------------------------- 9
    t0 = step(9, "TEMPORAL AND MACHINE-LEVEL ANALYSIS")
    temporal = temporal_analysis.run(df)
    ctx.update({"panel": temporal["panel"],
                "panel_trends": temporal["panel_trends"],
                "collection_trends": temporal["collection_trends"],
                "degradation_candidates": temporal["degradation_candidates"],
                "pre_failure": temporal["pre_failure"]})
    tp = temporal["panel_trends"]
    if tp is not None and not tp.empty:
        print()
        print(tp.head(10).to_string(index=False))
    pre = temporal["pre_failure"]
    if pre.get("conclusion"):
        print()
        print(pre["conclusion"])
    with timer("write model_dataset.csv"):
        df.to_csv(config.MODEL_CSV, index=False)
    done(t0, 9)

    # ---------------------------------------------------------------- 10
    t0 = step(10, "FIGURES, TABLES AND REPORTS")
    figs = visualization.make_all(df, ctx)
    ctx["figures"] = figs
    print(f"{len(figs)} figures written")

    ctx["runtime_seconds"] = round(time.perf_counter() - t_start, 1)
    reporting.write_project_report(df, ctx)
    reporting.write_final_results(df, ctx)
    save_json({"runtime_seconds": ctx["runtime_seconds"],
               "rows": len(df), "columns": df.shape[1],
               "n_features": len(feature_cols), "figures": len(figs),
               "best_model": best_name, "sampling": config.SAMPLE_ROWS},
              config.METRICS_DIR / "run_summary.json")
    done(t0, 10)

    banner("PIPELINE COMPLETE")
    print(f"total runtime : {ctx['runtime_seconds']:.1f}s")
    print(f"figures       : {len(figs)}  -> outputs/figures/")
    print(f"reports       : outputs/reports/  + PROJECT_REPORT.md")
    print(f"metrics       : outputs/metrics/")
    print(f"tables        : outputs/tables/")
    print()
    print("Read outputs/reports/final_results.txt for the headline numbers.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PipelineError as exc:
        print(f"\nPIPELINE ERROR\n{'-' * 78}\n{exc}\n")
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("\ninterrupted by user")
        raise SystemExit(130)
    except Exception:
        print("\nUNEXPECTED ERROR -- full traceback follows so it can be fixed:")
        traceback.print_exc()
        raise SystemExit(1)
