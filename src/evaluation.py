"""
evaluation.py
=============
Metric computation and comparison tables.

WHY PR-AUC IS THE HEADLINE METRIC
---------------------------------
The target is imbalanced. A model that predicts "never fails" for everything
scores high accuracy and is useless. Accuracy is still reported (the brief asks
for it) but every ranking in this project is by PR-AUC, which is sensitive to
performance on the minority class and does not reward the trivial majority
predictor. Balanced accuracy and MCC are included as second opinions.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from . import config
from .utils import save_table, save_text


def evaluate(y_true, y_pred, y_proba=None) -> dict:
    """All requested metrics for one model, plus the confusion matrix cells."""
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 balanced_accuracy_score, confusion_matrix,
                                 f1_score, matthews_corrcoef, precision_score,
                                 recall_score, roc_auc_score)

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    m = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 6),
    }
    # roc_auc / pr_auc are undefined without probabilities or with a single
    # class present. We use float NaN (not None) as the "undefined metric"
    # sentinel: it keeps every value in this dict numeric, propagates correctly
    # through downstream arithmetic instead of raising, and is detected uniformly
    # by pandas .notna() and np.isnan().
    if y_proba is not None and len(np.unique(y_true)) > 1:
        m["roc_auc"] = round(float(roc_auc_score(y_true, y_proba)), 6)
        m["pr_auc"] = round(float(average_precision_score(y_true, y_proba)), 6)
    else:
        m["roc_auc"] = np.nan
        m["pr_auc"] = np.nan

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        m.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
        m["specificity"] = round(float(tn / (tn + fp)), 6) if (tn + fp) else np.nan
        # In failure prediction a false negative (missed failure) usually costs
        # more than a false positive (a needless check), so both are surfaced.
        m["missed_failures"] = int(fn)
        m["false_alarms"] = int(fp)
    m["positive_rate_true"] = round(float(y_true.mean()), 6)
    m["positive_rate_pred"] = round(float(y_pred.mean()), 6)
    return m


def cross_validate_best(model, X, y, groups, folds: int = None) -> dict:
    """
    Grouped cross-validation of one model, as a stability check on the single
    hold-out. Grouped, not plain KFold -- for the replication reason in models.py.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold

    folds = folds or config.CV_FOLDS
    gkf = GroupKFold(n_splits=folds)
    aucs, prs = [], []
    for tr, te in gkf.split(X, y, groups=groups):
        if len(np.unique(y[te])) < 2:
            continue
        from sklearn.base import clone
        m = clone(model)
        m.fit(X[tr], y[tr])
        p = (m.predict_proba(X[te])[:, 1] if hasattr(m, "predict_proba")
             else m.predict(X[te]).astype(float))
        aucs.append(roc_auc_score(y[te], p))
        prs.append(average_precision_score(y[te], p))
    if not aucs:
        return {"status": "no fold had both classes present"}
    return {"folds_used": len(aucs),
            "roc_auc_mean": round(float(np.mean(aucs)), 6),
            "roc_auc_std": round(float(np.std(aucs)), 6),
            "pr_auc_mean": round(float(np.mean(prs)), 6),
            "pr_auc_std": round(float(np.std(prs)), 6),
            "roc_auc_folds": [round(float(a), 6) for a in aucs]}


def write_evaluation_report(model_out: dict, cv: dict | None = None) -> None:
    buf = io.StringIO()

    def w(s: str = ""):
        buf.write(s + "\n")

    res = model_out.get("comparison", pd.DataFrame())

    w("=" * 78)
    w(" MODEL EVALUATION REPORT (STAGE 6)")
    w("=" * 78)
    w()
    w("TASK      : predict the EXPLICIT failure flag ('failed')")
    w(f"POS. RATE : {model_out.get('positive_rate', float('nan')):.6f}")
    w(f"FEATURES  : {len(model_out.get('feature_cols', []))} model-safe features")
    w()
    w("METRIC CHOICE")
    w("-" * 78)
    w("The target is imbalanced, so accuracy is not the ranking metric -- a model")
    w("that always predicts 'no failure' would score well and be worthless.")
    w("Rankings use PR-AUC (sensitive to the minority class); balanced accuracy")
    w("and MCC are reported as second opinions, and false negatives are surfaced")
    w("separately because a missed failure costs more than a false alarm.")
    w()
    if res.empty:
        w("No model produced results.")
    else:
        for proto in res.protocol.unique():
            sub = res[res.protocol == proto]
            w("=" * 78)
            w(f" PROTOCOL: {proto}"
              + ("   <-- HEADLINE" if proto == config.HONEST_PROTOCOL else
                 "   (reported to quantify leakage only)" if proto == "random" else ""))
            w("=" * 78)
            cols = [c for c in ["model", "accuracy", "precision", "recall", "f1",
                                "roc_auc", "pr_auc", "balanced_accuracy", "mcc",
                                "missed_failures", "false_alarms", "fit_seconds",
                                "subsampled"] if c in sub.columns]
            w(sub[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}",
                                  na_rep="n/a"))
            w()
        # confusion matrices for the honest protocol
        hon = res[res.protocol == config.HONEST_PROTOCOL]
        if not hon.empty and "tn" in hon.columns:
            w("=" * 78)
            w(f" CONFUSION MATRICES ({config.HONEST_PROTOCOL} protocol)")
            w("=" * 78)
            for _, r in hon.iterrows():
                if pd.isna(r.get("tn")):
                    continue
                w(f"{r['model']}")
                w(f"                 predicted 0   predicted 1")
                w(f"    actual 0     {int(r['tn']):>11,}   {int(r['fp']):>11,}")
                w(f"    actual 1     {int(r['fn']):>11,}   {int(r['tp']):>11,}")
                w(f"    -> {int(r['fn']):,} failures missed, "
                  f"{int(r['fp']):,} false alarms")
                w()
        # honest interpretation
        w("=" * 78)
        w(" INTERPRETATION")
        w("=" * 78)
        if not hon.empty and hon["pr_auc"].notna().any():
            b = hon.sort_values("pr_auc", ascending=False).iloc[0]
            base = model_out.get("positive_rate", 0)
            w(f"Best honest model : {b['model']}")
            w(f"  PR-AUC {b['pr_auc']:.4f} against a base rate of {base:.4f} "
              f"({b['pr_auc'] / max(base, 1e-9):.1f}x baseline)")
            w(f"  ROC-AUC {b['roc_auc']:.4f}, recall {b['recall']:.4f}, "
              f"precision {b['precision']:.4f}")
            w()
            if b["roc_auc"] and b["roc_auc"] >= 0.75:
                w("Resource-utilisation and machine-level behaviour carry substantial")
                w("information about explicit failure, even with all target")
                w("re-encodings removed and group leakage eliminated.")
            elif b["roc_auc"] and b["roc_auc"] >= 0.60:
                w("Resource behaviour carries real but MODERATE information about")
                w("explicit failure once leakage is removed. The honest ceiling on")
                w("this trace is much lower than the leaky numbers suggest, and that")
                w("gap is the main quantitative finding of the project.")
            else:
                w("Once leakage is removed, resource behaviour predicts explicit")
                w("failure only weakly. Insufficient evidence for a deployable")
                w("failure predictor from these features alone.")
        if "random" in res.protocol.unique() and not hon.empty:
            rnd = res[res.protocol == "random"]
            if rnd["roc_auc"].notna().any() and hon["roc_auc"].notna().any():
                w()
                w(f"For contrast, the naive random split reaches a best ROC-AUC of "
                  f"{rnd['roc_auc'].max():.4f} versus {hon['roc_auc'].max():.4f} "
                  f"honestly.")
                w("Reporting the first number would have been the single easiest way")
                w("to produce a fraudulent result in this project.")
    if cv:
        w()
        w("=" * 78)
        w(" GROUPED CROSS-VALIDATION (stability of the best honest model)")
        w("=" * 78)
        for k, v in cv.items():
            w(f"    {k}: {v}")
    w()
    save_text(buf.getvalue(), config.REPORT_DIR / "evaluation_report.txt")


def save_predictions(model_out: dict, df: pd.DataFrame, model_name: str) -> None:
    """Persist the honest-protocol test-set predictions for the dashboard."""
    if model_name not in model_out.get("predictions", {}):
        return
    y_true, y_pred, y_proba = model_out["predictions"][model_name]
    te = model_out["test_index"][model_name]
    cols = [c for c in ["machine_id", "collection_id", "cluster",
                        "silent_failure_risk_score", "risk_band"]
            if c in df.columns]
    out = df.iloc[te][cols].reset_index(drop=True)
    out["y_true"] = y_true
    out["y_pred"] = y_pred
    out["y_proba"] = np.round(y_proba, 6)
    out["model"] = model_name
    save_table(out, config.PRED_DIR / "test_predictions.csv")
