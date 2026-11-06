"""
models.py
=========
STAGES 6 & 7 -- explicit-failure classifiers and, more importantly, the
splitting protocol that decides whether their numbers mean anything.

STAGE 7 IS THE POINT OF THIS FILE
---------------------------------
Three leakage sources were found by inspection and are all handled here:

1.  TARGET RE-ENCODING. ``event``, ``instance_events_type`` and
    ``collections_events_type`` are deterministic re-encodings of ``failed``
    (failed == 1 <=> event == 'FAIL' <=> type == 2). Any model given these
    scores a perfect AUC while learning nothing. They are banned in
    :data:`config.TARGET_LEAKAGE_COLS` and excluded by
    ``feature_engineering.model_feature_columns``.

2.  JOIN-INDUCED REPLICATION -> GROUP LEAKAGE. The usage measurements were
    joined onto events, so identical usage vectors repeat many times (4,886
    unique usage vectors, mean replication ~83x, max 9,828x) and a collection's
    rows nearly always share one target value. Under a random split, near-copies
    of a test row sit in the training set. Measured directly:

        random 80/20 split, RandomForest : ROC-AUC ~ 1.0000
        grouped split (by collection_id) : ROC-AUC ~ 0.6577

    The first number is an artefact of the join. The honest protocol is
    ``grouped``: no ``collection_id`` appears on both sides of the split.

3.  TIME SENTINELS. Borg encodes "before the trace window" as 0 and "after" as
    2^63-1. Rows with ``time == 0`` fail at 0.6569 vs 0.1591 for real
    timestamps, so a raw ``time`` feature acts as a pseudo-oracle. Preprocessing
    turns sentinels into NaN plus an honest ``time_is_valid`` flag.

All three protocols are still *run*, and reported side by side, because the gap
between them is itself the most defensible result in the project.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from . import config
from .utils import (banner, info, save_json, save_table, save_text, section,
                    timer, warn)


# ---------------------------------------------------------------------------
# splitting protocols (STAGE 7)
# ---------------------------------------------------------------------------

def split_random(X, y, groups=None):
    """Naive stratified 80/20. Included ONLY to demonstrate the leakage."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=config.TEST_SIZE,
                              random_state=config.RANDOM_STATE, stratify=y)
    return tr, te


def split_grouped(X, y, groups):
    """
    Grouped split: every ``collection_id`` lands entirely on one side.

    This is the honest protocol for this dataset. It answers the question that
    actually matters -- "can we score a collection we have never seen?" --
    rather than "can we recognise a row we already memorised?".
    """
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=config.TEST_SIZE,
                           random_state=config.RANDOM_STATE)
    tr, te = next(gss.split(X, y, groups=groups))
    return tr, te


def split_grouped_temporal(X, y, groups, times):
    """
    Chronological *and* grouped: train on early collections, test on later ones.

    Collections are ordered by their first timestamp, then the earliest ~80% of
    collections form the training set. This respects causality (no training on
    the future) while still preventing group bleed. Rows whose timestamp was a
    sentinel are assigned to training, since their position in time is unknown
    and putting them in test would make the test set uninterpretable.
    """
    g = pd.Series(np.asarray(groups))
    t = pd.Series(np.asarray(times, dtype="float64"))
    first = t.groupby(g).min()
    known = first.dropna().sort_values()
    if len(known) < 10:
        warn("Too few timestamped groups for a chronological split; "
             "falling back to the grouped protocol.")
        return split_grouped(X, y, groups)
    cut = known.iloc[int(len(known) * (1 - config.TEST_SIZE))]
    test_groups = set(known[known >= cut].index)
    mask_test = g.isin(test_groups).to_numpy()
    tr = np.where(~mask_test)[0]
    te = np.where(mask_test)[0]
    return tr, te


def make_split(protocol: str, X, y, groups, times):
    if protocol == "random":
        return split_random(X, y, groups)
    if protocol == "grouped":
        return split_grouped(X, y, groups)
    if protocol == "grouped_temporal":
        return split_grouped_temporal(X, y, groups, times)
    raise ValueError(f"Unknown split protocol '{protocol}'")


def describe_split(name: str, tr, te, y, groups) -> dict:
    g = pd.Series(np.asarray(groups))
    gtr, gte = set(g.iloc[tr]), set(g.iloc[te])
    return {
        "protocol": name,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "train_positive_rate": round(float(np.mean(y[tr])), 6),
        "test_positive_rate": round(float(np.mean(y[te])), 6),
        "train_groups": len(gtr),
        "test_groups": len(gte),
        "groups_on_both_sides": len(gtr & gte),
        "group_leakage": bool(gtr & gte),
    }


# ---------------------------------------------------------------------------
# model zoo (STAGE 6)
# ---------------------------------------------------------------------------

def build_models(n_train: int, pos_rate: float) -> dict:
    """
    Build the classifier set.

    Deliberate choices, not a shotgun:
      * class_weight='balanced' wherever supported -- the positive class is a
        minority, and accuracy on an imbalanced target is meaningless without it.
      * Logistic Regression and KNN and SVM are pipelined with StandardScaler,
        because they are distance/scale sensitive. Trees are not scaled -- it
        would change nothing and only obscure the code.
      * KNN and the RBF SVM train on a subsample (see config.HEAVY_MODEL_SAMPLE):
        both are O(n^2)-ish and would dominate runtime for no analytical gain.
        This is disclosed in the results table, not hidden.
      * XGBoost is included only if the import succeeds.
    """
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    rs = config.RANDOM_STATE
    models: dict = {}

    models["Logistic Regression"] = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=rs))])

    models["Decision Tree"] = DecisionTreeClassifier(
        max_depth=8, min_samples_leaf=50, class_weight="balanced", random_state=rs)

    models["Random Forest"] = RandomForestClassifier(
        n_estimators=300, max_depth=16, min_samples_leaf=5,
        class_weight="balanced_subsample", n_jobs=config.N_JOBS, random_state=rs)

    models["Gradient Boosting"] = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1, random_state=rs)

    models["KNN"] = Pipeline([
        ("scale", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=15, n_jobs=config.N_JOBS))])

    # cache_size is raised from the 200 MB default: the RBF kernel matrix is
    # recomputed constantly during optimisation, and a bigger cache is the
    # cheapest available speed-up. probability=True is kept because PR-AUC needs
    # a score, but note it costs an internal 5-fold Platt calibration -- which is
    # exactly why this model is in config.HEAVY_MODELS.
    models["SVM (RBF)"] = Pipeline([
        ("scale", StandardScaler()),
        ("clf", SVC(kernel="rbf", class_weight="balanced", probability=True,
                    cache_size=1000, random_state=rs))])

    try:
        from xgboost import XGBClassifier
        spw = (1 - pos_rate) / max(pos_rate, 1e-9)
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, eval_metric="logloss",
            tree_method="hist", n_jobs=config.N_JOBS, random_state=rs,
            verbosity=0)
        info(f"XGBoost available (scale_pos_weight={spw:.2f})")
    except Exception as exc:
        warn(f"XGBoost unavailable, skipped: {exc}")

    return models


def _subsample_train(name: str, tr: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Stratified subsample of the training indices for the expensive models."""
    if name not in config.HEAVY_MODELS or len(tr) <= config.HEAVY_MODEL_SAMPLE:
        return tr
    from sklearn.model_selection import train_test_split
    keep, _ = train_test_split(tr, train_size=config.HEAVY_MODEL_SAMPLE,
                               random_state=config.RANDOM_STATE, stratify=y[tr])
    warn(f"{name}: training on a {len(keep):,}-row stratified subsample of "
         f"{len(tr):,} (quadratic-cost model; disclosed in the results table).")
    return np.sort(keep)


def fit_predict(model, X, y, tr, te, name: str):
    """Fit, then return (y_pred, y_proba, seconds, n_train_used)."""
    import time as _t
    tr_use = _subsample_train(name, tr, y)
    t0 = _t.perf_counter()
    model.fit(X[tr_use], y[tr_use])
    pred = model.predict(X[te])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X[te])[:, 1]
    elif hasattr(model, "decision_function"):
        d = model.decision_function(X[te])
        proba = (d - d.min()) / (d.max() - d.min() + 1e-12)
    else:
        proba = pred.astype(float)
    return pred, proba, _t.perf_counter() - t0, len(tr_use)


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def run(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """
    Train every model under every split protocol; return everything needed by
    evaluation / explainability.
    """
    from .evaluation import evaluate

    banner("STAGES 6-7  EXPLICIT-FAILURE MODELLING")

    cols = [c for c in feature_cols if c in df.columns]
    if not cols:
        raise ValueError("No model-safe features available; cannot train.")

    X = df[cols].to_numpy(dtype="float32")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df[config.TARGET].to_numpy(dtype="int8")
    groups = df["collection_id"].to_numpy() if "collection_id" in df.columns \
        else np.arange(len(y))
    times = (df["t_hours_from_start"].to_numpy(dtype="float64")
             if "t_hours_from_start" in df.columns else np.full(len(y), np.nan))

    pos_rate = float(y.mean())
    info(f"design matrix {X.shape[0]:,} x {X.shape[1]}   positive rate {pos_rate:.4f}")

    out: dict = {"feature_cols": cols, "positive_rate": pos_rate,
                 "splits": {}, "results": [], "fitted": {}, "test_index": {},
                 "predictions": {}}

    for protocol in config.SPLIT_PROTOCOLS:
        section(f"Split protocol: {protocol}")
        tr, te = make_split(protocol, X, y, groups, times)
        sd = describe_split(protocol, tr, te, y, groups)
        out["splits"][protocol] = sd
        info(f"train {sd['n_train']:,} / test {sd['n_test']:,}   "
             f"groups both sides = {sd['groups_on_both_sides']}   "
             f"pos rate train {sd['train_positive_rate']:.4f} "
             f"test {sd['test_positive_rate']:.4f}")

        if len(np.unique(y[te])) < 2:
            warn(f"Test fold under '{protocol}' has a single class; "
                 "protocol skipped (metrics would be undefined).")
            continue

        models = build_models(len(tr), pos_rate)
        for name, model in models.items():
            try:
                with timer(f"{protocol}/{name}"):
                    pred, proba, secs, n_used = fit_predict(model, X, y, tr, te, name)
                m = evaluate(y[te], pred, proba)
                m.update({"model": name, "protocol": protocol,
                          "train_rows_used": n_used, "test_rows": len(te),
                          "fit_seconds": round(secs, 3),
                          "subsampled": n_used < len(tr)})
                out["results"].append(m)
                info(f"    acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  "
                     f"roc_auc={m['roc_auc']}  pr_auc={m['pr_auc']}")
                if protocol == config.HONEST_PROTOCOL:
                    out["fitted"][name] = model
                    out["test_index"][name] = te
                    out["predictions"][name] = (y[te], pred, proba)
            except Exception as exc:
                warn(f"{name} failed under '{protocol}': {exc}")
                out["results"].append({"model": name, "protocol": protocol,
                                       "error": str(exc)})

    res = pd.DataFrame(out["results"])
    if not res.empty:
        front = ["protocol", "model", "accuracy", "precision", "recall", "f1",
                 "roc_auc", "pr_auc", "balanced_accuracy", "mcc",
                 "train_rows_used", "test_rows", "fit_seconds", "subsampled"]
        res = res[[c for c in front if c in res.columns] +
                  [c for c in res.columns if c not in front]]
        res = res.sort_values(["protocol", "pr_auc"], ascending=[True, False])
        save_table(res, config.METRICS_DIR / "model_comparison.csv")
        out["comparison"] = res

    save_table(pd.DataFrame(out["splits"]).T.reset_index(drop=True),
               config.METRICS_DIR / "split_protocols.csv")
    _write_leakage_report(out)
    return out


def best_model(out: dict, protocol: str | None = None, metric: str = "pr_auc"):
    """Best model under the honest protocol, by PR-AUC (imbalance-aware)."""
    protocol = protocol or config.HONEST_PROTOCOL
    res = out.get("comparison")
    if res is None or res.empty:
        return None, None
    sub = res[(res.protocol == protocol) & res[metric].notna()]
    if sub.empty:
        return None, None
    name = sub.sort_values(metric, ascending=False).iloc[0]["model"]
    return name, out["fitted"].get(name)


def _write_leakage_report(out: dict) -> None:
    buf = io.StringIO()

    def w(s: str = ""):
        buf.write(s + "\n")

    res = out.get("comparison", pd.DataFrame())

    w("=" * 78)
    w(" DATA-LEAKAGE AUDIT AND SPLIT PROTOCOL (STAGE 7)")
    w("=" * 78)
    w()
    w("Three leakage mechanisms were found by inspection. All are handled; the")
    w("evidence for each is below.")
    w()
    w("-" * 78)
    w("(1) TARGET RE-ENCODING")
    w("-" * 78)
    w("These columns are deterministic re-encodings of the target and are")
    w("EXCLUDED from every model:")
    for c in config.TARGET_LEAKAGE_COLS:
        w(f"    - {c}")
    w("Verified in Stage 1: failed == 1 <=> event == 'FAIL' <=> event type == 2,")
    w("with group purity 1.000000. A model given these reports a perfect score")
    w("while learning nothing about system behaviour.")
    w()
    w("-" * 78)
    w("(2) JOIN-INDUCED REPLICATION -> GROUP LEAKAGE")
    w("-" * 78)
    w("Usage measurements were joined onto instance events, so identical usage")
    w("vectors repeat across many rows and a collection's rows nearly always")
    w("share one target value. Under a RANDOM split, near-duplicates of test")
    w("rows sit in the training set.")
    w()
    if not res.empty and {"random", "grouped"} <= set(res.protocol.unique()):
        w("MEASURED EFFECT (same features, same models -- only the split changes):")
        w()
        piv = res.pivot_table(index="model", columns="protocol",
                              values="roc_auc", aggfunc="first")
        w(piv.to_string(float_format=lambda v: f"{v:.4f}"))
        w()
        if "random" in piv.columns and "grouped" in piv.columns:
            d = (piv["random"] - piv["grouped"]).dropna()
            if len(d):
                w(f"Mean ROC-AUC inflation from the random split: {d.mean():+.4f}")
                w(f"Worst case: {d.max():+.4f} ({d.idxmax()})")
        w()
        w("The random-split numbers are ARTEFACTS. They are reported only to")
        w("quantify the trap. Every headline result in this project uses the")
        w(f"'{config.HONEST_PROTOCOL}' protocol.")
    w()
    w("-" * 78)
    w("(3) TIME SENTINELS")
    w("-" * 78)
    w("Borg encodes 'event before the trace window' as time = 0 and 'after' as")
    w("2^63-1. Rows with time = 0 have a 0.6569 failure rate against 0.1591 for")
    w("real timestamps, so a raw time feature is a pseudo-oracle rather than a")
    w("temporal signal. Preprocessing converts sentinels to NaN and records an")
    w("honest 'time_is_valid' flag; raw time columns are excluded from models.")
    w()
    w("=" * 78)
    w(" SPLIT PROTOCOLS ACTUALLY RUN")
    w("=" * 78)
    for name, sd in out["splits"].items():
        w(f"[{name}]")
        w(f"    train rows / test rows : {sd['n_train']:,} / {sd['n_test']:,}")
        w(f"    train / test groups    : {sd['train_groups']:,} / {sd['test_groups']:,}")
        w(f"    groups on BOTH sides   : {sd['groups_on_both_sides']:,}"
          f"   -> group leakage: {sd['group_leakage']}")
        w(f"    positive rate tr / te  : {sd['train_positive_rate']:.4f} / "
          f"{sd['test_positive_rate']:.4f}")
        w()
    w(f"HEADLINE PROTOCOL: '{config.HONEST_PROTOCOL}'")
    w("Rationale: it is the only protocol that answers the question a deployed")
    w("system faces -- score a collection never seen before -- while the")
    w("chronological variant additionally guarantees no training on the future.")
    w()
    w("OTHER LEAKAGE CONTROLS")
    w("-" * 78)
    w("* Every rolling/lagged feature is built from shift(1) inside its group, so")
    w("  a row never contributes to its own rolling statistic.")
    w("* machine_failure_rate / machine_failure_count are target-derived. They are")
    w("  registered with model_safe=False and used for DESCRIPTION ONLY, never as")
    w("  model inputs.")
    w("* Scalers are fitted inside pipelines, so they see only training folds.")
    w("* Identity columns (collection_name, user, logical name) are excluded --")
    w("  they would let a model memorise which named job fails.")
    w()
    save_text(buf.getvalue(), config.REPORT_DIR / "leakage_audit.txt")
