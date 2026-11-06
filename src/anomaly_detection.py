"""
anomaly_detection.py
=====================
STAGE 5 -- unsupervised anomaly detection.

Three complementary detectors, chosen because they disagree in useful ways:

  * Isolation Forest  -- global, tree-based, cheap; run on the FULL dataset.
  * Local Outlier Factor (LOF) -- density-based, catches *local* anomalies a
    global method misses; O(n^2)-ish, so run on a reproducible subsample.
  * One-Class SVM     -- boundary-based; the most expensive, run on a smaller
    subsample and only if it converges in reasonable time.

The detectors are unsupervised: they never see ``failed``. After scoring we
*compare* the anomaly flags against ``failed`` purely as an external sanity
check -- not as training signal and not as a performance claim.

Nothing here invents a "true" contamination rate. ``ANOMALY_CONTAMINATION`` is a
declared assumption (5%), stated as such in the report; the ranking of points by
anomaly score is the durable output, and the threshold is secondary.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from . import config
from .utils import (banner, fmt, info, save_table, save_text, section, timer,
                    warn)


# ---------------------------------------------------------------------------
# feature matrix
# ---------------------------------------------------------------------------

def _matrix(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    """Standardised, finite feature matrix for the detectors."""
    from sklearn.preprocessing import StandardScaler

    cols = [c for c in feature_cols if c in df.columns]
    X = df[cols].to_numpy(dtype="float64")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Xs = StandardScaler().fit_transform(X)
    return Xs, cols


# ---------------------------------------------------------------------------
# detectors
# ---------------------------------------------------------------------------

def run_isolation_forest(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full-data Isolation Forest. Returns (labels{1=anomaly}, score)."""
    from sklearn.ensemble import IsolationForest

    with timer("IsolationForest.fit_predict"):
        clf = IsolationForest(
            n_estimators=config.IFOREST_ESTIMATORS,
            contamination=config.ANOMALY_CONTAMINATION,
            random_state=config.RANDOM_STATE,
            n_jobs=config.N_JOBS,
        )
        raw = clf.fit_predict(X)                     # -1 anomaly, +1 normal
    labels = (raw == -1).astype(int)
    # Higher = more anomalous. score_samples is higher for normal, so negate.
    score = -clf.score_samples(X)
    return labels, score


def run_lof(X: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    LOF on a subsample (novelty=False => fit_predict only on the sample).

    LOF is density-based and O(n * k) in the neighbour search; on ~400k rows
    with the default tree it is slow and memory-hungry, so we subsample. Points
    outside the sample get NaN, which is honest -- LOF simply did not evaluate
    them -- rather than a fabricated 0.
    """
    from sklearn.neighbors import LocalOutlierFactor

    labels = np.full(X.shape[0], np.nan)
    score = np.full(X.shape[0], np.nan)
    if X.shape[0] > n:
        rng = np.random.RandomState(config.RANDOM_STATE)
        idx = rng.choice(X.shape[0], size=n, replace=False)
        warn(f"LOF on a {n:,}-row subsample of {X.shape[0]:,} "
             "(density methods do not scale to the full trace).")
    else:
        idx = np.arange(X.shape[0])

    with timer("LOF.fit_predict"):
        lof = LocalOutlierFactor(n_neighbors=config.LOF_NEIGHBORS,
                                 contamination=config.ANOMALY_CONTAMINATION,
                                 n_jobs=config.N_JOBS)
        raw = lof.fit_predict(X[idx])
    labels[idx] = (raw == -1).astype(int)
    score[idx] = -lof.negative_outlier_factor_
    return labels, score


def run_ocsvm(X: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """One-Class SVM on a small subsample; skipped gracefully if too slow."""
    from sklearn.svm import OneClassSVM

    labels = np.full(X.shape[0], np.nan)
    score = np.full(X.shape[0], np.nan)
    rng = np.random.RandomState(config.RANDOM_STATE)
    idx = (rng.choice(X.shape[0], size=n, replace=False)
           if X.shape[0] > n else np.arange(X.shape[0]))
    try:
        with timer("OneClassSVM.fit_predict"):
            svm = OneClassSVM(nu=config.ANOMALY_CONTAMINATION, kernel="rbf",
                              gamma="scale")
            raw = svm.fit_predict(X[idx])
        labels[idx] = (raw == -1).astype(int)
        score[idx] = -svm.score_samples(X[idx])
        if X.shape[0] > n:
            warn(f"One-Class SVM on a {n:,}-row subsample (RBF SVM is O(n^2)).")
    except Exception as exc:                                    # pragma: no cover
        warn(f"One-Class SVM skipped ({exc}).")
        return np.full(X.shape[0], np.nan), np.full(X.shape[0], np.nan)
    return labels, score


# ---------------------------------------------------------------------------
# cross-check vs explicit failure (sanity only)
# ---------------------------------------------------------------------------

def _cross_check(df: pd.DataFrame, flag: pd.Series) -> dict:
    if config.TARGET not in df.columns:
        return {}
    y = df[config.TARGET].astype(int)
    m = flag.notna()
    if m.sum() == 0:
        return {}
    f = flag[m].astype(int)
    yy = y[m]
    both = int(((f == 1) & (yy == 1)).sum())
    anom = int((f == 1).sum())
    fails = int((yy == 1).sum())
    out = {
        "anomalies_flagged": anom,
        "anomaly_rate": round(anom / len(f), 6),
        "failures_in_window": fails,
        "anomalies_that_failed": both,
        "precision_wrt_failed": round(both / anom, 6) if anom else None,
        "recall_wrt_failed": round(both / fails, 6) if fails else None,
        "failure_rate_in_anomalies": round(both / anom, 6) if anom else None,
        "failure_rate_in_normals": round(
            int(((f == 0) & (yy == 1)).sum()) / max(int((f == 0).sum()), 1), 6),
    }
    if out["failure_rate_in_normals"]:
        out["failure_enrichment"] = round(
            out["failure_rate_in_anomalies"] / out["failure_rate_in_normals"], 4)
    return out


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def run(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    banner("STAGE 5  ANOMALY DETECTION")
    Xs, used = _matrix(df, feature_cols)
    info(f"anomaly feature matrix: {Xs.shape[0]:,} x {Xs.shape[1]} "
         f"(contamination assumption = {config.ANOMALY_CONTAMINATION})")

    results: dict = {"contamination": config.ANOMALY_CONTAMINATION,
                     "n_features": len(used), "features": used, "detectors": {}}

    section("Isolation Forest (full data)")
    lab_if, sc_if = run_isolation_forest(Xs)
    df["anomaly_iforest"] = lab_if
    df["anomaly_score_iforest"] = sc_if
    results["detectors"]["isolation_forest"] = {
        "scope": "full", "n": int(len(lab_if)),
        "n_anomalies": int(lab_if.sum()),
        "cross_check": _cross_check(df, pd.Series(lab_if, index=df.index))}

    section("Local Outlier Factor (subsampled)")
    lab_lof, sc_lof = run_lof(Xs, config.LOF_SAMPLE_SIZE)
    df["anomaly_lof"] = lab_lof
    df["anomaly_score_lof"] = sc_lof
    results["detectors"]["lof"] = {
        "scope": f"subsample<={config.LOF_SAMPLE_SIZE}",
        "n_scored": int(np.isfinite(lab_lof).sum()),
        "n_anomalies": int(np.nansum(lab_lof)),
        "cross_check": _cross_check(df, pd.Series(lab_lof, index=df.index))}

    section("One-Class SVM (subsampled)")
    lab_svm, sc_svm = run_ocsvm(Xs, config.OCSVM_SAMPLE_SIZE)
    if np.isfinite(lab_svm).any():
        df["anomaly_ocsvm"] = lab_svm
        df["anomaly_score_ocsvm"] = sc_svm
        results["detectors"]["ocsvm"] = {
            "scope": f"subsample<={config.OCSVM_SAMPLE_SIZE}",
            "n_scored": int(np.isfinite(lab_svm).sum()),
            "n_anomalies": int(np.nansum(lab_svm)),
            "cross_check": _cross_check(df, pd.Series(lab_svm, index=df.index))}

    # consensus among detectors that scored a given row
    section("Consensus")
    flags = [c for c in ["anomaly_iforest", "anomaly_lof", "anomaly_ocsvm"]
             if c in df.columns]
    votes = df[flags]
    df["anomaly_votes"] = votes.sum(axis=1, min_count=1)
    df["anomaly_detectors_scoring"] = votes.notna().sum(axis=1)
    df["anomaly_consensus"] = (
        (df["anomaly_votes"] >= 2) & (df["anomaly_detectors_scoring"] >= 2)
    ).astype(int)
    results["consensus"] = {
        "n_consensus_anomalies": int(df["anomaly_consensus"].sum()),
        "cross_check": _cross_check(df, df["anomaly_consensus"])}

    # agreement between IF and LOF where both scored
    both = df.dropna(subset=["anomaly_iforest", "anomaly_lof"])
    if len(both):
        from sklearn.metrics import cohen_kappa_score
        try:
            results["iforest_lof_cohen_kappa"] = round(float(cohen_kappa_score(
                both["anomaly_iforest"].astype(int),
                both["anomaly_lof"].astype(int))), 6)
        except Exception:
            pass

    _write_report(results, df)
    _write_summary_table(results)
    return df, results


def _write_summary_table(results: dict) -> None:
    rows = []
    for name, d in results["detectors"].items():
        cc = d.get("cross_check", {}) or {}
        rows.append({
            "detector": name,
            "scope": d.get("scope"),
            "n_scored": d.get("n") or d.get("n_scored"),
            "n_anomalies": d.get("n_anomalies"),
            "anomaly_rate": cc.get("anomaly_rate"),
            "failure_rate_in_anomalies": cc.get("failure_rate_in_anomalies"),
            "failure_rate_in_normals": cc.get("failure_rate_in_normals"),
            "failure_enrichment": cc.get("failure_enrichment"),
        })
    cc = results["consensus"]["cross_check"] or {}
    rows.append({"detector": "consensus(>=2)", "scope": "where>=2 scored",
                 "n_scored": None,
                 "n_anomalies": results["consensus"]["n_consensus_anomalies"],
                 "anomaly_rate": cc.get("anomaly_rate"),
                 "failure_rate_in_anomalies": cc.get("failure_rate_in_anomalies"),
                 "failure_rate_in_normals": cc.get("failure_rate_in_normals"),
                 "failure_enrichment": cc.get("failure_enrichment")})
    save_table(pd.DataFrame(rows), config.METRICS_DIR / "anomaly_summary.csv")


def _write_report(results: dict, df: pd.DataFrame) -> None:
    buf = io.StringIO()

    def w(s: str = ""):
        buf.write(s + "\n")

    w("=" * 78)
    w(" ANOMALY DETECTION REPORT (STAGE 5)")
    w("=" * 78)
    w()
    w(f"Feature matrix : {results['n_features']} standardised features")
    w(f"Contamination  : {results['contamination']} (a DECLARED ASSUMPTION, not")
    w("                 an estimated truth -- see note below)")
    w()
    w("METHOD NOTES")
    w("-" * 78)
    w("* Isolation Forest runs on the FULL dataset.")
    w("* LOF and One-Class SVM run on reproducible subsamples because their")
    w("  cost grows too fast for ~400k rows. Rows they did not evaluate are")
    w("  left as NaN, never as a fabricated 'normal'.")
    w("* The detectors are UNSUPERVISED -- they never see 'failed'. The")
    w("  comparison against 'failed' below is an external sanity check only.")
    w("* The 5% contamination is an assumption. The durable output is the")
    w("  RANKING of points by anomaly score; the cut-off is secondary and can")
    w("  be moved without re-training.")
    w()
    for name, d in results["detectors"].items():
        w("=" * 78)
        w(f" {name.upper()}")
        w("=" * 78)
        w(f"scope           : {d.get('scope')}")
        w(f"points scored   : {d.get('n') or d.get('n_scored'):,}")
        w(f"anomalies found : {d.get('n_anomalies'):,}")
        cc = d.get("cross_check") or {}
        if cc:
            w(f"anomaly rate                 : {fmt(cc.get('anomaly_rate'))}")
            w(f"failure rate WITHIN anomalies: {fmt(cc.get('failure_rate_in_anomalies'))}")
            w(f"failure rate within normals  : {fmt(cc.get('failure_rate_in_normals'))}")
            if cc.get("failure_enrichment"):
                w(f"enrichment (anom / normal)   : {cc['failure_enrichment']:.3f}x")
        w()
    if "iforest_lof_cohen_kappa" in results:
        w(f"IsolationForest vs LOF agreement (Cohen's kappa, shared rows): "
          f"{results['iforest_lof_cohen_kappa']:+.4f}")
        w("(Low kappa is expected and useful: the two methods detect different")
        w(" kinds of anomaly, so their union covers more failure modes.)")
        w()
    w("=" * 78)
    w(" CONSENSUS")
    w("=" * 78)
    w(f"Rows flagged by >= 2 detectors (of those scoring the row): "
      f"{results['consensus']['n_consensus_anomalies']:,}")
    cc = results["consensus"]["cross_check"] or {}
    if cc.get("failure_enrichment"):
        w(f"Failure enrichment of consensus anomalies: {cc['failure_enrichment']:.3f}x")
    w()
    w("INTERPRETATION")
    w("-" * 78)
    enr = [d.get("cross_check", {}).get("failure_enrichment")
           for d in results["detectors"].values()]
    enr = [e for e in enr if e]
    if enr and max(enr) > 1.5:
        w("Anomalous rows fail explicitly more often than normal rows. Since the")
        w("detectors never saw 'failed', this is independent evidence that the")
        w("resource-behaviour anomalies overlap with real trouble -- supporting")
        w("the use of anomaly detection as a silent-degradation screen.")
    else:
        w("Anomalous rows are NOT strongly enriched for explicit failure on this")
        w("trace. That is stated plainly: the anomalies capture unusual resource")
        w("behaviour, but there is INSUFFICIENT EVIDENCE that they coincide with")
        w("explicit failures here. They remain candidates for investigation.")
    w()
    save_text(buf.getvalue(), config.REPORT_DIR / "anomaly_report.txt")
