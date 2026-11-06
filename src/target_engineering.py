"""
target_engineering.py
=====================
STAGE 4 -- the silent-failure / degradation research component.

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not treat ``failed`` as a silent-failure label. ``failed`` is an
*explicit* failure indicator: inspection proved it is exactly equivalent to
``event == "FAIL"``. An explicitly reported failure is by definition not silent.

**This trace contains no ground truth for silent failure.** Nothing in the Borg
schema records "this machine was degrading but nobody noticed". Consequently
this module builds an *unsupervised, transparent, auditable degradation index*
and then tests whether it carries information -- rather than claiming to detect
a phenomenon it cannot label.

THE SCORE
---------
Six degradation indicators, each a documented symptom of stress that is *not
itself an explicit failure*:

  1. cpu_stall_pressure       -- CPI elevated vs cluster peers
  2. memory_pressure          -- resident memory near the assigned ceiling
  3. resource_variability     -- unstable / bursty CPU demand
  4. tail_latency_pressure    -- bad tail hidden behind a healthy mean
  5. telemetry_incompleteness -- monitoring itself losing samples
  6. provisioning_mismatch    -- requested vs actually-used divergence

Each is rank-normalised to [0, 1] (scale-free, outlier-proof, no unit
conversion invented), then combined.

ON WEIGHTS
----------
The brief forbids inventing weights, so no weight is invented. The composite is
computed three independent ways --

  * ``equal``       : principle of indifference (no evidence justifies unequal weights)
  * ``pca``         : weights are the first-principal-component loadings, i.e.
                      learned from the data's own covariance structure
  * ``mahalanobis`` : multivariate outlyingness, weight-free by construction

-- and their Spearman rank agreement is reported. If the three rankings agree,
the conclusion is robust to the weighting choice, which is a stronger claim than
any single defensible weight vector.

VALIDATION
----------
None of the six components uses ``failed``. That independence is what makes
comparing the score against ``failed`` a genuine external test rather than a
tautology. :func:`validate_risk_score` performs that test and reports the
result honestly -- including when the result is weak.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .utils import (fmt, info, rank_normalise, robust_z, safe_div, save_table,
                    save_text, section, slope, warn)


# ---------------------------------------------------------------------------
# 1. Build the six components
# ---------------------------------------------------------------------------

def build_risk_components(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Assemble the component matrix at row level.

    Returns ``(components_df, provenance)`` where provenance records which
    source columns each component was built from, so the score is fully
    traceable.
    """
    comp = pd.DataFrame(index=df.index)
    prov: dict[str, dict] = {}

    def add(name: str, series, sources: list[str], direction: str, note: str):
        s = pd.Series(np.asarray(series, dtype="float64"), index=df.index)
        comp[name] = rank_normalise(s)
        prov[name] = {"sources": sources, "direction": direction, "note": note,
                      "raw_mean": float(np.nanmean(s)) if s.notna().any() else None,
                      "raw_median": float(np.nanmedian(s)) if s.notna().any() else None}

    # -- (1) CPU stall pressure -------------------------------------------
    if "cpi_vs_cluster" in df.columns:
        add("cpu_stall_pressure", df["cpi_vs_cluster"], ["cpi_vs_cluster"],
            "higher = worse",
            "CPI normalised by the cluster median. Comparing inside a cell "
            "controls for hardware-generation differences, so what remains is "
            "local inefficiency: the same instruction stream consuming more "
            "cycles. This is the most direct available proxy for a machine "
            "doing its work worse without reporting an error.")
    elif "cpi" in df.columns:
        add("cpu_stall_pressure", df["cpi"], ["cpi"], "higher = worse",
            "Raw CPI (cluster normalisation unavailable).")
    else:
        warn("No CPI column -- cpu_stall_pressure component omitted.")

    # -- (2) Memory pressure ----------------------------------------------
    if "memory_pressure" in df.columns:
        base = df["memory_pressure"].astype("float64")
        if "cache_ratio" in df.columns:
            # Cache collapse amplifies pressure: the kernel evicts page cache
            # before it OOM-kills, so a shrinking cache under load is an early
            # warning. We combine occupancy with cache scarcity.
            cache_scarcity = 1.0 / (1.0 + df["cache_ratio"].astype("float64"))
            mp = rank_normalise(base) * 0.5 + rank_normalise(cache_scarcity) * 0.5
            add("memory_pressure", mp, ["memory_pressure", "cache_ratio"],
                "higher = worse",
                "Equal blend of (a) resident memory as a share of the assigned "
                "allocation and (b) page-cache scarcity. Both are rank-"
                "normalised first; the 50/50 split is the indifference default, "
                "since no evidence favours one symptom over the other.")
        else:
            add("memory_pressure", base, ["memory_pressure"], "higher = worse",
                "Resident memory as a share of the assigned allocation.")
    else:
        warn("No memory_pressure column -- component omitted.")

    # -- (3) Resource variability -----------------------------------------
    var_srcs = [c for c in ["cpu_dist_cv", "cpu_burstiness", "machine_cpu_cv"]
                if c in df.columns]
    if var_srcs:
        v = sum(rank_normalise(df[c].astype("float64")) for c in var_srcs) / len(var_srcs)
        add("resource_variability", v, var_srcs, "higher = worse",
            "Mean of the available scale-free instability measures: "
            "within-window CPU coefficient of variation, peak-to-mean "
            "burstiness, and the machine's own CPU variability. Averaging "
            "several views of the same construct reduces the influence of any "
            "one noisy estimator.")
    else:
        warn("No variability columns -- component omitted.")

    # -- (4) Tail latency pressure ----------------------------------------
    if "cpu_tail_ratio" in df.columns:
        add("tail_latency_pressure", df["cpu_tail_ratio"], ["cpu_tail_ratio"],
            "higher = worse",
            "Tail-percentile CPU divided by mean CPU. This is the archetypal "
            "silent-degradation signature: mean-based dashboards stay green "
            "while the tail -- the part users actually experience -- deteriorates.")
    elif "cpu_tail_gap" in df.columns:
        add("tail_latency_pressure", df["cpu_tail_gap"], ["cpu_tail_gap"],
            "higher = worse", "Absolute gap between tail and mean CPU usage.")
    else:
        warn("No tail CPU columns -- component omitted.")

    # -- (5) Telemetry incompleteness -------------------------------------
    if "telemetry_loss" in df.columns:
        add("telemetry_incompleteness", df["telemetry_loss"], ["telemetry_loss"],
            "higher = worse",
            "Fraction of expected monitoring samples that never arrived "
            "(1 - sample_rate). Included because a monitoring pipeline that "
            "quietly stops collecting is itself a silent failure -- and it is "
            "the mechanism by which other silent failures stay invisible.")
    else:
        warn("No sample_rate column -- component omitted.")

    # -- (6) Provisioning mismatch ----------------------------------------
    if "provisioning_mismatch" in df.columns:
        add("provisioning_mismatch", df["provisioning_mismatch"],
            ["cpu_utilization_ratio", "mem_utilization_ratio"], "higher = worse",
            "Combined absolute divergence of CPU and memory usage from their "
            "requested amounts. Captures both starvation (usage pinned below a "
            "large request) and runaway growth (usage exceeding the request).")
    elif "cpu_utilization_ratio" in df.columns:
        add("provisioning_mismatch", (1 - df["cpu_utilization_ratio"]).abs(),
            ["cpu_utilization_ratio"], "higher = worse",
            "Absolute divergence of CPU usage from the request.")
    else:
        warn("No utilisation-ratio columns -- component omitted.")

    if comp.shape[1] == 0:
        raise ValueError(
            "None of the six degradation indicators could be constructed from "
            "the available columns. Silent-failure analysis cannot proceed; "
            "this is reported rather than faked."
        )
    if comp.shape[1] < 4:
        warn(f"Only {comp.shape[1]}/6 degradation components available -- the "
             "composite is correspondingly narrower and is reported as such.")

    info(f"built {comp.shape[1]} risk components: {list(comp.columns)}")
    return comp, prov


# ---------------------------------------------------------------------------
# 2. Combine into composites
# ---------------------------------------------------------------------------

def combine_components(comp: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compute the three composites and their agreement."""
    out = pd.DataFrame(index=comp.index)
    meta: dict = {"n_components": comp.shape[1], "components": list(comp.columns)}

    X = comp.to_numpy(dtype="float64")
    X = np.nan_to_num(X, nan=0.5)  # neutral rank for a missing component

    # (a) EQUAL ------------------------------------------------------------
    out["risk_equal"] = X.mean(axis=1)
    meta["equal_weights"] = {c: round(1 / comp.shape[1], 6) for c in comp.columns}

    # (b) PCA --------------------------------------------------------------
    # Weights = |PC1 loadings|, normalised to sum to 1. PC1 is the direction of
    # greatest joint variation among the symptoms, so this asks the data itself
    # which indicators carry the shared "stress" signal.
    try:
        from sklearn.decomposition import PCA
        Xc = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
        p = PCA(n_components=min(comp.shape[1], 3), random_state=config.RANDOM_STATE)
        p.fit(Xc)
        load = np.abs(p.components_[0])
        w = load / load.sum()
        # Orient so that higher score = worse. PC1's sign is arbitrary, so we
        # align it with the equal-weight composite, which is unambiguous by
        # construction.
        proj = Xc @ w
        if np.corrcoef(proj, out["risk_equal"])[0, 1] < 0:
            proj = -proj
        out["risk_pca"] = rank_normalise(pd.Series(proj, index=comp.index))
        meta["pca_weights"] = {c: round(float(x), 6) for c, x in zip(comp.columns, w)}
        meta["pca_explained_variance_ratio"] = [round(float(v), 6) for v in p.explained_variance_ratio_]
        meta["pca_pc1_explains"] = round(float(p.explained_variance_ratio_[0]), 6)
    except Exception as exc:                                    # pragma: no cover
        warn(f"PCA composite unavailable ({exc}); falling back to equal weights.")
        out["risk_pca"] = out["risk_equal"]
        meta["pca_weights"] = None

    # (c) MAHALANOBIS ------------------------------------------------------
    # Distance from the joint centre in the metric defined by the components'
    # own covariance. Needs no weights at all: correlated symptoms are
    # automatically not double-counted.
    try:
        Xc = X - X.mean(axis=0)
        cov = np.cov(Xc, rowvar=False)
        cov = np.atleast_2d(cov) + np.eye(X.shape[1]) * 1e-6
        inv = np.linalg.pinv(cov)
        d2 = np.einsum("ij,jk,ik->i", Xc, inv, Xc)
        out["risk_mahalanobis"] = rank_normalise(pd.Series(np.sqrt(np.maximum(d2, 0)),
                                                           index=comp.index))
        meta["mahalanobis_ok"] = True
    except Exception as exc:                                    # pragma: no cover
        warn(f"Mahalanobis composite unavailable ({exc}).")
        out["risk_mahalanobis"] = out["risk_equal"]
        meta["mahalanobis_ok"] = False

    # -- agreement between schemes ----------------------------------------
    agree = {}
    schemes = ["risk_equal", "risk_pca", "risk_mahalanobis"]
    for i in range(len(schemes)):
        for j in range(i + 1, len(schemes)):
            a, b = schemes[i], schemes[j]
            rho = stats.spearmanr(out[a], out[b]).statistic
            agree[f"{a}__vs__{b}"] = round(float(rho), 6)
    meta["spearman_agreement"] = agree
    meta["min_agreement"] = round(float(min(agree.values())), 6) if agree else None

    primary = {"equal": "risk_equal", "pca": "risk_pca",
               "mahalanobis": "risk_mahalanobis"}[config.PRIMARY_RISK_SCHEME]
    out["silent_failure_risk_score"] = out[primary]
    meta["primary_scheme"] = config.PRIMARY_RISK_SCHEME
    meta["primary_column"] = primary

    info(f"composite agreement (Spearman): {agree}")
    return out, meta


def assign_risk_bands(score: pd.Series) -> pd.Series:
    """
    Map the continuous score to ordinal bands using **quantiles of this trace**.

    These are explicitly distributional cut-points, not absolute engineering
    thresholds -- there is no external calibration data that could justify
    absolute ones. The continuous score is always retained alongside, so no
    information is lost by banding.
    """
    q = {lo: score.quantile(lo) for lo, _hi, _n in config.RISK_BANDS}
    labels = pd.Series("NORMAL", index=score.index, dtype="object")
    for lo, hi, name in config.RISK_BANDS:
        thr_lo = score.quantile(lo)
        thr_hi = score.quantile(min(hi, 1.0)) if hi <= 1.0 else score.max() + 1
        labels[(score >= thr_lo) & (score < thr_hi)] = name
    labels[score >= score.quantile(0.95)] = "HIGH_RISK"
    return pd.Categorical(labels, categories=["NORMAL", "LOW_RISK", "MEDIUM_RISK", "HIGH_RISK"],
                          ordered=True)


# ---------------------------------------------------------------------------
# 3. Validation against the independent explicit-failure signal
# ---------------------------------------------------------------------------

def validate_risk_score(df: pd.DataFrame, score_col: str = "silent_failure_risk_score") -> dict:
    """
    Test whether the degradation score carries real information.

    This is a *construct-validity* test, not a performance claim. The logic:
    the score is built entirely from resource-behaviour symptoms and never
    touches ``failed``; so if machines the score flags also fail explicitly
    more often, the score is tracking something real about system health.

    Reported honestly in both directions -- a weak result is stated as weak.
    """
    res: dict = {"score_column": score_col}
    if config.TARGET not in df.columns:
        res["status"] = "no target available; validation not possible"
        return res

    y = df[config.TARGET].astype(int)
    s = df[score_col].astype("float64")

    # (a) monotone association
    rho = stats.spearmanr(s, y)
    res["spearman_rho_vs_failed"] = round(float(rho.statistic), 6)
    res["spearman_p"] = float(rho.pvalue)

    # (b) does failure rate rise across score deciles?
    dec = pd.qcut(s, 10, labels=False, duplicates="drop")
    by_dec = pd.DataFrame({"decile": dec, "failed": y}).groupby("decile")["failed"].agg(
        ["mean", "size"])
    res["failure_rate_by_decile"] = {int(k): round(float(v), 6)
                                     for k, v in by_dec["mean"].items()}
    if len(by_dec) >= 2:
        res["failure_rate_bottom_decile"] = round(float(by_dec["mean"].iloc[0]), 6)
        res["failure_rate_top_decile"] = round(float(by_dec["mean"].iloc[-1]), 6)
        lo = by_dec["mean"].iloc[0]
        res["lift_top_vs_bottom"] = round(float(by_dec["mean"].iloc[-1] / lo), 4) if lo > 0 else None
        # Is the trend monotone? (a stricter test than a single correlation)
        tau = stats.kendalltau(np.arange(len(by_dec)), by_dec["mean"].to_numpy())
        res["decile_trend_kendall_tau"] = round(float(tau.statistic), 6)
        res["decile_trend_p"] = float(tau.pvalue)

    # (c) discrimination -- AUC of the score treated as a ranker
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        res["auc_score_vs_failed"] = round(float(roc_auc_score(y, s)), 6)
        res["pr_auc_score_vs_failed"] = round(float(average_precision_score(y, s)), 6)
        res["base_rate"] = round(float(y.mean()), 6)
    except Exception as exc:                                    # pragma: no cover
        res["auc_error"] = str(exc)

    # (d) honest verdict
    auc = res.get("auc_score_vs_failed")
    if auc is None:
        res["verdict"] = "Could not be evaluated."
    elif auc >= 0.65:
        res["verdict"] = (
            f"The degradation score discriminates explicit failures with "
            f"AUC = {auc:.4f}, despite never observing the failure label. This "
            f"is evidence of construct validity: the resource-behaviour "
            f"symptoms carry genuine information about system health.")
    elif auc >= 0.55:
        res["verdict"] = (
            f"The degradation score shows a weak but non-trivial association "
            f"with explicit failure (AUC = {auc:.4f}). It captures some real "
            f"signal, but is far from a failure predictor and must not be "
            f"presented as one.")
    else:
        res["verdict"] = (
            f"AUC = {auc:.4f} is close to chance. On this trace there is "
            f"INSUFFICIENT EVIDENCE that the constructed degradation score "
            f"anticipates explicit failure. The score remains a defensible "
            f"description of resource stress, but no predictive claim is made.")
    return res


# ---------------------------------------------------------------------------
# 4. Machine-level aggregation
# ---------------------------------------------------------------------------

def machine_risk_profile(df: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate row-level risk to machines.

    A ``support`` column records how many observations each machine has, and
    ``trend_slope`` is computed **only** where >= 3 timestamped observations
    exist (:data:`config.MIN_EVENTS_FOR_MACHINE_SLOPE`); elsewhere it is NaN
    rather than a fabricated zero. A ``confidence`` label makes the sample-size
    limitation visible in the deliverable itself, because with a median of 4
    events per machine most slopes are genuinely untrustworthy.
    """
    if "machine_id" not in df.columns:
        return pd.DataFrame()

    work = df[["machine_id"]].copy()
    if "cluster" in df.columns:
        work["cluster"] = df["cluster"]
    work["risk"] = df["silent_failure_risk_score"].to_numpy()
    for c in comp.columns:
        work[c] = comp[c].to_numpy()
    if config.TARGET in df.columns:
        work["failed"] = df[config.TARGET].to_numpy()
    for c in ["cpu_avg", "mem_avg", "cpu_efficiency", "memory_pressure", "cpi",
              "telemetry_loss", "cpu_tail_ratio"]:
        if c in df.columns:
            work[c] = df[c].to_numpy()
    tcol = "t_hours_from_start" if "t_hours_from_start" in df.columns else None
    if tcol:
        work["t"] = df[tcol].to_numpy()

    agg: dict = {"risk": ["mean", "max", "std", "size"]}
    for c in comp.columns:
        agg[c] = "mean"
    for c in ["cpu_avg", "mem_avg", "cpu_efficiency", "memory_pressure", "cpi",
              "telemetry_loss", "cpu_tail_ratio"]:
        if c in work.columns:
            agg[c] = "mean"
    if "failed" in work.columns:
        agg["failed"] = ["mean", "sum"]

    g = work.groupby("machine_id", sort=False).agg(agg)
    g.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in g.columns]
    g = g.rename(columns={"risk_size": "support",
                          "risk_mean": "risk_score_mean",
                          "risk_max": "risk_score_max",
                          "risk_std": "risk_score_std",
                          "failed_mean": "explicit_failure_rate",
                          "failed_sum": "explicit_failure_count"})

    if "cluster" in work.columns:
        g["cluster"] = work.groupby("machine_id", sort=False)["cluster"].first()

    # temporal slope where sample size permits
    if tcol:
        eligible = work.groupby("machine_id", sort=False).size()
        ok = eligible[eligible >= config.MIN_EVENTS_FOR_MACHINE_SLOPE].index
        slopes = (work[work.machine_id.isin(ok)]
                  .sort_values(["machine_id", "t"])
                  .groupby("machine_id", sort=False)
                  .apply(lambda d: slope(d["risk"], d["t"]), include_groups=False))
        g["risk_trend_slope"] = slopes.reindex(g.index)
        info(f"risk trend slope computed for {len(ok):,} / {len(g):,} machines "
             f"(>= {config.MIN_EVENTS_FOR_MACHINE_SLOPE} observations)")

    g["confidence"] = np.where(
        g["support"] >= 10, "moderate",
        np.where(g["support"] >= config.MIN_EVENTS_FOR_MACHINE_SLOPE, "low", "very_low"))

    g["risk_band"] = assign_risk_bands(g["risk_score_mean"])
    g = g.reset_index()
    return g.sort_values("risk_score_mean", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Orchestration + report
# ---------------------------------------------------------------------------

def run(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Full Stage 4. Returns ``(df_with_scores, machine_profile, meta)``.
    """
    section("Building degradation components")
    comp, prov = build_risk_components(df)

    section("Combining into composite scores")
    scores, meta = combine_components(comp)
    for c in scores.columns:
        df[c] = scores[c].to_numpy()
    for c in comp.columns:
        df[f"comp_{c}"] = comp[c].to_numpy()
    df["risk_band"] = assign_risk_bands(df["silent_failure_risk_score"])

    section("Validating the score against the independent explicit-failure signal")
    val = validate_risk_score(df)
    info(val.get("verdict", ""))

    section("Machine-level risk profile")
    machines = machine_risk_profile(df, comp)

    meta["provenance"] = prov
    meta["validation"] = val
    meta["band_counts"] = {str(k): int(v) for k, v in
                           df["risk_band"].value_counts().items()}

    # ---- persist ---------------------------------------------------------
    keep = ["machine_id", "collection_id", "cluster", "silent_failure_risk_score",
            "risk_equal", "risk_pca", "risk_mahalanobis", "risk_band"]
    keep += [f"comp_{c}" for c in comp.columns]
    if config.TARGET in df.columns:
        keep.append(config.TARGET)
    keep = [c for c in keep if c in df.columns]
    save_table(df[keep], config.TABLE_DIR / "silent_failure_risk.csv")

    if not machines.empty:
        save_table(machines, config.TABLE_DIR / "machine_risk_ranking.csv")
        save_table(machines, config.METRICS_DIR / "machine_risk_ranking.csv")

    _write_report(comp, meta, val, machines, df)
    return df, machines, meta


def _write_report(comp: pd.DataFrame, meta: dict, val: dict,
                  machines: pd.DataFrame, df: pd.DataFrame) -> None:
    buf = io.StringIO()

    def w(s: str = ""):
        buf.write(s + "\n")

    w("=" * 78)
    w(" SILENT-FAILURE / DEGRADATION ANALYSIS REPORT")
    w("=" * 78)
    w()
    w("SCOPE AND HONESTY STATEMENT")
    w("-" * 78)
    w("This trace contains NO ground truth for silent failure. The 'failed'")
    w("column is an EXPLICIT failure flag -- it is exactly equivalent to")
    w("event == 'FAIL', and an explicitly reported failure is by definition not")
    w("silent. Nothing in the Borg schema records undetected degradation.")
    w()
    w("Therefore this stage does NOT claim to detect silent failures. It builds a")
    w("transparent, unsupervised degradation index from documented symptoms of")
    w("stress, and then tests whether that index carries information. What is")
    w("demonstrated is a METHODOLOGY for silent-failure analysis, together with")
    w("an honest measurement of how much signal the available data supports.")
    w()
    w("=" * 78)
    w(" 1. DEGRADATION COMPONENTS")
    w("=" * 78)
    w(f"{comp.shape[1]} of 6 designed components could be constructed from this file.")
    w()
    for name, p in meta["provenance"].items():
        w(f"[{name}]")
        w(f"   built from : {p['sources']}")
        w(f"   direction  : {p['direction']}")
        w(f"   rationale  : {_wrap(p['note'], 62, 16)}")
        w()
    w("Every component is rank-normalised to [0, 1] before combination.")
    w("Rank-normalisation is used because the raw indicators have incompatible")
    w("units (CPI in cycles, sample_rate as a fraction, memory as a normalised")
    w("share). Ranks are scale-free and outlier-proof, so no conversion factor")
    w("has to be invented. A missing component contributes the neutral rank 0.5")
    w("rather than 0, so absence never looks like health.")
    w()
    w("=" * 78)
    w(" 2. WEIGHTING -- AND WHY NO WEIGHT WAS INVENTED")
    w("=" * 78)
    w("Three independent combination rules were computed:")
    w()
    w("  (a) EQUAL       -- 1/n each. The principle of indifference: no evidence")
    w("                     in this dataset justifies ranking one symptom above")
    w("                     another, so none is.")
    if meta.get("equal_weights"):
        w(f"                     weights: {meta['equal_weights']}")
    w()
    w("  (b) PCA         -- weights are |first principal component loadings|,")
    w("                     normalised to sum to 1. These are learned from the")
    w("                     components' own covariance structure, so the data")
    w("                     chooses the weights, not the author.")
    if meta.get("pca_weights"):
        w(f"                     weights: {meta['pca_weights']}")
        w(f"                     PC1 explains {100 * meta.get('pca_pc1_explains', 0):.2f}% of component variance")
    w()
    w("  (c) MAHALANOBIS -- multivariate outlyingness in the metric defined by")
    w("                     the component covariance. Weight-free by")
    w("                     construction, and it automatically avoids")
    w("                     double-counting correlated symptoms.")
    w()
    w("AGREEMENT BETWEEN THE THREE SCHEMES (Spearman rank correlation):")
    for k, v in meta.get("spearman_agreement", {}).items():
        w(f"   {k:44s} rho = {v:+.4f}")
    m = meta.get("min_agreement")
    if m is not None:
        w()
        if m >= 0.90:
            w(f"Minimum pairwise agreement is rho = {m:.4f}. The three schemes rank")
            w("machines almost identically, so the conclusions of this stage are")
            w("ROBUST TO THE WEIGHTING CHOICE. That is a stronger statement than")
            w("defending any single weight vector.")
        elif m >= 0.70:
            w(f"Minimum pairwise agreement is rho = {m:.4f} -- substantial but not")
            w("complete. Rankings are broadly consistent; individual borderline")
            w("machines can move between bands depending on the scheme.")
        else:
            w(f"Minimum pairwise agreement is only rho = {m:.4f}. The schemes")
            w("DISAGREE materially, so no single composite should be treated as")
            w("authoritative. All three are reported and the disagreement is")
            w("itself a finding.")
    w()
    w(f"Reported primary scheme: '{meta['primary_scheme']}' "
      f"({meta['primary_column']}) -- the most transparent of the three.")
    w()
    w("=" * 78)
    w(" 3. RISK BANDS")
    w("=" * 78)
    w("Bands are QUANTILES OF THIS TRACE, not absolute engineering thresholds:")
    for lo, hi, name in config.RISK_BANDS:
        w(f"   {name:14s} : score percentile [{lo:.2f}, {hi:.2f})")
    w()
    w("No external calibration data exists that could justify absolute")
    w("cut-points, so the banding is explicitly relative. The continuous score")
    w("is retained alongside the bands so nothing is lost to discretisation.")
    w()
    for k, v in meta.get("band_counts", {}).items():
        w(f"   {k:14s} {v:>10,} observations")
    w()
    w("=" * 78)
    w(" 4. VALIDATION AGAINST AN INDEPENDENT SIGNAL")
    w("=" * 78)
    w("None of the six components uses 'failed'. That independence is what makes")
    w("the following an external test rather than a circular one.")
    w()
    if "auc_score_vs_failed" in val:
        w(f"ROC-AUC of the score as a ranker of explicit failures : {val['auc_score_vs_failed']:.4f}")
        w(f"PR-AUC                                               : {val['pr_auc_score_vs_failed']:.4f}")
        w(f"Base failure rate                                    : {val['base_rate']:.4f}")
    if "spearman_rho_vs_failed" in val:
        w(f"Spearman rho (score vs failed)                       : {val['spearman_rho_vs_failed']:+.4f}"
          f"  (p = {val['spearman_p']:.3g})")
    if "failure_rate_by_decile" in val:
        w()
        w("Explicit-failure rate by risk decile:")
        for d, r in val["failure_rate_by_decile"].items():
            bar = "#" * int(round(r * 100))
            w(f"   decile {d:>2}  {r:6.4f}  {bar}")
        if val.get("lift_top_vs_bottom"):
            w()
            w(f"Top-vs-bottom decile lift : {val['lift_top_vs_bottom']:.3f}x")
        if "decile_trend_kendall_tau" in val:
            w(f"Monotonicity of the trend : Kendall tau = "
              f"{val['decile_trend_kendall_tau']:+.4f} (p = {val['decile_trend_p']:.3g})")
    w()
    w("VERDICT")
    w("-" * 78)
    w(_wrap(val.get("verdict", "n/a"), 74, 0))
    w()
    w("=" * 78)
    w(" 5. MACHINE-LEVEL PROFILE AND ITS LIMITS")
    w("=" * 78)
    if machines.empty:
        w("No machine identifier available.")
    else:
        w(f"Machines profiled : {len(machines):,}")
        w(f"Observations per machine -- median {machines['support'].median():.0f}, "
          f"max {machines['support'].max():.0f}")
        w()
        w("SAMPLE-SIZE LIMITATION (stated up front, not buried):")
        conf = machines["confidence"].value_counts()
        for k, v in conf.items():
            w(f"   confidence '{k}' : {v:,} machines")
        w()
        w("With a median of only a few observations per machine, per-machine")
        w("ROLLING-WINDOW degradation curves are not statistically supportable on")
        w("this file. That is why machine risk is computed CROSS-SECTIONALLY")
        w("(against cluster peers) and a trend slope is reported only where at")
        w("least 3 timestamped observations exist. Machines flagged at 'very_low'")
        w("confidence should be read as candidates for investigation, never as")
        w("conclusions.")
        w()
        w("TOP 15 MACHINES BY MEAN DEGRADATION SCORE")
        w("-" * 78)
        cols = [c for c in ["machine_id", "cluster", "risk_score_mean", "support",
                            "confidence", "explicit_failure_rate", "risk_band"]
                if c in machines.columns]
        w(machines[cols].head(15).to_string(index=False,
                                            float_format=lambda v: f"{v:.4f}"))
    w()
    save_text(buf.getvalue(), config.REPORT_DIR / "silent_failure_report.txt")

    rows = []
    for name, p in meta["provenance"].items():
        rows.append({"component": name, "sources": ", ".join(p["sources"]),
                     "direction": p["direction"],
                     "equal_weight": meta.get("equal_weights", {}).get(name),
                     "pca_weight": (meta.get("pca_weights") or {}).get(name),
                     "rationale": p["note"]})
    save_table(pd.DataFrame(rows), config.TABLE_DIR / "risk_components.csv")
    save_table(pd.DataFrame([{k: v for k, v in val.items()
                              if not isinstance(v, dict)}]),
               config.METRICS_DIR / "risk_validation.csv")


def _wrap(text: str, width: int, indent: int) -> str:
    import textwrap
    lines = textwrap.wrap(str(text), width=width)
    pad = " " * indent
    return ("\n" + pad).join(lines) if lines else ""
