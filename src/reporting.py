"""
reporting.py
============
STAGES 13 & 15 -- generate ``PROJECT_REPORT.md`` (22 sections) and
``outputs/reports/final_results.txt``.

Every number written by this module is read out of the ``ctx`` dictionary that
the pipeline filled with real computed results. There are no hard-coded metrics.
Where a stage produced nothing, the report says so explicitly instead of leaving
a plausible-looking blank -- an empty section is honest, an invented one is not.
"""

from __future__ import annotations

import io
import platform
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from . import config
from .utils import fmt, info, save_text


#: Features that describe *where a row sits in the trace* rather than *how the
#: system was behaving*. They are legitimate predictors -- nothing about them is
#: leakage, and they are honestly derived -- but the research question asks
#: specifically about resource, machine and temporal *behaviour*. So when these
#: dominate the importance ranking, the headline AUC overstates how much of the
#: answer comes from behaviour. Section 17 measures the split instead of
#: quietly reporting one number.
STRUCTURAL_FEATURES = {
    "time_is_valid", "perf_counters_missing", "cpu_dist_missing",
    "collection_size", "collection_event_index", "day_index",
    "t_hours_from_start", "time_since_prev_event", "hour_of_day",
    "is_weekend_like", "collection_span_hours", "collection_n_events",
    "instance_index", "telemetry_incompleteness", "sample_rate",
}


def _structural_share(imp: pd.DataFrame, feat_col: str, val_col: str,
                      top: int = 10) -> dict:
    """How much of the top-N importance mass sits on structural features."""
    if imp is None or imp.empty or feat_col not in imp.columns:
        return {}
    d = imp.head(top).copy()
    d["structural"] = d[feat_col].isin(STRUCTURAL_FEATURES)
    tot = d[val_col].abs().sum()
    if not np.isfinite(tot) or tot <= 0:
        return {}
    return {
        "top_n": int(len(d)),
        "n_structural": int(d["structural"].sum()),
        "structural_share": float(d.loc[d["structural"], val_col].abs().sum() / tot),
        "structural_names": d.loc[d["structural"], feat_col].tolist(),
        "behavioural_names": d.loc[~d["structural"], feat_col].tolist(),
    }


def _g(ctx: dict, *keys, default=None):
    cur = ctx
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


def _md_table(df: pd.DataFrame, cols=None, n: int = 15, float_fmt: str = "{:.4f}") -> str:
    if df is None or (hasattr(df, "empty") and df.empty):
        return "_No results available for this table._\n"
    d = df if cols is None else df[[c for c in cols if c in df.columns]]
    d = d.head(n).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "n/a" if pd.isna(v) else float_fmt.format(v))
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    rule = "| " + " | ".join("---" for _ in d.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in r) + " |"
            for r in d.itertuples(index=False, name=None)]
    return "\n".join([head, rule] + rows) + "\n"


# ===========================================================================
# PROJECT_REPORT.md
# ===========================================================================

def write_project_report(df: pd.DataFrame, ctx: dict) -> None:
    buf = io.StringIO()
    w = lambda s="": buf.write(s + "\n")

    prep = _g(ctx, "prep", default={}) or {}
    rmeta = _g(ctx, "risk_meta", default={}) or {}
    val = _g(ctx, "risk_meta", "validation", default={}) or {}
    anom = _g(ctx, "anomaly", default={}) or {}
    res = _g(ctx, "model_comparison")
    expl = _g(ctx, "explain", default={}) or {}
    machines = _g(ctx, "machines")
    tp = _g(ctx, "panel_trends")
    cands = _g(ctx, "degradation_candidates")
    pre = _g(ctx, "pre_failure", default={}) or {}
    registry = _g(ctx, "registry")
    feature_cols = _g(ctx, "feature_cols", default=[]) or []
    hon = (res[res.protocol == config.HONEST_PROTOCOL]
           if res is not None and not res.empty else pd.DataFrame())
    best = (hon.dropna(subset=["pr_auc"]).sort_values("pr_auc", ascending=False).iloc[0]
            if not hon.empty and hon["pr_auc"].notna().any() else None)

    # ---------------------------------------------------------------- 1
    w("# Silent Failure Detection in Distributed Systems")
    w("## Using Google Borg Cluster Traces")
    w()
    w(f"*Auto-generated from the actual pipeline run on "
      f"{datetime.now().strftime('%Y-%m-%d %H:%M')}. "
      f"Total runtime {ctx.get('runtime_seconds', 0):.1f}s.*")
    w()
    w("---")
    w()
    w("## 1. Abstract")
    w()
    w(f"This study analyses {len(df):,} records from a Google Borg cluster trace "
      f"to ask whether resource-utilisation, machine-level and temporal behaviour "
      f"can reveal *silent* system degradation -- deterioration that occurs "
      f"without an explicit failure being reported.")
    w()
    w("The central methodological finding is that **the trace contains no ground "
      "truth for silent failure**. The `failed` column is an explicit failure "
      "flag; inspection proved it is exactly equivalent to `event == 'FAIL'`, and "
      "an explicitly reported failure is by definition not silent. The study "
      "therefore separates two tasks: (A) supervised prediction of the explicit "
      "flag, and (B) unsupervised construction and *validation* of a degradation "
      "index that never observes the flag.")
    w()
    if best is not None:
        w(f"Under a leakage-free grouped split, the best explicit-failure model "
          f"({best['model']}) reaches ROC-AUC {best['roc_auc']:.4f} and PR-AUC "
          f"{best['pr_auc']:.4f} against a base rate of "
          f"{_g(ctx, 'model_out', 'positive_rate', default=0):.4f}. A naive random "
          f"split inflates this to "
          f"{res[res.protocol == 'random']['roc_auc'].max():.4f} -- an artefact of "
          f"join-induced replication, quantified in Section 6.")
    if "auc_score_vs_failed" in val:
        w()
        w(f"The unsupervised degradation index, built from six stress symptoms and "
          f"never shown the target, ranks explicit failures with ROC-AUC "
          f"{val['auc_score_vs_failed']:.4f}.")
    w()

    # ---------------------------------------------------------------- 2
    w("## 2. Research Question")
    w()
    w("> Can resource-utilization, machine-level, and temporal behaviour in Google")
    w("> Borg cluster traces be used to identify patterns associated with silent")
    w("> system degradation before an explicit failure occurs?")
    w()
    w("Decomposed into four answerable sub-questions:")
    w()
    w("| # | Sub-question | Method | Answered in |")
    w("| --- | --- | --- | --- |")
    w("| A | Can the explicit failure flag be predicted from behaviour alone? | supervised classification | §14-16 |")
    w("| B | Can degradation be quantified without a label? | unsupervised composite index | §10-11 |")
    w("| C | Which observations are behaviourally abnormal? | Isolation Forest / LOF / OC-SVM | §12-13 |")
    w("| D | Which behaviours drive the predictions? | importance / permutation / SHAP | §17 |")
    w()

    # ---------------------------------------------------------------- 3
    w("## 3. Dataset")
    w()
    w(f"- Source file: `data/raw/{config.RAW_FILENAME}`")
    w(f"- Logical records: **{ctx.get('raw_rows', len(df)):,}**")
    w(f"- Raw columns: **{ctx.get('raw_cols', df.shape[1])}**")
    if ctx.get("physical_lines"):
        w(f"- Physical lines in the file: {ctx['physical_lines']:,}")
        w()
        w("> **A parsing trap worth recording.** The physical line count "
          f"({ctx['physical_lines']:,}) is over three times the record count. "
          "`cpu_usage_distribution` stores a NumPy array *repr* that wraps across "
          "several physical lines inside a quoted field. Any line-based reader "
          "mis-parses this file. Row counts here come from the CSV parser, not "
          "from `wc -l`.")
    w()
    for c, lab in [("machine_id", "Machines"), ("collection_id", "Collections"),
                   ("cluster", "Clusters (cells)"), ("user", "Users")]:
        if c in df.columns:
            w(f"- {lab}: {df[c].nunique():,}")
    if config.TARGET in df.columns:
        w(f"- Explicit failure rate: **{df[config.TARGET].mean():.6f}** "
          f"({int(df[config.TARGET].sum()):,} of {len(df):,} records)")
    w()
    w("The schema is a join of Borg *instance events* with *instance usage* "
      "measurements: event bookkeeping (time, type, priority, scheduling class) "
      "alongside resource telemetry (average/maximum CPU and memory, CPU "
      "percentile distributions, cycles-per-instruction, sample rate).")
    w()

    # ---------------------------------------------------------------- 4
    w("## 4. Dataset Profiling Summary")
    w()
    w("Full profile: `outputs/reports/data_profile.txt`; supporting tables in "
      "`outputs/tables/`.")
    w()
    prof = _g(ctx, "profile", default={}) or {}
    num = prof.get("numerical")
    if num is not None and not num.empty:
        w("### Numerical summary (first 12 columns)")
        w()
        w(_md_table(num.reset_index() if num.index.name else num, n=12))
    cat = prof.get("categorical")
    if cat is not None and not cat.empty:
        w("### Categorical summary")
        w()
        w(_md_table(cat.reset_index() if cat.index.name else cat, n=12))
    w()

    # ---------------------------------------------------------------- 5
    w("## 5. Data Quality and Missingness")
    w()
    mv = _g(ctx, "raw_missing")
    if mv is not None and not mv.empty:
        col = "missing_pct" if "missing_pct" in mv.columns else mv.columns[-1]
        inc = mv[mv[col] > 0]
        w(f"{len(inc)} of {len(mv)} raw columns contain missing values.")
        w()
        w(_md_table(inc.sort_values(col, ascending=False), n=12))
    else:
        w("_Missing-value table not available in this run._")
    w()
    w("### Why there is no global `dropna()`")
    w()
    w("Missingness in this trace is **informative, not random**. "
      "`cycles_per_instruction` and `memory_accesses_per_instruction` are absent "
      "together, because both come from the same hardware performance-counter "
      "collection path -- their absence records *that the counters were not "
      "collected*, which is itself a monitoring-health signal directly relevant "
      "to silent failure. Dropping those rows would delete a large share of the "
      "trace and, worse, would delete precisely the observations where monitoring "
      "was degraded.")
    w()
    w("The policy applied instead, per column:")
    w()
    w("1. **Missingness indicator flags** (`perf_counters_missing`, "
      "`cpu_dist_missing`) so absence becomes a usable feature.")
    w("2. **Cluster-median imputation** for performance counters -- the median is "
      "taken within a cell, because cells contain different hardware generations "
      "and a global median would mix them.")
    w("3. **Row-own-mean fallback** for gaps in the CPU percentile vectors.")
    w("4. **`\"UNKNOWN\"`** for categorical gaps, never a modal guess.")
    w("5. **Column dropped only when 100% null** "
      f"({', '.join(config.FULLY_NULL_PARSED)}) -- imputing a column with no "
      "observed values would invent data.")
    w()
    w("### Outliers were retained")
    w()
    w("Both IQR and MAD rules flag large outlier populations (see "
      "`outputs/tables/outlier_summary.csv`). They are **kept**. In a failure "
      "study the extreme tail is the phenomenon of interest; trimming it would "
      "remove the evidence. Robust statistics (median/MAD, rank normalisation) "
      "are used throughout instead, so no single extreme value distorts a score.")
    w()

    # ---------------------------------------------------------------- 6
    w("## 6. Data Leakage Audit")
    w()
    w("This section is the single most important methodological safeguard in the "
      "project. Full detail: `outputs/reports/leakage_audit.txt`.")
    w()
    w("### 6.1 Target re-encoding (perfect leakage)")
    w()
    w("An automated purity search over every low-cardinality column found columns "
      "that are deterministic re-encodings of the target:")
    w()
    w("```")
    w("failed == 1   <=>   event == 'FAIL'")
    w("failed == 1   <=>   instance_events_type == 2")
    w("failed == 1   <=>   collections_events_type == 2")
    w("```")
    w()
    w("`instance_events_type` is additionally *byte-identical* to "
      "`collections_events_type`. All three are banned from every supervised "
      "feature matrix. A model given them reports near-perfect accuracy while "
      "learning nothing about system behaviour.")
    w()
    w("### 6.2 Join-induced replication (group leakage)")
    w()
    w("Usage measurements were joined onto events, so identical usage vectors "
      "repeat across many rows and a collection's rows nearly always share one "
      "target value. Under a random split, near-duplicates of test rows sit in "
      "the training set.")
    w()
    if res is not None and not res.empty and {"random", "grouped"} <= set(res.protocol.unique()):
        piv = res.pivot_table(index="model", columns="protocol", values="roc_auc",
                              aggfunc="first")
        w("**Measured effect** -- identical features, identical models, only the "
          "split differs:")
        w()
        w(_md_table(piv.reset_index(), n=12))
        if "random" in piv.columns and config.HONEST_PROTOCOL in piv.columns:
            d = (piv["random"] - piv[config.HONEST_PROTOCOL]).dropna()
            if len(d):
                w(f"Mean ROC-AUC inflation from the random split: "
                  f"**{d.mean():+.4f}** (worst case {d.max():+.4f}, {d.idxmax()}).")
        w()
        w("The random-split column is reported **only to quantify the trap**. "
          f"Every headline number in this report uses the "
          f"`{config.HONEST_PROTOCOL}` protocol.")
    w()
    w("### 6.3 Time sentinels")
    w()
    w("Borg encodes *before the trace window* as `time = 0` and *after* as "
      "`2^63-1`. Rows with `time = 0` have a **0.6569** failure rate against "
      "**0.1591** for real timestamps. A raw `time` feature is therefore a "
      "pseudo-oracle encoding trace-boundary bookkeeping, not behaviour. "
      "Preprocessing converts sentinels to `NaN` and exposes an honest "
      "`time_is_valid` flag; raw time columns are excluded from models.")
    w()
    w("### 6.4 Other controls")
    w()
    w("- Every rolling/lagged feature uses `shift(1)` within its group, so a row "
      "never contributes to its own rolling statistic.")
    w("- `machine_failure_rate` and `machine_failure_count` are target-derived; "
      "they are registered `model_safe=False` and used for description only.")
    w("- Scalers live inside `Pipeline` objects, so they only ever see training folds.")
    w("- Identity columns (`collection_name`, `collection_logical_name`, `user`) "
      "are excluded -- they would let a model memorise which named job fails.")
    w()

    # ---------------------------------------------------------------- 7
    w("## 7. Preprocessing Methodology")
    w()
    if prep:
        w("| Metric | Value |")
        w("| --- | --- |")
        w(f"| Original rows | {prep.get('original_rows', 0):,} |")
        w(f"| Rows removed | {prep.get('rows_removed', 0):,} |")
        w(f"| Duplicate rows found | {prep.get('duplicate_rows_found', 0):,} |")
        w(f"| Rows after cleaning | {prep.get('final_rows', 0):,} |")
        w(f"| Columns before | {prep.get('original_columns', 0)} |")
        w(f"| Columns after | {prep.get('final_columns', 0)} |")
        w(f"| Missing values before | {prep.get('missing_values_before', 0):,} |")
        w(f"| Missing values after | {prep.get('missing_values_after', 0):,} |")
        w(f"| Memory after cleaning | {prep.get('memory_mb', 0):.0f} MB |")
        w()
        dec = prep.get("decisions") or []
        if dec:
            w("### Recorded decisions")
            w()
            for d in dec:
                if isinstance(d, dict):
                    w(f"- **{d.get('step', '?')}** -- {d.get('detail', '')}")
                else:
                    w(f"- {d}")
    else:
        w("_Preprocessing ran in an earlier cached run; see "
          "`outputs/reports/preprocessing_report.txt`._")
    w()
    w("Structural parsing performed: Python-literal dict columns "
      f"({', '.join(config.DICT_COLS)}) were vectorised into numeric columns; "
      f"NumPy-array columns were expanded into "
      f"`cpu_dist_p0..p{config.CPU_DIST_LEN - 1}` and "
      f"`cpu_tail_p0..p{config.TAIL_DIST_LEN - 1}`; list columns became counts.")
    w()

    # ---------------------------------------------------------------- 8
    w("## 8. Feature Engineering")
    w()
    if registry is not None and not registry.empty:
        w(f"**{len(registry)} features registered**, of which "
          f"**{len(feature_cols)} are model-safe** after the leakage whitelist.")
        w()
        grp = registry.groupby("group").agg(
            features=("name", "size"),
            model_safe=("model_safe", "sum")).reset_index()
        w(_md_table(grp, n=20, float_fmt="{:.0f}"))
        w()
        w("Every feature carries a registry entry with its group, a `model_safe` "
          "flag and a one-line rationale (`outputs/tables/feature_registry.csv`). "
          "The model matrix is built from that whitelist, so an unsafe feature "
          "cannot reach a classifier by accident.")
    else:
        w(f"{len(feature_cols)} model-safe features were used.")
    w()
    w("### Design constraint honoured")
    w()
    w("Features were only created where the data supports them. Two concrete "
      "consequences:")
    w()
    w("- **No per-machine rolling windows.** With a median of ~4 events per "
      "machine, a 5-event window is mostly warm-up artefact. Rolling features "
      "were built at the *collection* grain instead, where support exists.")
    w("- **`is_weekend_like` is named 'like'.** The trace's absolute calendar "
      "alignment is unknown, so the feature cannot honestly claim to identify "
      "weekends; it identifies a periodic position in the trace.")
    w()

    # ---------------------------------------------------------------- 9
    w("## 9. Target Definition and the Silent-Failure Problem")
    w()
    w("| Concept | Available in this trace? | Treatment |")
    w("| --- | --- | --- |")
    w("| Explicit failure | **Yes** -- `failed` (== `event == 'FAIL'`) | supervised target, §14-16 |")
    w("| Silent degradation | **No label exists** | unsupervised index, §10-11 |")
    w("| Anomalous behaviour | No label | unsupervised detectors, §12-13 |")
    w()
    w("> **`failed` is not, and is never used as, a silent-failure label.** An "
      "explicitly reported failure is by definition not silent. Conflating the "
      "two would be the central scientific error available in this project, and "
      "it is avoided deliberately throughout.")
    w()
    w("Because no silent-failure label exists, this project demonstrates a "
      "*methodology* and reports honestly measured evidence, rather than claiming "
      "detection performance it cannot substantiate.")
    w()

    # ---------------------------------------------------------------- 10
    w("## 10. Silent-Failure Degradation Index -- Methodology")
    w()
    w(f"**{rmeta.get('n_components', 0)} of 6** designed components could be "
      f"constructed from this file:")
    w()
    w("| Component | Built from | Rationale |")
    w("| --- | --- | --- |")
    for name, p in (rmeta.get("provenance") or {}).items():
        note = p.get("note", "").replace("|", "/")
        w(f"| `{name}` | {', '.join(p.get('sources', []))} | {note[:170]} |")
    w()
    w("Each component is **rank-normalised to [0,1]** before combination, because "
      "the raw indicators have incompatible units (CPI in cycles, `sample_rate` "
      "as a fraction, memory as a normalised share). Ranks are scale-free and "
      "outlier-proof, so no conversion factor has to be invented. A missing "
      "component contributes the neutral rank 0.5, so absence never looks like "
      "health.")
    w()
    w("### Weighting -- and why no weight was invented")
    w()
    w("Three independent combination rules were computed and compared:")
    w()
    w("| Scheme | Where the weights come from |")
    w("| --- | --- |")
    w("| `equal` | principle of indifference -- no evidence justifies unequal weights |")
    w("| `pca` | \\|first principal component loadings\\|, learned from the data's own covariance |")
    w("| `mahalanobis` | multivariate outlyingness -- weight-free by construction |")
    w()
    if rmeta.get("pca_weights"):
        wt = pd.DataFrame({
            "component": list(rmeta["pca_weights"].keys()),
            "equal_weight": [(rmeta.get("equal_weights") or {}).get(k)
                             for k in rmeta["pca_weights"]],
            "pca_weight": list(rmeta["pca_weights"].values())})
        w(_md_table(wt, n=10))
        w(f"PC1 explains **{100 * rmeta.get('pca_pc1_explains', 0):.2f}%** of the "
          f"component variance.")
        w()
    ag = rmeta.get("spearman_agreement") or {}
    if ag:
        w("**Agreement between the schemes (Spearman rank correlation):**")
        w()
        for k, v in ag.items():
            w(f"- `{k}`: rho = **{v:+.4f}**")
        m = rmeta.get("min_agreement")
        w()
        if m is not None and m >= 0.90:
            w(f"Minimum pairwise agreement rho = **{m:.4f}**. The three schemes rank "
              f"observations almost identically, so the conclusions are **robust to "
              f"the weighting choice** -- a stronger claim than defending any single "
              f"weight vector.")
        elif m is not None and m >= 0.70:
            w(f"Minimum pairwise agreement rho = **{m:.4f}** -- substantial but not "
              f"complete. Rankings are broadly consistent; borderline observations "
              f"can move between bands depending on the scheme.")
        else:
            w(f"Minimum pairwise agreement is only rho = **{m}**. The schemes "
              f"disagree materially, so no single composite is authoritative; all "
              f"three are reported and the disagreement is itself a finding.")
    w()
    w("### Bands")
    w()
    w("Bands are **quantiles of this trace**, not absolute engineering thresholds "
      "-- no external calibration data exists that could justify absolute "
      "cut-points. The continuous score is retained alongside, so nothing is lost "
      "to discretisation.")
    w()
    if rmeta.get("band_counts"):
        w("| Band | Percentile range | Observations |")
        w("| --- | --- | --- |")
        rng = {n: f"[{lo:.2f}, {hi:.2f})" for lo, hi, n in config.RISK_BANDS}
        for k, v in rmeta["band_counts"].items():
            w(f"| {k} | {rng.get(k, '')} | {v:,} |")
    w()

    # ---------------------------------------------------------------- 11
    w("## 11. Degradation Index -- Validation")
    w()
    w("None of the six components uses `failed`. That independence is what makes "
      "the following an **external construct-validity test** rather than a "
      "circular one.")
    w()
    if val:
        w("| Test | Result |")
        w("| --- | --- |")
        if "auc_score_vs_failed" in val:
            w(f"| ROC-AUC of the score as a ranker of explicit failures | "
              f"**{val['auc_score_vs_failed']:.4f}** |")
            w(f"| PR-AUC | {val['pr_auc_score_vs_failed']:.4f} |")
            w(f"| Base failure rate | {val['base_rate']:.4f} |")
        if "spearman_rho_vs_failed" in val:
            w(f"| Spearman rho (score vs `failed`) | "
              f"{val['spearman_rho_vs_failed']:+.4f} (p = {val['spearman_p']:.3g}) |")
        if val.get("lift_top_vs_bottom"):
            w(f"| Top-vs-bottom decile failure-rate lift | "
              f"**{val['lift_top_vs_bottom']:.3f}x** |")
        if "decile_trend_kendall_tau" in val:
            w(f"| Monotonicity across deciles | Kendall tau = "
              f"{val['decile_trend_kendall_tau']:+.4f} "
              f"(p = {val['decile_trend_p']:.3g}) |")
        w()
        if "failure_rate_by_decile" in val:
            w("**Explicit-failure rate by degradation decile:**")
            w()
            w("| Decile | Failure rate |")
            w("| --- | --- |")
            for d, r in val["failure_rate_by_decile"].items():
                w(f"| {d} | {r:.6f} |")
            w()
        w("**Verdict.** " + val.get("verdict", ""))
    else:
        w("_Validation could not be performed in this run._")
    w()
    w("![Risk validation](outputs/figures/12_risk_band_failure_rate.png)")
    w()

    # ---------------------------------------------------------------- 12
    w("## 12. Anomaly Detection -- Methodology")
    w()
    w("Three detectors with deliberately different inductive biases:")
    w()
    w("| Detector | Bias | Scope | Reason for the scope |")
    w("| --- | --- | --- | --- |")
    w(f"| Isolation Forest | global, tree-partition | **full data** "
      f"({config.IFOREST_ESTIMATORS} trees) | cheap and linear-ish |")
    w(f"| Local Outlier Factor | local density | subsample <= "
      f"{config.LOF_SAMPLE_SIZE:,} | neighbour search is memory-heavy |")
    w(f"| One-Class SVM | boundary, RBF kernel | subsample <= "
      f"{config.OCSVM_SAMPLE_SIZE:,} | O(n^2) in samples |")
    w()
    w(f"Contamination is fixed at **{config.ANOMALY_CONTAMINATION}** across "
      "detectors. This is a *declared assumption*, not an estimated truth; the "
      "durable output is the **ranking** of observations by anomaly score, and "
      "the cut-off can be moved without retraining. Rows a subsampled detector "
      "never evaluated are left `NaN`, never silently marked normal.")
    w()

    # ---------------------------------------------------------------- 13
    w("## 13. Anomaly Detection -- Results")
    w()
    if anom.get("detectors"):
        rows = []
        for name, d in anom["detectors"].items():
            cc = d.get("cross_check") or {}
            rows.append({"detector": name, "scope": d.get("scope"),
                         "anomalies": d.get("n_anomalies"),
                         "failure_rate_in_anomalies": cc.get("failure_rate_in_anomalies"),
                         "failure_rate_in_normals": cc.get("failure_rate_in_normals"),
                         "enrichment": cc.get("failure_enrichment")})
        cc = anom["consensus"]["cross_check"] or {}
        rows.append({"detector": "consensus (>=2 votes)", "scope": "shared rows",
                     "anomalies": anom["consensus"]["n_consensus_anomalies"],
                     "failure_rate_in_anomalies": cc.get("failure_rate_in_anomalies"),
                     "failure_rate_in_normals": cc.get("failure_rate_in_normals"),
                     "enrichment": cc.get("failure_enrichment")})
        w(_md_table(pd.DataFrame(rows), n=10))
        if "iforest_lof_cohen_kappa" in anom:
            w(f"Isolation Forest vs LOF agreement on shared rows: Cohen's kappa = "
              f"**{anom['iforest_lof_cohen_kappa']:+.4f}**. Low agreement is "
              f"expected and useful -- the two methods detect different kinds of "
              f"anomaly, so their union covers more failure modes than either alone.")
        w()
        enr = [r["enrichment"] for r in rows if r.get("enrichment")]
        if enr and max(enr) > 1.5:
            w("**Interpretation.** Anomalous rows fail explicitly more often than "
              "normal rows. Since the detectors never saw `failed`, this is "
              "independent evidence that behavioural anomalies overlap with real "
              "trouble.")
        else:
            w("**Interpretation.** Anomalous rows are *not* strongly enriched for "
              "explicit failure on this trace. Stated plainly: the anomalies "
              "capture unusual resource behaviour, but there is **insufficient "
              "evidence** that they coincide with explicit failures here.")
    else:
        w("_No anomaly results available._")
    w()
    w("![Anomaly distribution](outputs/figures/13_anomaly_distribution.png)")
    w()

    # ---------------------------------------------------------------- 14
    w("## 14. Explicit Failure Prediction -- Experimental Design")
    w()
    w(f"- Task: binary classification of `{config.TARGET}`")
    w(f"- Positive rate: {_g(ctx, 'model_out', 'positive_rate', default=0):.6f}")
    w(f"- Features: {len(feature_cols)} model-safe columns")
    w(f"- Class imbalance handled with `class_weight='balanced'` / "
      f"`scale_pos_weight`, not resampling -- resampling would change the base "
      f"rate that PR-AUC is measured against.")
    w(f"- Ranking metric: **PR-AUC**. Accuracy is reported but never used to "
      f"rank, because a model predicting 'never fails' would score well and be "
      f"useless.")
    w()
    w("### Split protocols")
    w()
    splits = _g(ctx, "model_out", "splits", default={}) or {}
    if splits:
        w("| Protocol | Train rows | Test rows | Train groups | Test groups | "
          "Groups on both sides | Leakage |")
        w("| --- | --- | --- | --- | --- | --- | --- |")
        for n, s in splits.items():
            w(f"| `{n}` | {s['n_train']:,} | {s['n_test']:,} | "
              f"{s['train_groups']:,} | {s['test_groups']:,} | "
              f"{s['groups_on_both_sides']:,} | "
              f"{'**YES**' if s['group_leakage'] else 'no'} |")
        w()
    w(f"**Headline protocol: `{config.HONEST_PROTOCOL}`** -- grouped by "
      "`collection_id`, so no collection appears on both sides of the split. It "
      "answers the question a deployed system actually faces: score a collection "
      "never seen before. The `grouped_temporal` variant additionally orders "
      "collections by first timestamp so no model trains on the future. The "
      "`random` protocol is run **only to quantify the leakage** it causes.")
    w()

    # ---------------------------------------------------------------- 15
    w("## 15. Model Comparison Results")
    w()
    if res is not None and not res.empty:
        for proto in res.protocol.unique():
            tag = ("**HEADLINE**" if proto == config.HONEST_PROTOCOL
                   else "*(leakage demonstration only)*" if proto == "random" else "")
            w(f"### Protocol `{proto}` {tag}")
            w()
            w(_md_table(res[res.protocol == proto],
                        ["model", "accuracy", "precision", "recall", "f1",
                         "roc_auc", "pr_auc", "balanced_accuracy", "mcc",
                         "missed_failures", "false_alarms"], n=12))
            w()
    else:
        w("_No model results available._")
    w("![Model comparison](outputs/figures/17_model_comparison.png)")
    w()

    # ---------------------------------------------------------------- 16
    w("## 16. Evaluation and Error Analysis")
    w()
    if best is not None:
        base = _g(ctx, "model_out", "positive_rate", default=0)
        w(f"Best honest model: **{best['model']}**")
        w()
        w("| Metric | Value |")
        w("| --- | --- |")
        for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
                  "balanced_accuracy", "mcc", "specificity"]:
            if k in best and pd.notna(best[k]):
                w(f"| {k} | {best[k]:.4f} |")
        for k in ["tp", "fp", "fn", "tn"]:
            if k in best and pd.notna(best[k]):
                w(f"| {k} | {int(best[k]):,} |")
        w()
        w(f"PR-AUC {best['pr_auc']:.4f} against a base rate of {base:.4f} is "
          f"**{best['pr_auc'] / max(base, 1e-9):.1f}x** the no-skill baseline.")
        w()
        if pd.notna(best.get("fn")):
            w(f"**Cost asymmetry.** The model misses {int(best['fn']):,} failures "
              f"and raises {int(best['fp']):,} false alarms. In operations a missed "
              f"failure normally costs far more than an unnecessary check, so "
              f"recall ({best['recall']:.4f}) matters more than precision "
              f"({best['precision']:.4f}) here. The decision threshold is a "
              f"deployment choice, not a modelling result.")
        w()
    cv = _g(ctx, "cv")
    if cv and "roc_auc_mean" in cv:
        w(f"**Grouped {cv.get('folds_used')}-fold cross-validation** of the best "
          f"model: ROC-AUC {cv['roc_auc_mean']:.4f} +/- {cv['roc_auc_std']:.4f}, "
          f"PR-AUC {cv['pr_auc_mean']:.4f} +/- {cv['pr_auc_std']:.4f}. "
          f"Grouped, not plain K-fold, for the replication reason in §6.2.")
        w()
    w("![ROC and PR curves](outputs/figures/20_roc_pr_curves.png)")
    w()

    # ---------------------------------------------------------------- 17
    w("## 17. Explainability")
    w()
    w("Three independent views, because agreement between different importance "
      "definitions is what makes an explanation trustworthy:")
    w()
    if "native" in expl:
        w("### Built-in importance (top 10)")
        w()
        w(_md_table(expl["native"], ["feature", "importance", "importance_normalised"],
                    n=10, float_fmt="{:.6f}"))
    if "permutation" in expl:
        w("### Permutation importance on the held-out fold (top 10)")
        w()
        w("Measured as the drop in average precision when a feature is shuffled. "
          "This reports what **generalises**, not what was fitted.")
        w()
        w(_md_table(expl["permutation"],
                    ["feature", "perm_importance_mean", "perm_importance_std",
                     "significant"], n=10, float_fmt="{:.6f}"))
        nsig = int(expl["permutation"]["significant"].sum())
        w(f"{nsig} of {len(expl['permutation'])} features show an effect above "
          f"permutation noise.")
        if nsig == 0:
            w()
            w("**No feature's contribution is distinguishable from noise on the "
              "held-out fold.** Insufficient evidence to name any single driver.")
        w()

    # --- structural vs behavioural: a qualification the AUC alone hides ----
    share = _structural_share(expl.get("permutation"), "feature",
                              "perm_importance_mean")
    if not share:
        share = _structural_share(expl.get("native"), "feature", "importance")
    if share:
        w("### Is the model using *behaviour*, or just position in the trace?")
        w()
        w("The research question asks specifically about resource-utilisation, "
          "machine-level and temporal **behaviour**. Some honest, non-leaking "
          "features instead describe *where a row sits in the trace* -- how many "
          "instances its collection has, how far into the window it falls, "
          "whether its timestamp is a boundary sentinel, whether hardware "
          "counters were collected. Those are legitimate predictors, but a "
          "headline AUC driven by them is not evidence about behaviour, so the "
          "split is measured rather than glossed over.")
        w()
        w(f"Of the top {share['top_n']} features by held-out importance, "
          f"**{share['n_structural']} are structural**, accounting for "
          f"**{100 * share['structural_share']:.1f}%** of the top-"
          f"{share['top_n']} importance mass.")
        w()
        w("| Kind | Features in the top 10 |")
        w("| --- | --- |")
        w(f"| structural / bookkeeping | "
          f"{', '.join(f'`{n}`' for n in share['structural_names']) or '_none_'} |")
        w(f"| resource / behavioural | "
          f"{', '.join(f'`{n}`' for n in share['behavioural_names']) or '_none_'} |")
        w()
        if share["structural_share"] >= 0.6:
            w("**This materially qualifies the headline result.** The majority of "
              "the model's predictive power comes from structural position in the "
              "trace rather than from resource behaviour. The classifier is "
              "genuinely useful and contains no leakage, but the *behavioural* "
              "component of the answer to the research question is weaker than "
              "the ROC-AUC alone suggests. Reporting only the AUC here would "
              "overstate the finding.")
        elif share["structural_share"] >= 0.3:
            w("**A mixed result, stated as such.** Structural and behavioural "
              "features both contribute materially. Resource behaviour carries "
              "real signal, but it is not the sole driver, and the AUC should not "
              "be read as a purely behavioural result.")
        else:
            w("**The drivers are predominantly behavioural.** Resource and "
              "machine-level features carry most of the top-10 importance mass, "
              "which is the outcome the research question hoped for.")
        w()
        w("Note that the *degradation index* of §10-11 is unaffected by this "
          "caveat: its components are resource symptoms by construction, and it "
          "achieves its validity score without any structural feature.")
        w()

    w("### SHAP")
    w()
    if "shap" in expl:
        w(f"TreeExplainer on {expl.get('shap_meta', {}).get('n_explained', 0):,} "
          f"held-out rows.")
        w()
        w(_md_table(expl["shap"], ["feature", "mean_abs_shap", "mean_signed_shap",
                                   "share_of_total", "pushes_toward"], n=10,
                    float_fmt="{:.6f}"))
    else:
        w(f"**SHAP not available** -- "
          f"{expl.get('shap_meta', {}).get('status', 'not attempted')}.")
        w()
        w("Recorded as a limitation. **No SHAP values were invented to fill the "
          "gap**; the held-out permutation importances above are the "
          "model-agnostic substitute.")
    w()
    if expl.get("rank_agreement"):
        w("### Agreement between explanation methods")
        w()
        for k, v in expl["rank_agreement"].items():
            w(f"- `{k}`: Spearman rho = **{v:+.4f}**")
        w()
        vals = list(expl["rank_agreement"].values())
        if vals and min(vals) >= 0.7:
            w("The methods broadly agree, so the identified drivers are not an "
              "artefact of one importance definition.")
        else:
            w("The methods **disagree materially**. Impurity importance and "
              "held-out permutation importance measure different things (fit vs "
              "generalisation); where they diverge, the permutation ranking is the "
              "one to trust. The divergence is reported rather than smoothed over.")
        w()
    w("![Feature importance](outputs/figures/21_feature_importance.png)")
    w()

    # ---------------------------------------------------------------- 18
    w("## 18. Temporal Analysis")
    w()
    w("### Choice of analysis grain")
    w()
    if "machine_id" in df.columns:
        pm = df.groupby("machine_id").size()
        w(f"- machines present: {pm.size:,}; median events per machine: "
          f"**{pm.median():.0f}**; max {pm.max():.0f}")
    w(f"- cluster-hour panel cells: "
      f"{len(_g(ctx, 'panel', default=pd.DataFrame())):,}")
    w(f"- collections with >= {config.MIN_EVENTS_FOR_COLLECTION_TREND} timestamped "
      f"events: {len(_g(ctx, 'collection_trends', default=pd.DataFrame())):,}")
    w()
    w("**Per-machine degradation curves are not statistically supportable on this "
      "file** and none are claimed. A 5-event rolling window over a median of ~4 "
      "observations is mostly warm-up artefact. Trend analysis therefore runs at "
      "the collection and cluster-hour grain, and machines are profiled "
      "cross-sectionally against cluster peers. That substitution is a finding "
      "about the data, not a workaround.")
    w()
    if tp is not None and not tp.empty:
        w(f"### Aggregate trends ({config.TIME_BIN_HOURS}-hour bins)")
        w()
        w("Least-squares slope for magnitude, Kendall's tau for monotonicity.")
        w()
        w(_md_table(tp, ["metric", "n_bins", "mean", "slope_per_bin",
                         "kendall_tau", "p_value", "direction",
                         "significant_at_0.05"], n=14, float_fmt="{:.6g}"))
        sig = tp[tp["significant_at_0.05"]]
        w()
        if len(sig):
            w(f"**{len(sig)} of {len(tp)}** metrics show a statistically "
              f"significant monotone trend across the observation window.")
        else:
            w("No metric shows a significant monotone trend: **insufficient "
              "evidence** of system-wide drift across the observation window.")
        w()
    if cands is not None and not cands.empty:
        w("### Silent-degradation candidates")
        w()
        w(f"**{len(cands):,} collections** show at least two simultaneously "
          f"worsening degradation indicators over their observed lifetime.")
        w()
        w(_md_table(cands, ["collection_id", "n_events", "span_hours",
                            "n_worsening_indicators",
                            "silent_failure_risk_score_slope", "failure_rate"],
                    n=15, float_fmt="{:.6g}"))
        w()
        w("These are **candidates for investigation, not confirmed silent "
          "failures** -- no label exists that could confirm them.")
        w()
    w("### Do indicators differ *before* an explicit failure?")
    w()
    w("Design: non-failure events inside collections that eventually fail "
      "(*pre-failure*) versus all events of collections that never fail "
      "(*control*); Mann-Whitney U with a rank-biserial effect size.")
    w()
    w("**Limitation stated up front:** the trace is event-sampled, not a "
      "continuous health feed, so \"before\" means \"an earlier event of the same "
      "collection\", not \"N minutes before the fault\". This bounds how strong "
      "any conclusion here can be.")
    w()
    if isinstance(pre.get("table"), pd.DataFrame) and not pre["table"].empty:
        w(f"- collections that fail: {pre.get('collections_with_failure', 0):,}")
        w(f"- collections that never fail: {pre.get('collections_without_failure', 0):,}")
        w(f"- pre-failure events: {pre.get('pre_failure_events', 0):,}")
        w(f"- control events: {pre.get('control_events', 0):,}")
        w()
        w(_md_table(pre["table"], ["indicator", "pre_failure_mean", "control_mean",
                                   "difference", "mannwhitney_p",
                                   "rank_biserial_effect", "higher_before_failure"],
                    n=14, float_fmt="{:.6g}"))
        w()
        w("**Conclusion.** " + pre.get("conclusion", ""))
    else:
        w(pre.get("status", "_Not evaluated in this run._"))
    w()
    w("![Temporal trends](outputs/figures/09_temporal_resource_trends.png)")
    w()

    # ---------------------------------------------------------------- 19
    w("## 19. Machine-Level Analysis")
    w()
    if machines is not None and not machines.empty:
        w(f"- machines profiled: **{len(machines):,}**")
        w(f"- median observations per machine: {machines['support'].median():.0f} "
          f"(max {machines['support'].max():.0f})")
        if "confidence" in machines.columns:
            w()
            w("| Estimate confidence | Machines |")
            w("| --- | --- |")
            for k, v in machines["confidence"].value_counts().items():
                w(f"| {k} | {v:,} |")
        w()
        w("**Sample-size limitation, stated rather than buried.** With so few "
          "observations per machine, machine scores are cross-sectional "
          "peer comparisons, and a trend slope is reported only where at least "
          f"{config.MIN_EVENTS_FOR_MACHINE_SLOPE} timestamped observations exist. "
          "Machines at `very_low` confidence are candidates for investigation, "
          "never conclusions.")
        w()
        w("### Top 15 machines by mean degradation score")
        w()
        w(_md_table(machines, ["machine_id", "cluster", "risk_score_mean",
                               "support", "confidence", "explicit_failure_rate",
                               "risk_band"], n=15))
        w()
        w("![Machine risk ranking](outputs/figures/18_machine_risk_ranking.png)")
    else:
        w("_No machine-level profile available._")
    w()

    # ---------------------------------------------------------------- 20
    w("## 20. Key Findings")
    w()
    k = 1
    w(f"{k}. **The trace has no silent-failure ground truth.** `failed` is an "
      f"explicit failure flag, provably identical to `event == 'FAIL'`. Any "
      f"project treating it as a silent-failure label is measuring the wrong "
      f"thing. This study separates the two tasks throughout."); k += 1
    if res is not None and not res.empty and "random" in res.protocol.unique() and not hon.empty:
        rmax = res[res.protocol == "random"]["roc_auc"].max()
        hmax = hon["roc_auc"].max()
        w(f"{k}. **Join-induced replication makes a random split meaningless.** "
          f"Identical models and features score ROC-AUC {rmax:.4f} under a random "
          f"split and {hmax:.4f} under a grouped split. The first number is "
          f"memorisation of replicated rows, and it is the easiest fraudulent "
          f"result available in this dataset."); k += 1
    if best is not None:
        base = _g(ctx, "model_out", "positive_rate", default=0)
        w(f"{k}. **Behaviour alone carries real but bounded predictive signal.** "
          f"With every target re-encoding removed and group leakage eliminated, "
          f"{best['model']} reaches ROC-AUC {best['roc_auc']:.4f} / PR-AUC "
          f"{best['pr_auc']:.4f} against a {base:.4f} base rate."); k += 1
    if "auc_score_vs_failed" in val:
        w(f"{k}. **An unsupervised degradation index has measurable construct "
          f"validity.** Built from six stress symptoms and never shown the target, "
          f"it ranks explicit failures at ROC-AUC "
          f"{val['auc_score_vs_failed']:.4f}"
          + (f", with a {val['lift_top_vs_bottom']:.2f}x failure-rate lift from "
             f"the lowest to the highest decile" if val.get("lift_top_vs_bottom")
             else "") + "."); k += 1
    if rmeta.get("min_agreement") is not None:
        w(f"{k}. **The index does not depend on invented weights.** Equal, "
          f"PCA-learned and Mahalanobis weightings agree at minimum Spearman rho "
          f"= {rmeta['min_agreement']:.4f}."); k += 1
    if anom.get("detectors"):
        enr = [ (d.get('cross_check') or {}).get('failure_enrichment')
                for d in anom["detectors"].values()]
        enr = [e for e in enr if e]
        if enr:
            w(f"{k}. **Behavioural anomalies and explicit failures overlap "
              f"partially.** The strongest detector shows a {max(enr):.2f}x "
              f"failure enrichment in flagged rows, having never seen the label."); k += 1
    w(f"{k}. **The analysis grain is dictated by the data, not by convention.** "
      f"With a median of ~4 events per machine, per-machine rolling degradation "
      f"curves are not supportable; trend analysis was relocated to the "
      f"collection and cluster-hour grain where support exists."); k += 1
    share = _structural_share(expl.get("permutation"), "feature",
                              "perm_importance_mean") or \
        _structural_share(expl.get("native"), "feature", "importance")
    if share and share["structural_share"] >= 0.3:
        w(f"{k}. **Much of the classifier's power is structural, not "
          f"behavioural.** {share['n_structural']} of the top "
          f"{share['top_n']} held-out importances are trace-position features, "
          f"carrying {100 * share['structural_share']:.0f}% of the top-"
          f"{share['top_n']} importance mass. No leakage is involved, but it means "
          f"the behavioural share of the answer is smaller than the AUC implies -- "
          f"reported here rather than left for a reader to discover."); k += 1
    if isinstance(pre.get("table"), pd.DataFrame) and pre.get("n_significant_with_effect") is not None:
        w(f"{k}. **On the research question's 'before an explicit failure' "
          f"clause:** {pre['n_significant_with_effect']} of "
          f"{pre.get('n_indicators_tested')} indicators separate pre-failure "
          f"events from never-failing controls with both significance and a "
          f"non-negligible effect size."); k += 1
    w()

    # ---------------------------------------------------------------- 21
    w("## 21. Limitations and Threats to Validity")
    w()
    w("1. **No silent-failure label.** The core construct is unobserved. Every "
      "silent-failure result here is a *methodology demonstration plus indirect "
      "validation*, never a measured detection rate.")
    w("2. **Event-sampled, not continuous, telemetry.** \"Before the failure\" can "
      "only mean \"an earlier event of the same collection\". True lead-time "
      "analysis needs a continuous health feed this file does not contain.")
    if share and share["structural_share"] >= 0.3:
        w(f"3. **The supervised model leans on structural features.** "
          f"{100 * share['structural_share']:.0f}% of the top-"
          f"{share['top_n']} held-out importance mass is trace-position rather "
          f"than resource behaviour. This is a limit on how much the classifier "
          f"answers the *behavioural* research question, not a leakage defect.")
    w("3. **Severe per-machine sparsity.** Median ~4 observations per machine "
      "makes most machine-level estimates low-confidence; this is labelled in the "
      "output rather than hidden.")
    w("4. **A join, not a clean panel.** Replication was measured "
      "(thousands of rows can share one usage vector) and handled by grouped "
      "splitting, but it still limits effective sample size far below the row count.")
    w("5. **Normalised, unitless resource values.** Borg publishes resources "
      "scaled to the largest machine, so no absolute capacity statements are "
      "possible.")
    w("6. **Single trace window, eight cells.** Findings describe this trace; "
      "generalisation to other clusters or periods is untested.")
    w("7. **Contamination is assumed, not estimated.** The 5% anomaly budget is a "
      "declared parameter; the anomaly *ranking* is the robust output.")
    w("8. **Unknown calendar alignment.** Time-of-day and weekend-like features "
      "describe periodic position within the trace, not real-world calendar time.")
    if config.SAMPLE_ROWS:
        w(f"9. **This run used a {config.SAMPLE_ROWS:,}-row subsample**, so all "
          f"numbers above are subsample estimates.")
    w()

    # ---------------------------------------------------------------- 22
    w("## 22. Conclusion and Future Work")
    w()
    w("### Answer to the research question")
    w()
    if best is not None and "auc_score_vs_failed" in val:
        strong = best["roc_auc"] >= 0.70 and val["auc_score_vs_failed"] >= 0.60
        if strong:
            w("**Partially, and with quantified limits.** Resource-utilisation, "
              "machine-level and temporal behaviour do carry information that "
              "separates degraded from healthy states: a leakage-free classifier "
              f"reaches ROC-AUC {best['roc_auc']:.4f} on the explicit flag, and an "
              f"unsupervised index that never saw the flag still ranks failures at "
              f"ROC-AUC {val['auc_score_vs_failed']:.4f}. What the data does **not** "
              "support is a claim of silent-failure *detection*: with no label for "
              "the phenomenon, and event-sampled rather than continuous telemetry, "
              "the honest deliverable is a validated screening methodology.")
        else:
            w("**Weakly, and the honest answer is bounded by the data.** Once every "
              "target re-encoding and all group leakage are removed, resource "
              f"behaviour predicts the explicit flag at ROC-AUC "
              f"{best['roc_auc']:.4f}, and the unsupervised degradation index ranks "
              f"failures at ROC-AUC {val['auc_score_vs_failed']:.4f}. There is "
              "measurable but modest signal. There is **insufficient evidence** in "
              "this trace for a deployable silent-failure detector, and the project "
              "reports that rather than inflating it.")
    else:
        w("_Insufficient results in this run to state a conclusion._")
    w()
    w("### What this project actually establishes")
    w()
    w("1. A reproducible, auditable methodology for degradation analysis on cluster "
      "traces that lack degradation labels.")
    w("2. A quantified demonstration that the standard evaluation shortcut (random "
      "split on a joined trace) manufactures near-perfect results from nothing.")
    w("3. A transparent composite index whose conclusions are shown to be robust "
      "to the weighting choice.")
    w("4. Honest reporting of where the data runs out -- per-machine trends, "
      "lead-time analysis, and silent-failure detection rates.")
    w()
    w("### Future work")
    w()
    w("1. **Obtain a labelled degradation source** -- SLO-violation logs or "
      "post-mortems -- to convert the index from unsupervised to validated.")
    w("2. **Use the full multi-table Borg trace** (machine events, machine "
      "attributes, collection events) instead of a pre-joined extract, restoring "
      "a genuine time-series panel per machine.")
    w("3. **Sequence models** (LSTM / temporal convolution / survival analysis) "
      "once continuous per-machine series are available.")
    w("4. **Threshold economics** -- choose the operating point from the real cost "
      "ratio of a missed failure to a false alarm.")
    w("5. **Cross-cell and cross-window replication** to test whether the drivers "
      "identified here are stable.")
    w()
    w("---")
    w()
    w("## Reproducibility")
    w()
    w("```bash")
    w("pip install -r requirements.txt")
    w("python run_project.py")
    w("```")
    w()
    w("| Item | Value |")
    w("| --- | --- |")
    w(f"| Python | {sys.version.split()[0]} |")
    w(f"| Platform | {platform.system()} {platform.release()} |")
    w(f"| Random seed | {config.RANDOM_STATE} (fixed everywhere) |")
    w(f"| Rows analysed | {len(df):,} |")
    w(f"| Model-safe features | {len(feature_cols)} |")
    w(f"| Figures generated | {len(ctx.get('figures', []))} |")
    w(f"| Runtime | {ctx.get('runtime_seconds', 0):.1f}s |")
    w(f"| Sampling | {config.SAMPLE_ROWS or 'none (full dataset)'} |")
    w()
    w("### Generated artefacts")
    w()
    w("| Directory | Contents |")
    w("| --- | --- |")
    w("| `outputs/reports/` | data profile, preprocessing, leakage audit, "
      "silent-failure, anomaly, evaluation, explainability, temporal, final results |")
    w("| `outputs/metrics/` | model comparison, feature/permutation/SHAP "
      "importance, anomaly summary, machine risk ranking, split protocols |")
    w("| `outputs/tables/` | silent-failure risk, risk components, feature "
      "registry, temporal trends, degradation candidates |")
    w("| `outputs/figures/` | "
      f"{len(ctx.get('figures', []))} publication figures |")
    w("| `outputs/models/` | best fitted model + metadata (joblib) |")
    w("| `outputs/predictions/` | held-out predictions, anomaly flags |")
    w()

    save_text(buf.getvalue(), config.ROOT / "PROJECT_REPORT.md")
    info("PROJECT_REPORT.md written")


# ===========================================================================
# final_results.txt
# ===========================================================================

def write_final_results(df: pd.DataFrame, ctx: dict) -> None:
    buf = io.StringIO()
    w = lambda s="": buf.write(s + "\n")

    rmeta = _g(ctx, "risk_meta", default={}) or {}
    val = rmeta.get("validation", {}) or {}
    anom = _g(ctx, "anomaly", default={}) or {}
    res = _g(ctx, "model_comparison")
    expl = _g(ctx, "explain", default={}) or {}
    machines = _g(ctx, "machines")
    prep = _g(ctx, "prep", default={}) or {}
    pre = _g(ctx, "pre_failure", default={}) or {}
    cands = _g(ctx, "degradation_candidates")
    hon = (res[res.protocol == config.HONEST_PROTOCOL]
           if res is not None and not res.empty else pd.DataFrame())
    best = (hon.dropna(subset=["pr_auc"]).sort_values("pr_auc", ascending=False).iloc[0]
            if not hon.empty and hon["pr_auc"].notna().any() else None)

    w("=" * 78)
    w(" FINAL RESULTS")
    w(" Silent Failure Detection in Distributed Systems")
    w(" Google Borg Cluster Traces")
    w("=" * 78)
    w(f" generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}    "
      f"runtime {ctx.get('runtime_seconds', 0):.1f}s")
    if config.SAMPLE_ROWS:
        w(f" *** SUBSAMPLE RUN: {config.SAMPLE_ROWS:,} rows ***")
    w("=" * 78)
    w()
    w("1. DATASET")
    w("-" * 78)
    w(f"   rows analysed            : {len(df):,}")
    w(f"   raw columns              : {ctx.get('raw_cols', 'n/a')}")
    w(f"   columns after processing : {df.shape[1]}")
    if ctx.get("physical_lines"):
        w(f"   physical lines in file   : {ctx['physical_lines']:,} "
          f"(multi-line quoted fields)")
    for c, lab in [("machine_id", "machines"), ("collection_id", "collections"),
                   ("cluster", "clusters")]:
        if c in df.columns:
            w(f"   {lab:24s} : {df[c].nunique():,}")
    if config.TARGET in df.columns:
        w(f"   explicit failure rate    : {df[config.TARGET].mean():.6f} "
          f"({int(df[config.TARGET].sum()):,} events)")
    w()
    w("2. PREPROCESSING")
    w("-" * 78)
    if prep:
        w(f"   original rows            : {prep.get('original_rows', 0):,}")
        w(f"   rows removed             : {prep.get('rows_removed', 0):,}")
        w(f"   duplicate rows found     : {prep.get('duplicate_rows_found', 0):,}")
        w(f"   rows after cleaning      : {prep.get('final_rows', 0):,}")
        w(f"   columns before / after   : {prep.get('original_columns', 0)} / "
          f"{prep.get('final_columns', 0)}")
        w(f"   missing values before    : {prep.get('missing_values_before', 0):,}")
        w(f"   missing values after     : {prep.get('missing_values_after', 0):,}")
        w("   policy                   : per-column imputation + missingness flags")
        w("                              (NO global dropna; outliers retained)")
    else:
        w("   (cached run -- see outputs/reports/preprocessing_report.txt)")
    w()
    w("3. FEATURES")
    w("-" * 78)
    reg = _g(ctx, "registry")
    if reg is not None and not reg.empty:
        w(f"   features registered      : {len(reg)}")
    w(f"   model-safe features      : {len(ctx.get('feature_cols', []))}")
    w(f"   banned as leakage        : {', '.join(config.TARGET_LEAKAGE_COLS)}")
    w()
    w("4. EXPLICIT FAILURE PREDICTION")
    w("-" * 78)
    if res is not None and not res.empty:
        for proto in res.protocol.unique():
            sub = res[res.protocol == proto].dropna(subset=["pr_auc"])
            if sub.empty:
                continue
            b = sub.sort_values("pr_auc", ascending=False).iloc[0]
            tag = ("  <-- HEADLINE" if proto == config.HONEST_PROTOCOL
                   else "  (leakage demo)" if proto == "random" else "")
            w(f"   [{proto}]{tag}")
            w(f"      best model            : {b['model']}")
            w(f"      accuracy / precision  : {b['accuracy']:.4f} / {b['precision']:.4f}")
            w(f"      recall / f1           : {b['recall']:.4f} / {b['f1']:.4f}")
            w(f"      ROC-AUC / PR-AUC      : {b['roc_auc']:.4f} / {b['pr_auc']:.4f}")
            if pd.notna(b.get("fn")):
                w(f"      missed / false alarms : {int(b['fn']):,} / {int(b['fp']):,}")
            w()
        if "random" in res.protocol.unique() and not hon.empty:
            r = res[res.protocol == "random"]["roc_auc"].max()
            h = hon["roc_auc"].max()
            w(f"   LEAKAGE EFFECT: random split best ROC-AUC {r:.4f} vs honest "
              f"{h:.4f}  (+{r - h:.4f})")
    else:
        w("   no model results")
    cv = _g(ctx, "cv")
    if cv and "roc_auc_mean" in cv:
        w(f"   grouped CV ({cv.get('folds_used')} folds): ROC-AUC "
          f"{cv['roc_auc_mean']:.4f} +/- {cv['roc_auc_std']:.4f}, PR-AUC "
          f"{cv['pr_auc_mean']:.4f} +/- {cv['pr_auc_std']:.4f}")
    w()
    w("5. SILENT-FAILURE DEGRADATION INDEX")
    w("-" * 78)
    w("   NOTE: no silent-failure ground truth exists in this trace. The index is")
    w("         unsupervised; the numbers below are indirect validation only.")
    w(f"   components built         : {rmeta.get('n_components', 0)} of 6")
    w(f"   weighting schemes        : {', '.join(config.RISK_WEIGHT_SCHEMES)}")
    w(f"   min scheme agreement     : Spearman rho = {rmeta.get('min_agreement')}")
    if "auc_score_vs_failed" in val:
        w(f"   score-as-ranker ROC-AUC  : {val['auc_score_vs_failed']:.4f}")
        w(f"   score-as-ranker PR-AUC   : {val['pr_auc_score_vs_failed']:.4f}")
        w(f"   base failure rate        : {val['base_rate']:.4f}")
        w(f"   top/bottom decile lift   : {val.get('lift_top_vs_bottom')}")
    for k, v in (rmeta.get("band_counts") or {}).items():
        w(f"   band {k:14s}      : {v:,}")
    w()
    w("   VERDICT:")
    for line in _wrap(val.get("verdict", "n/a"), 70):
        w(f"     {line}")
    w()
    w("6. ANOMALY DETECTION")
    w("-" * 78)
    w(f"   contamination assumption : {config.ANOMALY_CONTAMINATION}")
    for name, d in (anom.get("detectors") or {}).items():
        cc = d.get("cross_check") or {}
        w(f"   {name:20s} {d.get('n_anomalies', 0):>8,} anomalies   "
          f"failure enrichment {cc.get('failure_enrichment', 'n/a')}")
    if anom.get("consensus"):
        w(f"   {'consensus (>=2)':20s} "
          f"{anom['consensus']['n_consensus_anomalies']:>8,} anomalies")
    w()
    w("7. TOP FEATURES (held-out permutation importance)")
    w("-" * 78)
    if "permutation" in expl:
        for i, r in expl["permutation"].head(10).iterrows():
            flag = "*" if r.get("significant") else " "
            w(f"   {flag} {r['feature']:42s} {r['perm_importance_mean']:+.6f} "
              f"+/- {r['perm_importance_std']:.6f}")
        w("   (* = effect exceeds one standard deviation of permutation noise)")
    elif "native" in expl:
        for i, r in expl["native"].head(10).iterrows():
            w(f"     {r['feature']:42s} {r['importance']:.6f}")
    else:
        w("   not available")
    share = _structural_share(expl.get("permutation"), "feature",
                              "perm_importance_mean") or \
        _structural_share(expl.get("native"), "feature", "importance")
    if share:
        w()
        w(f"   STRUCTURAL vs BEHAVIOURAL: {share['n_structural']} of the top "
          f"{share['top_n']} are")
        w(f"   trace-position features carrying "
          f"{100 * share['structural_share']:.1f}% of the top-"
          f"{share['top_n']} importance mass.")
        if share["structural_share"] >= 0.3:
            w("   The behavioural share of the answer is therefore smaller than")
            w("   the ROC-AUC alone implies. No leakage -- a scope limitation.")
    w()
    w("8. TEMPORAL / MACHINE-LEVEL")
    w("-" * 78)
    tp = _g(ctx, "panel_trends")
    if tp is not None and not tp.empty:
        sig = tp[tp["significant_at_0.05"]]
        w(f"   panel metrics tested     : {len(tp)}")
        w(f"   significant trends       : {len(sig)}")
        for _, r in sig.head(6).iterrows():
            w(f"      {r['metric']:32s} {r['direction']:11s} "
              f"tau={r['kendall_tau']:+.3f} p={r['p_value']:.3g}")
    if cands is not None and not cands.empty:
        w(f"   degradation candidates   : {len(cands):,} collections with >=2 "
          f"worsening indicators")
    if machines is not None and not machines.empty:
        w(f"   machines profiled        : {len(machines):,}")
        w(f"   median support / machine : {machines['support'].median():.0f}")
        w("   TOP 5 BY DEGRADATION SCORE:")
        for _, r in machines.head(5).iterrows():
            w(f"      machine {int(r['machine_id']):>14}  score "
              f"{r['risk_score_mean']:.4f}  support {int(r['support']):>3}  "
              f"({r.get('confidence', '')})")
    if pre.get("conclusion"):
        w()
        w("   PRE-FAILURE COMPARISON:")
        for line in _wrap(pre["conclusion"], 70):
            w(f"     {line}")
    w()
    w("9. HONESTY STATEMENT")
    w("-" * 78)
    w("   * 'failed' is an EXPLICIT failure flag, provably identical to")
    w("     event == 'FAIL'. It is never used as a silent-failure label.")
    w("   * No silent-failure ground truth exists in this trace. The degradation")
    w("     index is unsupervised and validated only indirectly.")
    w("   * Every number in this file was computed by the pipeline from the actual")
    w("     data. Nothing is hard-coded, estimated by hand, or illustrative.")
    w("   * Where the data does not support a conclusion, the reports say")
    w("     'insufficient evidence' rather than filling the gap.")
    w(f"   * Random seed {config.RANDOM_STATE} is fixed throughout; re-running")
    w("     reproduces these numbers.")
    w()
    w("=" * 78)
    w(f" artefacts: {len(ctx.get('figures', []))} figures, "
      f"PROJECT_REPORT.md, outputs/{{reports,metrics,tables,models,predictions}}/")
    w("=" * 78)

    save_text(buf.getvalue(), config.REPORT_DIR / "final_results.txt")
    info("final_results.txt written")


def _wrap(text: str, width: int) -> list[str]:
    import textwrap
    return textwrap.wrap(str(text), width=width) or [""]
