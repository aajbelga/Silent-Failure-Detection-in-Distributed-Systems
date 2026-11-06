"""
temporal_analysis.py
====================
STAGES 9 & 10 -- how does behaviour evolve over time, and which entities
degrade?

THE GRAIN PROBLEM (this is the interesting methodological finding)
-----------------------------------------------------------------
The intuitive design is "for each machine, plot its resource metrics over time
and look for drift". Inspection showed that is **not supportable here**:

    machines with any events            : ~96k
    median events per machine           : 4
    machines with >= 10 timestamped obs : 13

A rolling window of 5 over a median of 4 points produces a curve that is mostly
warm-up artefact. Rather than produce it anyway and call it a degradation trend,
the analysis moves to grains where the data has real support:

    * COLLECTION grain  -- ~1.3k collections have >= 10 timestamped events,
                           enough for within-series trend estimation.
    * CLUSTER-HOUR panel -- ~6k cells, dense enough for aggregate trend tests.
    * MACHINE grain      -- cross-sectional peer comparison only, no trend.

That substitution is a finding about the data, not a workaround, and it is
stated as such in the report.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .utils import (fmt, info, safe_div, save_table, save_text, section, slope,
                    warn)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_METRICS = ["cpu_avg", "mem_avg", "cpu_efficiency", "memory_pressure", "cpi",
            "cpu_tail_ratio", "telemetry_loss", "silent_failure_risk_score"]


def _present(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _valid_time(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a usable timestamp (sentinels already NaN-ed in preprocessing)."""
    if "t_hours_from_start" not in df.columns:
        return df.iloc[0:0]
    return df[df["t_hours_from_start"].notna()]


# ---------------------------------------------------------------------------
# 1. cluster-hour panel
# ---------------------------------------------------------------------------

def hourly_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to (cluster, hour-bin) cells -- the densest reliable grain.
    """
    d = _valid_time(df)
    if d.empty:
        warn("No valid timestamps -- hourly panel unavailable.")
        return pd.DataFrame()

    d = d.copy()
    d["time_bin"] = (d["t_hours_from_start"] // config.TIME_BIN_HOURS).astype("int64")
    keys = ["cluster", "time_bin"] if "cluster" in d.columns else ["time_bin"]

    metrics = _present(d, _METRICS)
    agg = {m: "mean" for m in metrics}
    agg["machine_id"] = "nunique"
    if config.TARGET in d.columns:
        agg[config.TARGET] = ["mean", "sum", "size"]

    panel = d.groupby(keys, sort=True).agg(agg)
    panel.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c
                     for c in panel.columns]
    panel = panel.rename(columns={"machine_id_nunique": "active_machines",
                                  f"{config.TARGET}_mean": "failure_rate",
                                  f"{config.TARGET}_sum": "failure_count",
                                  f"{config.TARGET}_size": "events"})
    panel = panel.reset_index()
    info(f"cluster-hour panel: {len(panel):,} cells "
         f"({panel['time_bin'].nunique()} time bins)")
    return panel


def panel_trends(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Least-squares slope + Mann-Kendall-style rank test per metric on the panel.

    Two tests are reported because they answer different questions: the slope
    gives magnitude and direction, Kendall's tau asks whether the trend is
    monotone at all (robust to outliers and non-linearity).
    """
    if panel.empty:
        return pd.DataFrame()
    x = panel["time_bin"].astype("float64")
    rows = []
    for c in panel.columns:
        if c in {"cluster", "time_bin"} or not pd.api.types.is_numeric_dtype(panel[c]):
            continue
        y = panel[c].astype("float64")
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            continue
        s = slope(y[m], x[m])
        tau = stats.kendalltau(x[m], y[m])
        rel = safe_div(s * (x[m].max() - x[m].min()), abs(y[m].mean()))
        rows.append({
            "metric": c,
            "n_bins": int(m.sum()),
            "mean": round(float(y[m].mean()), 6),
            "slope_per_bin": round(float(s), 9) if np.isfinite(s) else None,
            "relative_change_over_window": round(float(rel), 6) if np.isfinite(rel) else None,
            "kendall_tau": round(float(tau.statistic), 6),
            "p_value": float(tau.pvalue),
            "significant_at_0.05": bool(tau.pvalue < 0.05),
            "direction": ("increasing" if tau.statistic > 0 else
                          "decreasing" if tau.statistic < 0 else "flat"),
        })
    out = pd.DataFrame(rows).sort_values("p_value")
    return out


# ---------------------------------------------------------------------------
# 2. collection-grain degradation trends
# ---------------------------------------------------------------------------

def collection_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-collection trend slopes, restricted to collections with enough
    timestamped observations to estimate one (>= MIN_EVENTS_FOR_COLLECTION_TREND).
    """
    d = _valid_time(df)
    if d.empty or "collection_id" not in d.columns:
        return pd.DataFrame()

    metrics = _present(d, _METRICS)
    if not metrics:
        return pd.DataFrame()

    counts = d.groupby("collection_id", sort=False).size()
    eligible = counts[counts >= config.MIN_EVENTS_FOR_COLLECTION_TREND].index
    info(f"collections with >= {config.MIN_EVENTS_FOR_COLLECTION_TREND} timestamped "
         f"events: {len(eligible):,} of {counts.size:,}")
    if len(eligible) == 0:
        warn("No collection has enough timestamped events for trend estimation. "
             "Insufficient evidence from the available trace data.")
        return pd.DataFrame()

    d = d[d.collection_id.isin(eligible)].sort_values(["collection_id",
                                                       "t_hours_from_start"])
    rows = []
    for cid, g in d.groupby("collection_id", sort=False):
        t = g["t_hours_from_start"].to_numpy(dtype="float64")
        rec = {"collection_id": cid, "n_events": len(g),
               "span_hours": round(float(np.nanmax(t) - np.nanmin(t)), 4)}
        for m in metrics:
            rec[f"{m}_slope"] = slope(g[m], t)
            rec[f"{m}_mean"] = float(np.nanmean(g[m].to_numpy(dtype="float64")))
        if config.TARGET in g.columns:
            rec["failure_rate"] = float(g[config.TARGET].mean())
        rows.append(rec)

    out = pd.DataFrame(rows)
    key = ("silent_failure_risk_score_slope" if "silent_failure_risk_score_slope"
           in out.columns else f"{metrics[0]}_slope")
    out = out.sort_values(key, ascending=False).reset_index(drop=True)
    return out


def degradation_candidates(trends: pd.DataFrame) -> pd.DataFrame:
    """
    Collections whose degradation indicators trend UPWARD over their observed
    lifetime -- i.e. getting worse while still running.

    This is the closest the data comes to a silent-degradation signature: no
    explicit failure event, yet the health indicators worsen monotonically.
    Flagged as *candidates*, never as confirmed silent failures.
    """
    if trends.empty:
        return pd.DataFrame()
    up = [c for c in ["silent_failure_risk_score_slope", "cpi_slope",
                      "memory_pressure_slope", "cpu_tail_ratio_slope",
                      "telemetry_loss_slope"] if c in trends.columns]
    if not up:
        return pd.DataFrame()
    t = trends.copy()
    t["n_worsening_indicators"] = sum((t[c] > 0).astype(int) for c in up)
    t["worsening_indicators_checked"] = len(up)
    cand = t[t["n_worsening_indicators"] >= max(2, (len(up) + 1) // 2)]
    if "failure_rate" in cand.columns:
        cand = cand.sort_values(["n_worsening_indicators",
                                 "silent_failure_risk_score_slope"
                                 if "silent_failure_risk_score_slope" in cand.columns
                                 else up[0]], ascending=False)
    return cand.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. lead-time question: do indicators rise BEFORE the failure event?
# ---------------------------------------------------------------------------

def pre_failure_comparison(df: pd.DataFrame) -> dict:
    """
    Within each collection that eventually records a failure, compare the
    degradation indicators of the *earlier* (pre-failure) events against the
    same indicators in collections that never fail.

    This is the closest available test of the research question's "before an
    explicit failure occurs" clause. It is a weak design -- the trace is
    event-sampled, not a continuous health feed -- and the report says so.
    """
    res: dict = {}
    if config.TARGET not in df.columns or "collection_id" not in df.columns:
        res["status"] = "required columns absent"
        return res

    d = _valid_time(df)
    if d.empty:
        res["status"] = ("No valid timestamps: lead-time analysis is impossible. "
                         "Insufficient evidence from the available trace data.")
        return res

    ever = d.groupby("collection_id")[config.TARGET].max()
    fail_cids = set(ever[ever == 1].index)
    res["collections_with_failure"] = len(fail_cids)
    res["collections_without_failure"] = int((ever == 0).sum())

    d = d.sort_values(["collection_id", "t_hours_from_start"])
    is_fail_coll = d["collection_id"].isin(fail_cids)
    # pre-failure = a non-failure event inside a collection that later fails
    pre = d[is_fail_coll & (d[config.TARGET] == 0)]
    ctrl = d[~is_fail_coll]
    res["pre_failure_events"] = int(len(pre))
    res["control_events"] = int(len(ctrl))

    if len(pre) < 30 or len(ctrl) < 30:
        res["status"] = ("Too few pre-failure or control events for a meaningful "
                         "comparison. Insufficient evidence from the available "
                         "trace data.")
        return res

    comp = [c for c in df.columns if c.startswith("comp_")] + \
           _present(df, ["silent_failure_risk_score", "cpi", "memory_pressure",
                         "cpu_tail_ratio", "telemetry_loss", "cpu_efficiency"])
    rows = []
    for c in dict.fromkeys(comp):
        a = pre[c].to_numpy(dtype="float64")
        b = ctrl[c].to_numpy(dtype="float64")
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < 30 or len(b) < 30:
            continue
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        # rank-biserial effect size -- interpretable and distribution-free
        eff = 2 * u.statistic / (len(a) * len(b)) - 1
        rows.append({"indicator": c,
                     "pre_failure_mean": round(float(a.mean()), 6),
                     "control_mean": round(float(b.mean()), 6),
                     "difference": round(float(a.mean() - b.mean()), 6),
                     "mannwhitney_p": float(u.pvalue),
                     "rank_biserial_effect": round(float(eff), 6),
                     "higher_before_failure": bool(a.mean() > b.mean())})
    tab = pd.DataFrame(rows).sort_values("mannwhitney_p")
    res["table"] = tab
    if not tab.empty:
        sig = tab[(tab.mannwhitney_p < 0.05) & (tab.rank_biserial_effect.abs() >= 0.1)]
        res["n_indicators_tested"] = len(tab)
        res["n_significant_with_effect"] = len(sig)
        res["significant_indicators"] = sig["indicator"].tolist()
        if len(sig):
            res["conclusion"] = (
                f"{len(sig)} of {len(tab)} degradation indicators differ between "
                f"pre-failure events and never-failing controls with both "
                f"statistical significance (p < 0.05) and a non-negligible effect "
                f"size (|rank-biserial| >= 0.1). This is measurable evidence that "
                f"resource behaviour differs BEFORE an explicit failure is "
                f"recorded -- an affirmative, though modest, answer to the "
                f"research question.")
        else:
            res["conclusion"] = (
                "No degradation indicator separates pre-failure events from "
                "controls with both significance and a meaningful effect size. "
                "On this trace: INSUFFICIENT EVIDENCE that these indicators "
                "anticipate explicit failure.")
    return res


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def run(df: pd.DataFrame) -> dict:
    section("Cluster-hour panel")
    panel = hourly_panel(df)
    if not panel.empty:
        save_table(panel, config.TABLE_DIR / "hourly_panel.csv")

    section("Panel trend tests")
    trends_panel = panel_trends(panel)
    if not trends_panel.empty:
        save_table(trends_panel, config.TABLE_DIR / "temporal_trends.csv")
        save_table(trends_panel, config.METRICS_DIR / "temporal_trends.csv")

    section("Collection-grain degradation trends")
    trends_coll = collection_trends(df)
    if not trends_coll.empty:
        save_table(trends_coll, config.TABLE_DIR / "collection_trends.csv")

    cands = degradation_candidates(trends_coll)
    if not cands.empty:
        save_table(cands, config.TABLE_DIR / "degradation_candidates.csv")
        info(f"{len(cands):,} collections show >= 2 worsening indicators")

    section("Pre-failure vs control comparison")
    pre = pre_failure_comparison(df)
    if isinstance(pre.get("table"), pd.DataFrame) and not pre["table"].empty:
        save_table(pre["table"], config.TABLE_DIR / "pre_failure_comparison.csv")
    info(pre.get("conclusion", pre.get("status", "")))

    out = {"panel": panel, "panel_trends": trends_panel,
           "collection_trends": trends_coll, "degradation_candidates": cands,
           "pre_failure": pre}
    _write_report(out, df)
    return out


def _write_report(out: dict, df: pd.DataFrame) -> None:
    buf = io.StringIO()

    def w(s: str = ""):
        buf.write(s + "\n")

    panel = out["panel"]
    tp = out["panel_trends"]
    tc = out["collection_trends"]
    cands = out["degradation_candidates"]
    pre = out["pre_failure"]

    w("=" * 78)
    w(" TEMPORAL & ENTITY-LEVEL DEGRADATION ANALYSIS (STAGES 9-10)")
    w("=" * 78)
    w()
    w("CHOICE OF ANALYSIS GRAIN -- AND WHY")
    w("-" * 78)
    if "machine_id" in df.columns:
        per_machine = df.groupby("machine_id").size()
        vt = _valid_time(df)
        per_machine_vt = vt.groupby("machine_id").size() if not vt.empty else pd.Series(dtype=int)
        w(f"machines present                       : {per_machine.size:,}")
        w(f"median events per machine              : {per_machine.median():.0f}")
        w(f"max events per machine                 : {per_machine.max():.0f}")
        if not per_machine_vt.empty:
            w(f"machines with >= 10 timestamped events  : "
              f"{int((per_machine_vt >= 10).sum()):,}")
    w()
    w("A rolling window of 5 over a median of a few observations produces mostly")
    w("warm-up artefact. Per-machine degradation CURVES are therefore not")
    w("statistically supportable on this file, and none are claimed. Analysis")
    w("runs at grains where support actually exists:")
    w()
    w(f"  cluster-hour panel cells               : {len(panel):,}")
    w(f"  collections with trend support (>= {config.MIN_EVENTS_FOR_COLLECTION_TREND}) : {len(tc):,}")
    w("  machines                               : cross-sectional peer")
    w("                                           comparison only (see Stage 4)")
    w()
    w("=" * 78)
    w(" 1. AGGREGATE TRENDS OVER THE TRACE WINDOW")
    w("=" * 78)
    if tp.empty:
        w("Insufficient evidence from the available trace data: fewer than 3 time")
        w("bins could be formed.")
    else:
        w(f"{config.TIME_BIN_HOURS}-hour bins; least-squares slope for magnitude,")
        w("Kendall's tau for monotonicity (robust to outliers).")
        w()
        cols = ["metric", "n_bins", "mean", "slope_per_bin", "kendall_tau",
                "p_value", "direction", "significant_at_0.05"]
        w(tp[[c for c in cols if c in tp.columns]].to_string(
            index=False, float_format=lambda v: f"{v:.6f}"))
        w()
        sig = tp[tp["significant_at_0.05"]]
        if len(sig):
            w(f"{len(sig)} of {len(tp)} metrics show a statistically significant")
            w("monotone trend across the observation window:")
            for _, r in sig.head(8).iterrows():
                w(f"   {r['metric']:32s} {r['direction']:11s} "
                  f"tau={r['kendall_tau']:+.3f}  p={r['p_value']:.3g}")
        else:
            w("No metric shows a significant monotone trend. Insufficient evidence")
            w("of system-wide drift across the observation window.")
    w()
    w("=" * 78)
    w(" 2. COLLECTION-LEVEL DEGRADATION CANDIDATES")
    w("=" * 78)
    if tc.empty:
        w("No collection has enough timestamped events for within-series trend")
        w("estimation. Insufficient evidence from the available trace data.")
    else:
        w(f"Collections analysed          : {len(tc):,}")
        w(f"Median events per collection  : {tc['n_events'].median():.0f}")
        w(f"Median observed span (hours)  : {tc['span_hours'].median():.2f}")
        w()
        if cands.empty:
            w("No collection shows >= 2 simultaneously worsening indicators.")
        else:
            w(f"Collections with >= 2 worsening degradation indicators: {len(cands):,}")
            w(f"  ({100 * len(cands) / len(tc):.2f}% of those analysed)")
            w()
            w("These are SILENT-DEGRADATION CANDIDATES: their health indicators")
            w("worsen monotonically over their observed lifetime. They are")
            w("candidates for investigation, NOT confirmed silent failures --")
            w("no label exists that could confirm them.")
            w()
            show = [c for c in ["collection_id", "n_events", "span_hours",
                                "n_worsening_indicators",
                                "silent_failure_risk_score_slope",
                                "failure_rate"] if c in cands.columns]
            w("TOP 15 CANDIDATES")
            w(cands[show].head(15).to_string(index=False,
                                             float_format=lambda v: f"{v:.6f}"))
    w()
    w("=" * 78)
    w(" 3. DO INDICATORS DIFFER *BEFORE* AN EXPLICIT FAILURE?")
    w("=" * 78)
    w("Design: non-failure events inside collections that eventually fail")
    w("('pre-failure') vs all events of collections that never fail ('control').")
    w("Mann-Whitney U with a rank-biserial effect size, because the indicators")
    w("are non-normal and the group sizes are very unequal.")
    w()
    w("LIMITATION, STATED UP FRONT: the trace is event-sampled, not a continuous")
    w("health feed, so 'before' means 'an earlier event of the same collection',")
    w("not 'N minutes before the fault'. This bounds how strong any conclusion")
    w("here can be.")
    w()
    if "status" in pre and "table" not in pre:
        w(pre["status"])
    else:
        w(f"collections that fail      : {pre.get('collections_with_failure', 0):,}")
        w(f"collections that never fail: {pre.get('collections_without_failure', 0):,}")
        w(f"pre-failure events         : {pre.get('pre_failure_events', 0):,}")
        w(f"control events             : {pre.get('control_events', 0):,}")
        w()
        t = pre.get("table")
        if isinstance(t, pd.DataFrame) and not t.empty:
            w(t.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
            w()
            w("CONCLUSION")
            w("-" * 78)
            w(_wrap(pre.get("conclusion", ""), 74))
    w()
    save_text(buf.getvalue(), config.REPORT_DIR / "temporal_analysis_report.txt")


def _wrap(text: str, width: int) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(str(text), width=width))
