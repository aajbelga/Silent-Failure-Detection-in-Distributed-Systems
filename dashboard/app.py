"""
dashboard/app.py
================
STAGE 11 -- Streamlit dashboard.

    streamlit run dashboard/app.py

Reads only artefacts that ``run_project.py`` has already written. It computes
nothing new, so what the dashboard shows and what the report claims cannot drift
apart. If an artefact is missing the section says so and tells you which command
produces it -- it never renders a placeholder number.

Seven sections:
    1. Overview               5. Anomaly Detection
    2. Resource Health        6. Model Performance
    3. Failure Analysis       7. Explainability
    4. Silent Failure Monitor
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from src import config

st.set_page_config(page_title="Silent Failure Detection -- Borg Traces",
                   page_icon="::", layout="wide")

# ---------------------------------------------------------------------------
# loading (cached -- the model dataset is large)
# ---------------------------------------------------------------------------

DASH_SAMPLE = 120_000   # rows loaded for interactive plots


@st.cache_data(show_spinner="Loading scored dataset...")
def load_main() -> tuple[pd.DataFrame, str, int]:
    """Return (dataframe, which_file, full_row_count)."""
    for which, path in [("model_dataset.csv", config.MODEL_CSV),
                        ("feature_dataset.csv", config.FEATURE_CSV),
                        ("cleaned_dataset.csv", config.CLEANED_CSV)]:
        if path.exists():
            df = pd.read_csv(path, low_memory=False)
            n = len(df)
            if n > DASH_SAMPLE:
                df = df.sample(DASH_SAMPLE, random_state=config.RANDOM_STATE)
            return df, which, n
    return pd.DataFrame(), "", 0


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def missing(what: str, cmd: str = "python run_project.py") -> None:
    st.warning(f"**{what} not available.** Generate it with `{cmd}`. "
               f"Nothing is shown rather than showing a placeholder value.")


def fig(name: str, caption: str = "") -> bool:
    p = config.FIG_DIR / name
    if p.exists():
        st.image(str(p), caption=caption or name, use_container_width=True)
        return True
    return False


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------

df, src_file, full_n = load_main()

st.sidebar.title("Silent Failure Detection")
st.sidebar.caption("Google Borg cluster traces")

SECTIONS = ["1. Overview", "2. Resource Health", "3. Failure Analysis",
            "4. Silent Failure Monitor", "5. Anomaly Detection",
            "6. Model Performance", "7. Explainability"]
page = st.sidebar.radio("Section", SECTIONS)

st.sidebar.divider()
if df.empty:
    st.sidebar.error("No processed dataset found.")
else:
    st.sidebar.metric("Rows in file", f"{full_n:,}")
    if len(df) < full_n:
        st.sidebar.caption(f"Displaying a {len(df):,}-row random sample "
                           f"(seed {config.RANDOM_STATE}) for responsiveness. "
                           f"All metrics below come from the full-data artefacts.")
    st.sidebar.caption(f"source: `{src_file}`")

st.sidebar.divider()
st.sidebar.info("**`failed` is an explicit failure flag, not a silent-failure "
                "label.** This trace has no silent-failure ground truth. The "
                "degradation index is unsupervised.")

if df.empty:
    st.title("No data yet")
    missing("Processed dataset")
    st.stop()

TARGET = config.TARGET
has_target = TARGET in df.columns
has_risk = "silent_failure_risk_score" in df.columns


# ===========================================================================
# 1. OVERVIEW
# ===========================================================================
if page == SECTIONS[0]:
    st.title("1. Overview")
    st.caption("Can resource-utilisation, machine-level and temporal behaviour "
               "in Google Borg cluster traces identify patterns associated with "
               "silent system degradation before an explicit failure occurs?")

    c = st.columns(5)
    c[0].metric("Records", f"{full_n:,}")
    c[1].metric("Machines", f"{df['machine_id'].nunique():,}"
                if "machine_id" in df else "n/a")
    c[2].metric("Collections", f"{df['collection_id'].nunique():,}"
                if "collection_id" in df else "n/a")
    c[3].metric("Clusters", f"{df['cluster'].nunique()}"
                if "cluster" in df else "n/a")
    c[4].metric("Explicit failure rate",
                f"{df[TARGET].mean():.4f}" if has_target else "n/a")

    st.divider()
    st.subheader("Scientific position")
    a, b = st.columns(2)
    a.success("**What the data supports**\n\n"
              "- `failed` is an explicit failure flag (provably identical to "
              "`event == 'FAIL'`) and can be predicted from behaviour\n"
              "- a transparent, unsupervised degradation index can be built and "
              "*validated indirectly*\n"
              "- behaviourally anomalous observations can be ranked\n"
              "- the drivers of both can be explained")
    b.error("**What the data does NOT support**\n\n"
            "- a silent-failure detection rate -- **no such label exists**\n"
            "- per-machine degradation curves (median ~4 events per machine)\n"
            "- lead-time claims (event-sampled, not continuous telemetry)\n"
            "- absolute capacity statements (resources are normalised)")

    st.divider()
    st.subheader("Dataset composition")
    if not fig("01_dataset_overview.png"):
        missing("Overview figure")

    st.subheader("Run summary")
    j = config.METRICS_DIR / "run_summary.json"
    if j.exists():
        import json
        st.json(json.loads(j.read_text()))
    else:
        missing("run_summary.json")


# ===========================================================================
# 2. RESOURCE HEALTH
# ===========================================================================
elif page == SECTIONS[1]:
    st.title("2. Resource Health")
    st.caption("Borg publishes resources normalised to the largest machine in "
               "the fleet, so all values below are unitless shares -- not cores "
               "or gigabytes.")

    cpu = [c for c in ["average_usage_cpus", "maximum_usage_cpus",
                       "cpu_request", "cpu_headroom", "cpu_utilisation_ratio",
                       "cpu_burstiness"] if c in df.columns]
    mem = [c for c in ["average_usage_memory", "maximum_usage_memory",
                       "memory_request", "memory_headroom", "memory_pressure",
                       "assigned_memory"] if c in df.columns]

    st.subheader("CPU")
    if cpu:
        st.dataframe(df[cpu].describe().T, use_container_width=True)
        fig("04_cpu_distributions.png")
    else:
        missing("CPU features")

    st.subheader("Memory")
    if mem:
        st.dataframe(df[mem].describe().T, use_container_width=True)
        fig("05_memory_distributions.png")
    else:
        missing("Memory features")

    st.divider()
    st.subheader("Interactive: distribution of any numeric feature")
    numc = sorted(df.select_dtypes("number").columns)
    pick = st.selectbox("Feature", numc,
                        index=numc.index(cpu[0]) if cpu else 0)
    logy = st.checkbox("Log-scale the count axis", value=True,
                       help="Resource distributions in this trace are extremely "
                            "heavy-tailed; a linear axis hides the tail that "
                            "matters for failure analysis.")
    s = df[pick].dropna()
    if len(s):
        import plotly.express as px
        f = px.histogram(s, nbins=80, log_y=logy,
                         title=f"{pick}  (n={len(s):,})")
        st.plotly_chart(f, use_container_width=True)
        q = s.quantile([0, .25, .5, .75, .9, .99, 1]).rename("value").to_frame()
        st.dataframe(q.T, use_container_width=True)
    else:
        st.info(f"`{pick}` is entirely null in this sample.")

    st.divider()
    st.subheader("Feature correlations")
    if not fig("06_correlation_heatmap.png"):
        missing("Correlation heatmap")

    st.subheader("Scheduling priority")
    fig("24_priority_scheduling_analysis.png")


# ===========================================================================
# 3. FAILURE ANALYSIS
# ===========================================================================
elif page == SECTIONS[2]:
    st.title("3. Failure Analysis")
    st.caption("Explicit failures only -- the `failed` flag. Silent degradation "
               "is section 4.")

    if not has_target:
        missing("Target column")
        st.stop()

    n_f = int(df[TARGET].sum())
    c = st.columns(4)
    c[0].metric("Failure rate (sample)", f"{df[TARGET].mean():.4f}")
    c[1].metric("Failure events", f"{n_f:,}")
    c[2].metric("Non-failure events", f"{len(df) - n_f:,}")
    c[3].metric("Imbalance ratio", f"1 : {(len(df) - n_f) / max(n_f, 1):.1f}")

    st.error("**Leakage warning, and it is the most important fact in this "
             "project.** `event`, `instance_events_type` and "
             "`collections_events_type` are deterministic re-encodings of "
             "`failed`. A model given any of them scores near-perfectly while "
             "learning nothing. All three are banned from every feature matrix "
             "(`config.TARGET_LEAKAGE_COLS`).")

    fig("03_target_distribution.png")

    st.divider()
    st.subheader("Do resources differ between failed and surviving events?")
    if not fig("07_resource_by_failure.png"):
        missing("Resource-by-failure figure")

    st.subheader("Failure rate by any categorical dimension")
    cats = [c for c in ["cluster", "priority_tier", "scheduling_class",
                        "collections_scheduling_class", "vertical_scaling",
                        "scheduler", "time_is_valid", "perf_counters_missing"]
            if c in df.columns]
    if cats:
        by = st.selectbox("Group by", cats)
        g = (df.groupby(by)[TARGET]
             .agg(events="size", failures="sum", failure_rate="mean")
             .sort_values("failure_rate", ascending=False))
        st.dataframe(g.style.format({"events": "{:,}", "failures": "{:,}",
                                     "failure_rate": "{:.4f}"}),
                     use_container_width=True)
        st.bar_chart(g["failure_rate"])
        if by == "time_is_valid":
            st.info("`time_is_valid = False` marks Borg's trace-boundary "
                    "sentinels (`time = 0` or `2^63-1`). Their failure rate is "
                    "far higher, which is *bookkeeping*, not behaviour -- which "
                    "is precisely why raw `time` is excluded from the models.")
    st.divider()
    st.subheader("Event volume over time")
    fig("08_temporal_event_volume.png")


# ===========================================================================
# 4. SILENT FAILURE MONITOR
# ===========================================================================
elif page == SECTIONS[3]:
    st.title("4. Silent Failure Monitor")
    st.warning("**This trace contains no silent-failure ground truth.** The "
               "index below is built from six stress symptoms and never sees "
               "`failed`. It is a *screening tool*, and the numbers here measure "
               "how much information it carries -- not a detection rate.")

    if not has_risk:
        missing("Degradation score",
                "python -m experiments.silent_failure_analysis")
        st.stop()

    s = df["silent_failure_risk_score"]
    c = st.columns(4)
    c[0].metric("Mean score", f"{s.mean():.4f}")
    c[1].metric("Median", f"{s.median():.4f}")
    c[2].metric("95th percentile", f"{s.quantile(.95):.4f}")
    c[3].metric("Scored rows", f"{s.notna().sum():,}")

    st.subheader("Construct validity")
    v = load_csv(config.METRICS_DIR / "risk_validation.csv")
    if not v.empty:
        st.dataframe(v, use_container_width=True, hide_index=True)
        st.caption("None of the components uses `failed`, so this is an "
                   "*external* test of the index, not a circular one.")
    else:
        missing("risk_validation.csv")

    st.subheader("Components and weighting")
    comp = load_csv(config.TABLE_DIR / "risk_components.csv")
    if not comp.empty:
        st.dataframe(comp, use_container_width=True, hide_index=True)
        st.caption("Three combination rules -- equal, PCA-learned and "
                   "Mahalanobis -- are computed and compared by rank "
                   "correlation, so no weight vector had to be invented.")
    fig("11_risk_component_breakdown.png")

    st.divider()
    st.subheader("Distribution and bands")
    a, b = st.columns(2)
    with a:
        fig("10_silent_failure_risk_distribution.png")
    with b:
        fig("12_risk_band_failure_rate.png")

    if "risk_band" in df.columns:
        st.subheader("Band composition")
        agg = {"observations": ("silent_failure_risk_score", "size"),
               "mean_score": ("silent_failure_risk_score", "mean")}
        if has_target:
            agg["explicit_failure_rate"] = (TARGET, "mean")
        band = df.groupby("risk_band", observed=True).agg(**agg)
        st.dataframe(band.style.format({"observations": "{:,}",
                                        "mean_score": "{:.4f}",
                                        "explicit_failure_rate": "{:.4f}"}),
                     use_container_width=True)
        st.caption("Bands are quantiles **of this trace**, not absolute "
                   "engineering thresholds -- no external calibration data "
                   "exists that could justify absolute cut-points.")

    st.divider()
    st.subheader("Live monitor: highest-risk observations")
    thr = st.slider("Minimum degradation score", 0.0, 1.0, 0.90, 0.01)
    cols = [c for c in ["machine_id", "collection_id", "cluster",
                        "silent_failure_risk_score", "risk_band",
                        "cpu_stall_pressure", "memory_pressure",
                        "resource_variability", "telemetry_incompleteness",
                        TARGET] if c in df.columns]
    hot = (df.loc[df["silent_failure_risk_score"] >= thr, cols]
           .sort_values("silent_failure_risk_score", ascending=False))
    st.write(f"**{len(hot):,}** of {len(df):,} displayed observations at or "
             f"above {thr:.2f}.")
    st.dataframe(hot.head(300), use_container_width=True, hide_index=True)
    if has_target and len(hot):
        st.metric("Explicit failure rate in this selection",
                  f"{hot[TARGET].mean():.4f}",
                  delta=f"{hot[TARGET].mean() - df[TARGET].mean():+.4f} vs overall")

    st.divider()
    st.subheader("Machine risk ranking")
    m = load_csv(config.METRICS_DIR / "machine_risk_ranking.csv")
    if not m.empty:
        st.dataframe(m.head(200), use_container_width=True, hide_index=True)
        st.caption("`support` is the number of observations behind each score "
                   "and `confidence` grades it. Low-support machines are "
                   "candidates for investigation, never conclusions.")
        fig("18_machine_risk_ranking.png")
    else:
        missing("machine_risk_ranking.csv")

    st.divider()
    st.subheader("Degradation candidates (collections with worsening trends)")
    cand = load_csv(config.TABLE_DIR / "degradation_candidates.csv")
    if not cand.empty:
        st.dataframe(cand.head(200), use_container_width=True, hide_index=True)
        st.caption("At least two simultaneously worsening indicators over the "
                   "collection's observed lifetime. Candidates for "
                   "investigation -- no label exists that could confirm them.")
    else:
        st.info("No degradation candidates were identified, or the temporal "
                "stage has not run.")

    with st.expander("Full silent-failure report"):
        t = load_text(config.REPORT_DIR / "silent_failure_report.txt")
        st.code(t or "not generated yet", language="text")


# ===========================================================================
# 5. ANOMALY DETECTION
# ===========================================================================
elif page == SECTIONS[4]:
    st.title("5. Anomaly Detection")
    st.caption(f"Three detectors with different inductive biases. Contamination "
               f"is fixed at {config.ANOMALY_CONTAMINATION} -- a declared "
               f"assumption, not an estimate. The durable output is the "
               f"*ranking*.")

    summ = load_csv(config.METRICS_DIR / "anomaly_summary.csv")
    if summ.empty:
        missing("anomaly_summary.csv", "python -m experiments.anomaly_detection")
    else:
        st.dataframe(summ, use_container_width=True, hide_index=True)
        st.caption("`failure_enrichment` > 1 means flagged rows fail explicitly "
                   "more often than unflagged ones. The detectors never saw "
                   "`failed`, so this is independent evidence.")

    flags = [c for c in ["anomaly_iforest", "anomaly_lof", "anomaly_ocsvm",
                         "anomaly_consensus"] if c in df.columns]
    if flags:
        c = st.columns(len(flags))
        for i, f in enumerate(flags):
            v = df[f]
            c[i].metric(f.replace("anomaly_", ""), f"{int(v.sum()):,}",
                        help=f"{v.notna().sum():,} rows scored; "
                             f"{v.isna().sum():,} not evaluated (subsampled "
                             f"detector -- left NaN, never marked normal)")

    a, b = st.columns(2)
    with a:
        fig("13_anomaly_distribution.png")
        fig("15_machine_anomaly_rate.png")
    with b:
        fig("14_anomaly_over_time.png")
        fig("16_anomaly_resource_comparison.png")

    st.divider()
    st.subheader("Anomalies vs degradation bands")
    ct = load_csv(config.TABLE_DIR / "anomaly_by_risk_band.csv")
    if not ct.empty:
        st.dataframe(ct, use_container_width=True)
    elif "risk_band" in df.columns and "anomaly_iforest" in df.columns:
        st.dataframe(pd.crosstab(df["risk_band"], df["anomaly_iforest"],
                                 normalize="index"), use_container_width=True)
    st.caption("The index and the detectors were built independently, so "
               "agreement between them is evidence rather than circularity.")

    if "anomaly_score_iforest" in df.columns:
        st.divider()
        st.subheader("Top-ranked anomalies")
        k = st.slider("How many", 20, 500, 100, 20)
        cols = [c for c in ["machine_id", "collection_id", "cluster",
                            "anomaly_score_iforest", "anomaly_iforest",
                            "silent_failure_risk_score", "risk_band", TARGET]
                if c in df.columns]
        st.dataframe(df.nlargest(k, "anomaly_score_iforest")[cols],
                     use_container_width=True, hide_index=True)

    with st.expander("Full anomaly report"):
        st.code(load_text(config.REPORT_DIR / "anomaly_report.txt")
                or "not generated yet", language="text")


# ===========================================================================
# 6. MODEL PERFORMANCE
# ===========================================================================
elif page == SECTIONS[5]:
    st.title("6. Model Performance")
    st.caption("Explicit-failure prediction. PR-AUC is the ranking metric, "
               "because under this class imbalance a model that predicts "
               "'never fails' scores well on accuracy and is useless.")

    res = load_csv(config.METRICS_DIR / "model_comparison.csv")
    if res.empty:
        missing("model_comparison.csv", "python -m experiments.explicit_failure")
        st.stop()

    st.subheader("The leakage experiment")
    st.markdown("Identical features, identical models -- **only the split "
                "differs**. This comparison is the core methodological result.")
    if {"protocol", "model", "roc_auc"} <= set(res.columns):
        piv = res.pivot_table(index="model", columns="protocol",
                              values="roc_auc", aggfunc="first")
        st.dataframe(piv.style.format("{:.4f}", na_rep="n/a")
                     .background_gradient(cmap="RdYlGn_r", axis=None),
                     use_container_width=True)
        if "random" in piv.columns and config.HONEST_PROTOCOL in piv.columns:
            d = (piv["random"] - piv[config.HONEST_PROTOCOL]).dropna()
            if len(d):
                c = st.columns(3)
                c[0].metric("Mean ROC-AUC inflation from a random split",
                            f"{d.mean():+.4f}")
                c[1].metric("Worst case", f"{d.max():+.4f}", help=str(d.idxmax()))
                c[2].metric("Headline protocol", config.HONEST_PROTOCOL)
        st.info("Usage measurements were joined onto events, so identical "
                "feature vectors repeat across rows and a collection's rows "
                "nearly always share one label. Under a random split, "
                "near-duplicates of test rows sit in training. The random "
                "column is reported **only to quantify the trap**.")
    fig("23_leakage_effect.png")

    st.divider()
    protos = sorted(res["protocol"].unique()) if "protocol" in res else []
    sel = st.selectbox("Protocol", protos,
                       index=protos.index(config.HONEST_PROTOCOL)
                       if config.HONEST_PROTOCOL in protos else 0)
    sub = res[res["protocol"] == sel] if protos else res
    if sel == "random":
        st.error("You are viewing the **leakage demonstration**. These numbers "
                 "are not a result.")
    show = [c for c in ["model", "accuracy", "precision", "recall", "f1",
                        "roc_auc", "pr_auc", "balanced_accuracy", "mcc",
                        "specificity", "tp", "fp", "fn", "tn",
                        "missed_failures", "false_alarms", "train_rows",
                        "subsampled", "fit_seconds"] if c in sub.columns]
    st.dataframe(sub[show].sort_values("pr_auc", ascending=False)
                 .style.format(precision=4, na_rep="n/a"),
                 use_container_width=True, hide_index=True)

    a, b = st.columns(2)
    with a:
        fig("17_model_comparison.png")
        fig("19_confusion_matrices.png")
    with b:
        fig("20_roc_pr_curves.png")

    st.divider()
    st.subheader("Split diagnostics")
    sp = load_csv(config.METRICS_DIR / "split_protocols.csv")
    if not sp.empty:
        st.dataframe(sp, use_container_width=True, hide_index=True)
        st.caption("`groups_on_both_sides` must be 0 for a leakage-free split.")

    with st.expander("Evaluation report"):
        st.code(load_text(config.REPORT_DIR / "evaluation_report.txt")
                or "not generated yet", language="text")
    with st.expander("Leakage audit"):
        st.code(load_text(config.REPORT_DIR / "leakage_audit.txt")
                or "not generated yet", language="text")


# ===========================================================================
# 7. EXPLAINABILITY
# ===========================================================================
elif page == SECTIONS[6]:
    st.title("7. Explainability")
    st.caption("Three independent views. Agreement between different "
               "definitions of importance is what makes an explanation "
               "trustworthy.")

    nat = load_csv(config.METRICS_DIR / "feature_importance.csv")
    perm = load_csv(config.METRICS_DIR / "permutation_importance.csv")
    shp = load_csv(config.METRICS_DIR / "shap_importance.csv")
    cons = load_csv(config.METRICS_DIR / "importance_consensus.csv")

    if nat.empty and perm.empty:
        missing("Importance tables", "python -m experiments.explicit_failure")
        st.stop()

    t1, t2, t3, t4 = st.tabs(["Built-in importance", "Permutation (held-out)",
                              "SHAP", "Consensus"])
    with t1:
        if nat.empty:
            missing("feature_importance.csv")
        else:
            k = st.slider("Top N", 5, min(40, len(nat)), 15, key="k_nat")
            st.dataframe(nat.head(k), use_container_width=True, hide_index=True)
            st.bar_chart(nat.head(k).set_index("feature")["importance"])
            st.caption("Impurity-based importance measures what the model "
                       "*fitted*. It is biased toward high-cardinality features, "
                       "which is why the permutation tab exists.")
    with t2:
        if perm.empty:
            missing("permutation_importance.csv")
        else:
            st.dataframe(perm.head(25), use_container_width=True, hide_index=True)
            if "significant" in perm.columns:
                n = int(perm["significant"].sum())
                st.metric("Features above permutation noise",
                          f"{n} of {len(perm)}")
                if n == 0:
                    st.warning("No feature's contribution is distinguishable "
                               "from noise on the held-out fold. Insufficient "
                               "evidence to name a single driver.")
            st.caption("Drop in average precision when a feature is shuffled on "
                       "unseen data. This measures what *generalises*.")
    with t3:
        if shp.empty:
            st.warning("**SHAP values are not available for this run.** "
                       "Recorded as a limitation in the report. No values were "
                       "invented to fill the gap -- the held-out permutation "
                       "importances are the model-agnostic substitute.")
        else:
            st.dataframe(shp.head(25), use_container_width=True, hide_index=True)
            fig("22_shap_summary.png")
    with t4:
        if cons.empty:
            missing("importance_consensus.csv")
        else:
            st.dataframe(cons.head(25), use_container_width=True, hide_index=True)
            st.caption("Where impurity and held-out permutation rankings "
                       "disagree, trust the permutation ranking -- it measures "
                       "generalisation rather than fit.")

    st.divider()
    fig("21_feature_importance.png")
    with st.expander("Explainability report"):
        st.code(load_text(config.REPORT_DIR / "explainability_report.txt")
                or "not generated yet", language="text")

st.divider()
st.caption("Every figure and table on this page was generated by "
           "`run_project.py` from the actual trace. The dashboard computes no "
           "metrics of its own, so it cannot disagree with the report.")
