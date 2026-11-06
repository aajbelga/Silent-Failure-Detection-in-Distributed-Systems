"""
visualization.py
================
STAGE 12 -- every figure in the report.

Design rules followed throughout:
  * Each function is independent and DEFENSIVE: if the columns it needs are
    absent it logs a skip and returns None instead of crashing the pipeline.
  * Nothing is plotted that the data cannot support. Where a panel would be
    misleading (e.g. a per-machine trend line over 4 points) the figure says so
    in its own annotation rather than drawing a confident-looking curve.
  * Large scatter plots are subsampled for legibility, and the subsample size is
    written into the title so no reader mistakes it for the full trace.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")                      # headless: no display needed
import matplotlib.pyplot as plt

from . import config
from .utils import info, warn

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="notebook")
    HAVE_SNS = True
except Exception:                                               # pragma: no cover
    HAVE_SNS = False

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": config.FIG_DPI,
    "savefig.bbox": "tight", "figure.autolayout": False,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10, "font.size": 9,
})

C_OK, C_BAD, C_NEU, C_ALT = "#2c7fb8", "#d7301f", "#7f7f7f", "#4daf4a"
_MADE: list[str] = []


def _save(fig, name: str):
    path = config.FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    _MADE.append(name)
    info(f"figure -> {name}")
    return path


def _skip(name: str, why: str):
    warn(f"figure {name} skipped: {why}")
    return None


def _sub(df: pd.DataFrame, n: int = 30_000) -> pd.DataFrame:
    return df.sample(n=n, random_state=config.RANDOM_STATE) if len(df) > n else df


def figures_made() -> list[str]:
    return list(_MADE)


# ===========================================================================
# 01-03  dataset overview
# ===========================================================================

def fig_dataset_overview(df: pd.DataFrame, profile: dict | None = None):
    name = "01_dataset_overview.png"
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Dataset overview: Google Borg cluster trace", fontsize=14,
                 fontweight="bold")

    dtypes = df.dtypes.astype(str).value_counts()
    ax[0, 0].bar(dtypes.index.astype(str), dtypes.values, color=C_OK)
    ax[0, 0].set_title(f"Column dtypes ({df.shape[1]} columns)")
    ax[0, 0].set_ylabel("columns")
    ax[0, 0].tick_params(axis="x", rotation=30)

    miss = (df.isna().mean() * 100).sort_values(ascending=False).head(15)
    if (miss > 0).any():
        ax[0, 1].barh(miss.index[::-1], miss.values[::-1], color=C_BAD)
        ax[0, 1].set_xlabel("% missing")
        ax[0, 1].set_title("Most-incomplete columns")
    else:
        ax[0, 1].text(0.5, 0.5, "no missing values remain\n(after Stage 2)",
                      ha="center", va="center")
        ax[0, 1].set_axis_off()

    if config.TARGET in df.columns:
        vc = df[config.TARGET].value_counts().sort_index()
        bars = ax[1, 0].bar([str(i) for i in vc.index], vc.values,
                            color=[C_OK, C_BAD][: len(vc)])
        ax[1, 0].set_title(f"Explicit failure flag '{config.TARGET}'")
        ax[1, 0].set_ylabel("rows")
        for b, v in zip(bars, vc.values):
            ax[1, 0].text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v/len(df):.1%})",
                          ha="center", va="bottom", fontsize=8)

    txt = [f"rows: {len(df):,}", f"columns: {df.shape[1]}"]
    for c, lab in [("machine_id", "machines"), ("collection_id", "collections"),
                   ("cluster", "clusters"), ("user", "users")]:
        if c in df.columns:
            txt.append(f"{lab}: {df[c].nunique():,}")
    if config.TARGET in df.columns:
        txt.append(f"failure rate: {df[config.TARGET].mean():.4f}")
    ax[1, 1].text(0.02, 0.95, "\n".join(txt), va="top", family="monospace",
                  fontsize=10, transform=ax[1, 1].transAxes)
    ax[1, 1].set_title("Scale")
    ax[1, 1].set_axis_off()
    fig.tight_layout()
    return _save(fig, name)


def fig_missing_values(raw_missing: pd.DataFrame):
    name = "02_missing_values.png"
    if raw_missing is None or raw_missing.empty:
        return _skip(name, "no missing-value table")
    d = raw_missing.copy()
    col = "missing_pct" if "missing_pct" in d.columns else d.columns[-1]
    d = d[d[col] > 0].sort_values(col, ascending=False).head(20)
    if d.empty:
        return _skip(name, "raw data had no missing values")
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(d))))
    lab = d["column"] if "column" in d.columns else d.index.astype(str)
    ax.barh(list(lab)[::-1], list(d[col])[::-1], color=C_BAD)
    ax.set_xlabel("% missing in the RAW file")
    ax.set_title("Missingness before preprocessing\n"
                 "(handled per-column; no global dropna)", loc="left")
    fig.tight_layout()
    return _save(fig, name)


def fig_target_distribution(df: pd.DataFrame):
    name = "03_target_distribution.png"
    if config.TARGET not in df.columns:
        return _skip(name, "no target column")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    y = df[config.TARGET]

    vc = y.value_counts().sort_index()
    ax[0].pie(vc.values, labels=[f"no failure\n{vc.get(0,0):,}",
                                 f"failure\n{vc.get(1,0):,}"][: len(vc)],
              autopct="%1.2f%%", colors=[C_OK, C_BAD][: len(vc)],
              startangle=90, wedgeprops={"edgecolor": "w"})
    ax[0].set_title("Class balance")

    if "cluster" in df.columns:
        by = df.groupby("cluster")[config.TARGET].agg(["mean", "size"])
        by = by.sort_values("mean", ascending=False)
        ax[1].bar(by.index.astype(str), by["mean"], color=C_BAD)
        ax[1].axhline(y.mean(), ls="--", color=C_NEU,
                      label=f"overall {y.mean():.3f}")
        ax[1].set_title("Failure rate by cluster")
        ax[1].set_ylabel("failure rate")
        ax[1].legend(fontsize=8)
        ax[1].tick_params(axis="x", rotation=45)

    if "priority_tier" in df.columns:
        by = df.groupby("priority_tier")[config.TARGET].agg(["mean", "size"])
        by = by[by["size"] >= 30]
        if not by.empty:
            ax[2].bar(by.index.astype(str), by["mean"], color="#756bb1")
            ax[2].set_title("Failure rate by priority tier\n(tiers with >=30 rows)")
            ax[2].set_xlabel("priority tier (ordinal)")
            ax[2].set_ylabel("failure rate")
    else:
        ax[2].set_axis_off()
    fig.tight_layout()
    return _save(fig, name)


# ===========================================================================
# 04-07  resource behaviour
# ===========================================================================

def _dist_panel(ax, s: pd.Series, title: str, color=C_OK, log=False):
    v = s.replace([np.inf, -np.inf], np.nan).dropna()
    if v.empty:
        ax.set_axis_off()
        return
    ax.hist(v, bins=60, color=color, alpha=0.85)
    ax.axvline(v.median(), ls="--", color="k", lw=1,
               label=f"median {v.median():.4g}")
    ax.set_title(title)
    ax.legend(fontsize=7)
    if log:
        ax.set_yscale("log")


def fig_cpu_distributions(df: pd.DataFrame):
    name = "04_cpu_distributions.png"
    cols = [c for c in ["cpu_avg", "cpu_max", "cpu_efficiency", "cpu_burstiness",
                        "cpu_dist_cv", "cpu_tail_ratio"] if c in df.columns]
    if not cols:
        return _skip(name, "no CPU features")
    n = len(cols)
    fig, axes = plt.subplots((n + 2) // 3, 3, figsize=(14, 3.6 * ((n + 2) // 3)))
    axes = np.atleast_1d(axes).ravel()
    for a, c in zip(axes, cols):
        _dist_panel(a, df[c], c, C_OK, log=True)
    for a in axes[n:]:
        a.set_axis_off()
    fig.suptitle("CPU behaviour distributions (log-count axis)", fontweight="bold")
    fig.tight_layout()
    return _save(fig, name)


def fig_memory_distributions(df: pd.DataFrame):
    name = "05_memory_distributions.png"
    cols = [c for c in ["mem_avg", "mem_max", "memory_pressure", "cache_ratio",
                        "mem_utilization_ratio", "memory_access_intensity"]
            if c in df.columns]
    if not cols:
        return _skip(name, "no memory features")
    n = len(cols)
    fig, axes = plt.subplots((n + 2) // 3, 3, figsize=(14, 3.6 * ((n + 2) // 3)))
    axes = np.atleast_1d(axes).ravel()
    for a, c in zip(axes, cols):
        _dist_panel(a, df[c], c, "#31a354", log=True)
    for a in axes[n:]:
        a.set_axis_off()
    fig.suptitle("Memory behaviour distributions (log-count axis)", fontweight="bold")
    fig.tight_layout()
    return _save(fig, name)


def fig_correlation_heatmap(df: pd.DataFrame, feature_cols: list[str]):
    name = "06_correlation_heatmap.png"
    cols = [c for c in feature_cols if c in df.columns][:28]
    if len(cols) < 3:
        return _skip(name, "too few numeric features")
    sub = _sub(df, 60_000)[cols].replace([np.inf, -np.inf], np.nan)
    corr = sub.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    if HAVE_SNS:
        sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                    square=True, linewidths=0.3, cbar_kws={"shrink": 0.7}, ax=ax)
    else:
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols)), cols, rotation=90)
        ax.set_yticks(range(len(cols)), cols)
        fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title(f"Feature correlation ({len(cols)} model-safe features, "
                 f"{len(sub):,}-row sample)")
    fig.tight_layout()
    return _save(fig, name)


def fig_resource_by_failure(df: pd.DataFrame):
    name = "07_resource_by_failure.png"
    if config.TARGET not in df.columns:
        return _skip(name, "no target")
    cols = [c for c in ["cpu_avg", "mem_avg", "cpu_efficiency", "memory_pressure",
                        "cpi", "cpu_tail_ratio", "telemetry_loss",
                        "resource_intensity"] if c in df.columns][:8]
    if not cols:
        return _skip(name, "no comparison features")
    sub = _sub(df, 80_000)
    n = len(cols)
    fig, axes = plt.subplots((n + 3) // 4, 4, figsize=(16, 3.8 * ((n + 3) // 4)))
    axes = np.atleast_1d(axes).ravel()
    for a, c in zip(axes, cols):
        d = [sub.loc[sub[config.TARGET] == k, c].replace([np.inf, -np.inf], np.nan)
             .dropna() for k in (0, 1)]
        if min(len(x) for x in d) < 10:
            a.set_axis_off()
            continue
        bp = a.boxplot(d, labels=["no fail", "fail"], showfliers=False,
                       patch_artist=True)
        for patch, col in zip(bp["boxes"], [C_OK, C_BAD]):
            patch.set_facecolor(col)
            patch.set_alpha(0.7)
        a.set_title(c, fontsize=10)
    for a in axes[n:]:
        a.set_axis_off()
    fig.suptitle(f"Resource behaviour by explicit failure outcome "
                 f"({len(sub):,}-row sample, outliers hidden)", fontweight="bold")
    fig.tight_layout()
    return _save(fig, name)


# ===========================================================================
# 08-09  temporal
# ===========================================================================

def fig_temporal_volume(df: pd.DataFrame, panel: pd.DataFrame):
    name = "08_temporal_event_volume.png"
    if panel is None or panel.empty:
        return _skip(name, "no hourly panel")
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    if "cluster" in panel.columns:
        for cl, g in panel.groupby("cluster"):
            g = g.sort_values("time_bin")
            ax[0].plot(g.time_bin, g.get("events", pd.Series(dtype=float)),
                       lw=1, label=f"cluster {cl}")
        ax[0].legend(fontsize=7, ncol=4)
    else:
        g = panel.sort_values("time_bin")
        ax[0].plot(g.time_bin, g["events"], lw=1, color=C_OK)
    ax[0].set_ylabel("events per bin")
    ax[0].set_title(f"Event volume over the trace window "
                    f"({config.TIME_BIN_HOURS}h bins)")

    if "failure_rate" in panel.columns:
        g = panel.groupby("time_bin")["failure_rate"].mean().sort_index()
        ax[1].plot(g.index, g.values, lw=1, color=C_BAD)
        ax[1].axhline(g.mean(), ls="--", color=C_NEU, label=f"mean {g.mean():.3f}")
        ax[1].set_ylabel("failure rate")
        ax[1].legend(fontsize=8)
        ax[1].set_title("Explicit failure rate over time")
    ax[1].set_xlabel(f"hours from first valid timestamp "
                     f"({config.TIME_BIN_HOURS}h bins)")
    fig.tight_layout()
    return _save(fig, name)


def fig_temporal_resource_trends(panel: pd.DataFrame, trends: pd.DataFrame):
    name = "09_temporal_resource_trends.png"
    if panel is None or panel.empty:
        return _skip(name, "no hourly panel")
    metrics = [c for c in ["cpu_avg", "mem_avg", "cpi", "memory_pressure",
                           "cpu_tail_ratio", "silent_failure_risk_score"]
               if c in panel.columns]
    if not metrics:
        return _skip(name, "no panel metrics")
    n = len(metrics)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(14, 3.2 * ((n + 1) // 2)),
                             sharex=True)
    axes = np.atleast_1d(axes).ravel()
    tinfo = {}
    if trends is not None and not trends.empty:
        tinfo = trends.set_index("metric").to_dict("index")
    for a, m in zip(axes, metrics):
        g = panel.groupby("time_bin")[m].mean().sort_index()
        a.plot(g.index, g.values, lw=1.2, color=C_OK)
        if len(g) >= 3:
            z = np.polyfit(g.index.astype(float), g.values, 1)
            a.plot(g.index, np.polyval(z, g.index.astype(float)), ls="--",
                   color=C_BAD, lw=1.2, label="OLS trend")
        t = tinfo.get(m, {})
        sig = t.get("significant_at_0.05")
        ttl = m + (f"   tau={t['kendall_tau']:+.3f}, p={t['p_value']:.2g}"
                   if "kendall_tau" in t else "")
        a.set_title(ttl + ("  [significant]" if sig else "  [not significant]"
                           if sig is not None else ""), fontsize=9)
        a.legend(fontsize=7)
    for a in axes[n:]:
        a.set_axis_off()
    fig.suptitle("Aggregate resource trends over the trace window\n"
                 "(cluster-hour panel; per-machine curves are NOT supportable "
                 "at this sample size)", fontweight="bold")
    fig.tight_layout()
    return _save(fig, name)


# ===========================================================================
# 10-12  silent-failure risk
# ===========================================================================

def fig_risk_distribution(df: pd.DataFrame, meta: dict):
    name = "10_silent_failure_risk_distribution.png"
    if "silent_failure_risk_score" not in df.columns:
        return _skip(name, "no risk score")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    s = df["silent_failure_risk_score"]
    ax[0].hist(s.dropna(), bins=60, color="#feb24c", edgecolor="none")
    for lo, _hi, lab in config.RISK_BANDS:
        ax[0].axvline(s.quantile(lo), ls="--", lw=1, color=C_NEU)
    ax[0].set_title("Composite degradation score\n(rank-normalised, bands = quantiles)")
    ax[0].set_xlabel("silent_failure_risk_score")

    if "risk_band" in df.columns:
        vc = df["risk_band"].value_counts().reindex(
            ["NORMAL", "LOW_RISK", "MEDIUM_RISK", "HIGH_RISK"]).dropna()
        ax[1].bar(vc.index.astype(str), vc.values,
                  color=["#31a354", "#addd8e", "#fec44f", C_BAD][: len(vc)])
        ax[1].set_title("Observations per risk band")
        ax[1].tick_params(axis="x", rotation=20)
        for i, v in enumerate(vc.values):
            ax[1].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)

    schemes = [c for c in ["risk_equal", "risk_pca", "risk_mahalanobis"]
               if c in df.columns]
    if len(schemes) >= 2:
        sub = _sub(df, 20_000)
        ax[2].scatter(sub[schemes[0]], sub[schemes[1]], s=3, alpha=0.15, color=C_OK)
        rho = meta.get("spearman_agreement", {})
        k = f"{schemes[0]}__vs__{schemes[1]}"
        ax[2].set_xlabel(schemes[0])
        ax[2].set_ylabel(schemes[1])
        ax[2].set_title(f"Weighting-scheme agreement\nSpearman rho = "
                        f"{rho.get(k, float('nan')):.4f} ({len(sub):,}-row sample)")
    fig.suptitle("Silent-failure degradation index", fontweight="bold")
    fig.tight_layout()
    return _save(fig, name)


def fig_risk_components(df: pd.DataFrame, meta: dict):
    name = "11_risk_component_breakdown.png"
    comps = [c for c in df.columns if c.startswith("comp_")]
    if not comps:
        return _skip(name, "no risk components")
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))

    w_eq = meta.get("equal_weights", {}) or {}
    w_pca = meta.get("pca_weights", {}) or {}
    names = [c[5:] for c in comps]
    x = np.arange(len(names))
    ax[0].bar(x - 0.2, [w_eq.get(n, 0) for n in names], 0.4, label="equal", color=C_OK)
    ax[0].bar(x + 0.2, [w_pca.get(n, 0) for n in names], 0.4, label="PCA PC1", color=C_ALT)
    ax[0].set_xticks(x, names, rotation=35, ha="right", fontsize=8)
    ax[0].set_ylabel("weight")
    ax[0].set_title("Component weights: indifference vs learned\n"
                    "(no weight was hand-chosen)")
    ax[0].legend(fontsize=8)

    sub = _sub(df, 60_000)
    corr = sub[comps].corr()
    im = ax[1].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax[1].set_xticks(range(len(names)), names, rotation=90, fontsize=7)
    ax[1].set_yticks(range(len(names)), names, fontsize=7)
    ax[1].set_title("Component intercorrelation")
    fig.colorbar(im, ax=ax[1], shrink=0.75)

    if config.TARGET in df.columns:
        rows = []
        for c, n in zip(comps, names):
            rows.append((n, df.loc[df[config.TARGET] == 0, c].mean(),
                         df.loc[df[config.TARGET] == 1, c].mean()))
        r = pd.DataFrame(rows, columns=["comp", "no_fail", "fail"])
        xx = np.arange(len(r))
        ax[2].barh(xx - 0.2, r.no_fail, 0.4, color=C_OK, label="no failure")
        ax[2].barh(xx + 0.2, r.fail, 0.4, color=C_BAD, label="failure")
        ax[2].set_yticks(xx, r.comp, fontsize=8)
        ax[2].set_xlabel("mean rank-normalised value")
        ax[2].set_title("Components vs explicit failure\n(components never see the target)")
        ax[2].legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, name)


def fig_risk_validation(df: pd.DataFrame, val: dict):
    name = "12_risk_band_failure_rate.png"
    if not val or "failure_rate_by_decile" not in val:
        return _skip(name, "no validation result")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    d = val["failure_rate_by_decile"]
    ks = sorted(d)
    ax[0].bar([str(k) for k in ks], [d[k] for k in ks], color="#fd8d3c")
    if config.TARGET in df.columns:
        ax[0].axhline(df[config.TARGET].mean(), ls="--", color=C_NEU,
                      label=f"base rate {df[config.TARGET].mean():.4f}")
        ax[0].legend(fontsize=8)
    ax[0].set_xlabel("degradation-score decile (0 = lowest risk)")
    ax[0].set_ylabel("explicit failure rate")
    tau = val.get("decile_trend_kendall_tau")
    ax[0].set_title("Failure rate rises with degradation score"
                    + (f"\nKendall tau = {tau:+.3f}, p = {val['decile_trend_p']:.2g}"
                       if tau is not None else ""))

    if "risk_band" in df.columns and config.TARGET in df.columns:
        by = df.groupby("risk_band", observed=True)[config.TARGET].agg(["mean", "size"])
        by = by.reindex(["NORMAL", "LOW_RISK", "MEDIUM_RISK", "HIGH_RISK"]).dropna()
        ax[1].bar(by.index.astype(str), by["mean"],
                  color=["#31a354", "#addd8e", "#fec44f", C_BAD][: len(by)])
        for i, (m, s) in enumerate(zip(by["mean"], by["size"])):
            ax[1].text(i, m, f"{m:.4f}\nn={int(s):,}", ha="center", va="bottom",
                       fontsize=8)
        ax[1].set_ylabel("explicit failure rate")
        auc = val.get("auc_score_vs_failed")
        ax[1].set_title("Failure rate by risk band"
                        + (f"\nscore-as-ranker ROC-AUC = {auc:.4f}" if auc else ""))
        ax[1].tick_params(axis="x", rotation=20)
    fig.suptitle("Construct validity: the unsupervised score vs the independent "
                 "failure flag", fontweight="bold")
    fig.tight_layout()
    return _save(fig, name)


# ===========================================================================
# 13-16  anomalies
# ===========================================================================

def fig_anomaly_distribution(df: pd.DataFrame, res: dict):
    name = "13_anomaly_distribution.png"
    if "anomaly_score_iforest" not in df.columns:
        return _skip(name, "no anomaly scores")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    s = df["anomaly_score_iforest"]
    thr = s[df.anomaly_iforest == 1].min() if (df.anomaly_iforest == 1).any() else None
    ax[0].hist(s, bins=70, color=C_OK)
    if thr is not None:
        ax[0].axvline(thr, ls="--", color=C_BAD,
                      label=f"{config.ANOMALY_CONTAMINATION:.0%} cut-off")
        ax[0].legend(fontsize=8)
    ax[0].set_yscale("log")
    ax[0].set_title("Isolation Forest anomaly score (full data)")

    counts, labels = [], []
    for det, key in [("Isolation Forest", "anomaly_iforest"),
                     ("LOF", "anomaly_lof"), ("One-Class SVM", "anomaly_ocsvm"),
                     ("Consensus >=2", "anomaly_consensus")]:
        if key in df.columns:
            labels.append(det)
            counts.append(int(np.nansum(df[key])))
    ax[1].bar(labels, counts, color=[C_OK, C_ALT, "#756bb1", C_BAD][: len(labels)])
    ax[1].set_title("Anomalies flagged per detector")
    ax[1].tick_params(axis="x", rotation=20)
    for i, v in enumerate(counts):
        ax[1].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)

    rows = []
    for det, d in res.get("detectors", {}).items():
        cc = d.get("cross_check") or {}
        if cc.get("failure_rate_in_anomalies") is not None:
            rows.append((det, cc["failure_rate_in_anomalies"],
                         cc.get("failure_rate_in_normals", 0)))
    if rows:
        r = pd.DataFrame(rows, columns=["det", "anom", "norm"])
        x = np.arange(len(r))
        ax[2].bar(x - 0.2, r.anom, 0.4, color=C_BAD, label="flagged anomalous")
        ax[2].bar(x + 0.2, r.norm, 0.4, color=C_OK, label="not flagged")
        ax[2].set_xticks(x, r.det, rotation=20, fontsize=8)
        ax[2].set_ylabel("explicit failure rate")
        ax[2].set_title("Failure rate inside vs outside anomalies\n"
                        "(detectors never saw the target)")
        ax[2].legend(fontsize=8)
    else:
        ax[2].set_axis_off()
    fig.suptitle("Unsupervised anomaly detection", fontweight="bold")
    fig.tight_layout()
    return _save(fig, name)


def fig_anomaly_over_time(df: pd.DataFrame):
    name = "14_anomaly_over_time.png"
    if "anomaly_iforest" not in df.columns or "t_hours_from_start" not in df.columns:
        return _skip(name, "no anomaly flags or timestamps")
    d = df[df["t_hours_from_start"].notna()].copy()
    if d.empty:
        return _skip(name, "no valid timestamps")
    d["bin"] = (d["t_hours_from_start"] // config.TIME_BIN_HOURS).astype("int64")
    g = d.groupby("bin").agg(rate=("anomaly_iforest", "mean"),
                             n=("anomaly_iforest", "size"),
                             score=("anomaly_score_iforest", "mean"))
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    ax[0].plot(g.index, g.rate, lw=1.2, color=C_BAD)
    ax[0].axhline(config.ANOMALY_CONTAMINATION, ls="--", color=C_NEU,
                  label=f"expected {config.ANOMALY_CONTAMINATION:.0%}")
    ax[0].set_ylabel("anomaly rate")
    ax[0].legend(fontsize=8)
    ax[0].set_title(f"Anomaly rate over the trace window ({config.TIME_BIN_HOURS}h bins)")
    ax[1].plot(g.index, g.score, lw=1.2, color=C_OK)
    ax[1].set_ylabel("mean anomaly score")
    ax[1].set_xlabel("hours from first valid timestamp")
    ax[1].set_title("Mean anomaly score over time")
    fig.tight_layout()
    return _save(fig, name)


def fig_machine_anomaly_rate(df: pd.DataFrame):
    name = "15_machine_anomaly_rate.png"
    if "anomaly_iforest" not in df.columns or "machine_id" not in df.columns:
        return _skip(name, "no anomaly flags or machine ids")
    g = df.groupby("machine_id").agg(rate=("anomaly_iforest", "mean"),
                                     n=("anomaly_iforest", "size"))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].hist(g["rate"], bins=40, color="#756bb1")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("per-machine anomaly rate")
    ax[0].set_title(f"Anomaly rate across {len(g):,} machines")

    ok = g[g.n >= 10].sort_values("rate", ascending=False).head(20)
    if ok.empty:
        ok = g.sort_values(["rate", "n"], ascending=False).head(20)
        note = "(few machines have >=10 observations)"
    else:
        note = "(machines with >=10 observations)"
    ax[1].barh(ok.index.astype(str)[::-1], ok["rate"][::-1], color=C_BAD)
    ax[1].set_xlabel("anomaly rate")
    ax[1].set_title(f"Top 20 anomalous machines\n{note}", fontsize=10)
    ax[1].tick_params(axis="y", labelsize=6)

    ax[2].scatter(g.n, g.rate, s=5, alpha=0.2, color=C_NEU)
    ax[2].set_xscale("log")
    ax[2].set_xlabel("observations per machine (log)")
    ax[2].set_ylabel("anomaly rate")
    ax[2].set_title("Rate vs support\n(low support => unstable estimate)", fontsize=10)
    fig.suptitle("Machine-level anomaly concentration", fontweight="bold")
    fig.tight_layout()
    return _save(fig, name)


def fig_anomaly_resource_comparison(df: pd.DataFrame):
    name = "16_anomaly_resource_comparison.png"
    if "anomaly_iforest" not in df.columns:
        return _skip(name, "no anomaly flags")
    cols = [c for c in ["cpu_avg", "mem_avg", "cpi", "memory_pressure",
                        "cpu_tail_ratio", "cpu_burstiness", "telemetry_loss",
                        "cpu_efficiency"] if c in df.columns][:8]
    if not cols:
        return _skip(name, "no resource features")
    sub = _sub(df, 80_000)
    n = len(cols)
    fig, axes = plt.subplots((n + 3) // 4, 4, figsize=(16, 3.8 * ((n + 3) // 4)))
    axes = np.atleast_1d(axes).ravel()
    for a, c in zip(axes, cols):
        d = [sub.loc[sub.anomaly_iforest == k, c]
             .replace([np.inf, -np.inf], np.nan).dropna() for k in (0, 1)]
        if min(len(x) for x in d) < 10:
            a.set_axis_off()
            continue
        bp = a.boxplot(d, labels=["normal", "anomaly"], showfliers=False,
                       patch_artist=True)
        for p, col in zip(bp["boxes"], [C_OK, C_BAD]):
            p.set_facecolor(col)
            p.set_alpha(0.7)
        a.set_title(c, fontsize=10)
    for a in axes[n:]:
        a.set_axis_off()
    fig.suptitle(f"What makes a row anomalous? ({len(sub):,}-row sample, "
                 "outliers hidden)", fontweight="bold")
    fig.tight_layout()
    return _save(fig, name)


# ===========================================================================
# 17-18  models and machines
# ===========================================================================

def fig_model_comparison(res: pd.DataFrame):
    name = "17_model_comparison.png"
    if res is None or res.empty:
        return _skip(name, "no model results")
    hon = res[res.protocol == config.HONEST_PROTOCOL].dropna(subset=["pr_auc"])
    if hon.empty:
        return _skip(name, "no honest-protocol results")
    hon = hon.sort_values("pr_auc")
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    m = ["accuracy", "f1", "roc_auc", "pr_auc"]
    m = [c for c in m if c in hon.columns]
    x = np.arange(len(hon))
    wdt = 0.8 / len(m)
    for i, c in enumerate(m):
        ax[0].barh(x + i * wdt - 0.4, hon[c], wdt, label=c)
    ax[0].set_yticks(x, hon.model, fontsize=8)
    ax[0].legend(fontsize=8)
    ax[0].set_xlim(0, 1)
    ax[0].set_title(f"Model performance ({config.HONEST_PROTOCOL} protocol)")

    if "random" in res.protocol.unique():
        piv = res.pivot_table(index="model", columns="protocol", values="roc_auc",
                              aggfunc="first").dropna(how="all")
        piv = piv.sort_values(config.HONEST_PROTOCOL
                              if config.HONEST_PROTOCOL in piv.columns else piv.columns[0])
        xx = np.arange(len(piv))
        w2 = 0.8 / len(piv.columns)
        for i, c in enumerate(piv.columns):
            ax[1].barh(xx + i * w2 - 0.4, piv[c], w2, label=c)
        ax[1].set_yticks(xx, piv.index, fontsize=8)
        ax[1].axvline(0.5, ls=":", color="k", lw=1)
        ax[1].set_xlim(0, 1.02)
        ax[1].set_xlabel("ROC-AUC")
        ax[1].legend(fontsize=8, title="split protocol", title_fontsize=8)
        ax[1].set_title("Same models, different split\n"
                        "The random-split bars are LEAKAGE, not skill")
    fig.tight_layout()
    return _save(fig, name)


def fig_machine_risk_ranking(machines: pd.DataFrame):
    name = "18_machine_risk_ranking.png"
    if machines is None or machines.empty:
        return _skip(name, "no machine profile")
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    pref = machines[machines.support >= 10] if (machines.support >= 10).any() \
        else machines
    top = pref.sort_values("risk_score_mean", ascending=False).head(20)
    ax[0].barh(top.machine_id.astype(str)[::-1], top.risk_score_mean[::-1],
               color=C_BAD)
    ax[0].set_xlabel("mean degradation score")
    ax[0].tick_params(axis="y", labelsize=6)
    ax[0].set_title(f"Top 20 machines by degradation\n"
                    f"(support >= {'10' if (machines.support >= 10).any() else '1'} "
                    f"observations)", fontsize=10)

    ax[1].hist(machines.risk_score_mean, bins=50, color="#feb24c")
    ax[1].set_xlabel("mean degradation score")
    ax[1].set_ylabel("machines")
    ax[1].set_title(f"Distribution across {len(machines):,} machines")

    if "confidence" in machines.columns:
        vc = machines.confidence.value_counts()
        ax[2].bar(vc.index.astype(str), vc.values,
                  color=["#d9d9d9", "#fdae6b", C_ALT][: len(vc)])
        for i, v in enumerate(vc.values):
            ax[2].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
        ax[2].set_title("Estimate confidence by sample size\n"
                        "Most machines have very few observations", fontsize=10)
    fig.suptitle("Machine-level degradation ranking (cross-sectional, "
                 "peer-relative)", fontweight="bold")
    fig.tight_layout()
    return _save(fig, name)


# ===========================================================================
# 19-24  extras: curves, importance, SHAP, leakage, priority
# ===========================================================================

def fig_confusion_matrices(model_out: dict):
    name = "19_confusion_matrices.png"
    preds = model_out.get("predictions", {})
    if not preds:
        return _skip(name, "no stored predictions")
    from sklearn.metrics import confusion_matrix
    ks = list(preds)[:8]
    n = len(ks)
    fig, axes = plt.subplots((n + 3) // 4, min(4, n),
                             figsize=(4 * min(4, n), 3.6 * ((n + 3) // 4)))
    axes = np.atleast_1d(axes).ravel()
    for a, k in zip(axes, ks):
        yt, yp, _ = preds[k]
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        a.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                a.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black",
                       fontsize=9)
        a.set_xticks([0, 1], ["pred 0", "pred 1"], fontsize=8)
        a.set_yticks([0, 1], ["true 0", "true 1"], fontsize=8)
        a.set_title(k, fontsize=9)
    for a in axes[n:]:
        a.set_axis_off()
    fig.suptitle(f"Confusion matrices ({config.HONEST_PROTOCOL} protocol)",
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, name)


def fig_curves(model_out: dict):
    name = "20_roc_pr_curves.png"
    preds = model_out.get("predictions", {})
    if not preds:
        return _skip(name, "no stored predictions")
    from sklearn.metrics import (average_precision_score, precision_recall_curve,
                                roc_auc_score, roc_curve)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    for k, (yt, _yp, pr) in preds.items():
        if len(np.unique(yt)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yt, pr)
        ax[0].plot(fpr, tpr, lw=1.3, label=f"{k} ({roc_auc_score(yt, pr):.3f})")
        p, r, _ = precision_recall_curve(yt, pr)
        ax[1].plot(r, p, lw=1.3, label=f"{k} ({average_precision_score(yt, pr):.3f})")
        base = float(np.mean(yt))
    ax[0].plot([0, 1], [0, 1], ls=":", color="k", lw=1, label="chance")
    ax[0].set_xlabel("false positive rate")
    ax[0].set_ylabel("true positive rate")
    ax[0].set_title(f"ROC ({config.HONEST_PROTOCOL} protocol)")
    ax[0].legend(fontsize=7)
    try:
        ax[1].axhline(base, ls=":", color="k", lw=1, label=f"base rate {base:.3f}")
    except Exception:
        pass
    ax[1].set_xlabel("recall")
    ax[1].set_ylabel("precision")
    ax[1].set_title("Precision-Recall (the metric that matters under imbalance)")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, name)


def fig_feature_importance(expl: dict):
    name = "21_feature_importance.png"
    if not expl or ("native" not in expl and "permutation" not in expl):
        return _skip(name, "no importance results")
    have = [k for k in ["native", "permutation", "shap"] if k in expl]
    fig, axes = plt.subplots(1, len(have), figsize=(6.5 * len(have), 6.5))
    axes = np.atleast_1d(axes).ravel()
    spec = {"native": ("importance", "Built-in importance", C_OK),
            "permutation": ("perm_importance_mean",
                            "Permutation importance (held-out)", C_ALT),
            "shap": ("mean_abs_shap", "mean |SHAP|", "#756bb1")}
    for a, k in zip(axes, have):
        col, title, colr = spec[k]
        d = expl[k].head(15)
        a.barh(d.feature[::-1], d[col][::-1], color=colr)
        a.tick_params(axis="y", labelsize=7)
        a.set_title(title, fontsize=10)
    fig.suptitle(f"Feature importance for {expl.get('model', '')} "
                 "-- three independent views", fontweight="bold")
    fig.tight_layout()
    return _save(fig, name)


def fig_shap_summary(expl: dict, df: pd.DataFrame):
    name = "22_shap_summary.png"
    sv = expl.get("shap_matrix")
    if sv is None:
        return _skip(name, expl.get("shap_meta", {}).get("status", "no SHAP values"))
    try:
        import shap
        idx = expl["shap_index"]
        cols = expl["native"]["feature"].tolist() if "native" in expl else None
        X = df.iloc[idx][expl["shap"]["feature"].tolist()] if cols is None else \
            df.iloc[idx][cols]
        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(sv, X, show=False, max_display=18, plot_size=None)
        plt.title("SHAP summary (held-out sample)", fontweight="bold")
        return _save(plt.gcf(), name)
    except Exception as exc:
        return _skip(name, f"shap summary_plot failed: {exc}")


def fig_leakage_demo(res: pd.DataFrame):
    name = "23_leakage_effect.png"
    if res is None or res.empty or "random" not in res.protocol.unique():
        return _skip(name, "no random-split baseline to contrast")
    piv = res.pivot_table(index="model", columns="protocol",
                          values=["roc_auc", "pr_auc"], aggfunc="first")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for a, metric in zip(ax, ["roc_auc", "pr_auc"]):
        d = piv[metric].dropna(how="all")
        x = np.arange(len(d))
        w = 0.8 / len(d.columns)
        for i, c in enumerate(d.columns):
            a.bar(x + i * w - 0.4, d[c], w, label=c)
        a.set_xticks(x, d.index, rotation=30, ha="right", fontsize=8)
        a.set_ylabel(metric)
        a.set_ylim(0, 1.05)
        a.legend(fontsize=8, title="split", title_fontsize=8)
        a.set_title(f"{metric.upper()} by split protocol")
    fig.suptitle("Why the split protocol decides the headline number\n"
                 "Random split = memorising join-replicated near-duplicates",
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, name)


def fig_priority_and_class(df: pd.DataFrame):
    name = "24_priority_scheduling_analysis.png"
    if config.TARGET not in df.columns:
        return _skip(name, "no target")
    panels = []
    for c, lab in [("priority_tier", "priority tier"),
                   ("scheduling_class", "scheduling class"),
                   ("is_production", "production workload"),
                   ("is_evictable", "evictable workload")]:
        if c in df.columns:
            panels.append((c, lab))
    if not panels:
        return _skip(name, "no scheduling columns")
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.6))
    axes = np.atleast_1d(axes).ravel()
    for a, (c, lab) in zip(axes, panels):
        by = df.groupby(c)[config.TARGET].agg(["mean", "size"])
        by = by[by["size"] >= 30]
        if by.empty:
            a.set_axis_off()
            continue
        a.bar(by.index.astype(str), by["mean"], color="#3182bd")
        a.axhline(df[config.TARGET].mean(), ls="--", color=C_NEU, lw=1)
        a.set_title(f"failure rate by {lab}", fontsize=10)
        a.tick_params(axis="x", rotation=30, labelsize=8)
    fig.suptitle("Failure rate across workload classes (groups with >=30 rows; "
                 "dashed line = overall rate)", fontweight="bold")
    fig.tight_layout()
    return _save(fig, name)


# ===========================================================================
# driver
# ===========================================================================

def make_all(df: pd.DataFrame, ctx: dict) -> list[str]:
    """
    Build every figure the available results support.

    ``ctx`` carries whatever earlier stages produced; each figure checks for
    what it needs, so a partially-complete run still yields a usable figure set.
    """
    jobs = [
        (fig_dataset_overview, (df, ctx.get("profile"))),
        (fig_missing_values, (ctx.get("raw_missing"),)),
        (fig_target_distribution, (df,)),
        (fig_cpu_distributions, (df,)),
        (fig_memory_distributions, (df,)),
        (fig_correlation_heatmap, (df, ctx.get("feature_cols", []))),
        (fig_resource_by_failure, (df,)),
        (fig_temporal_volume, (df, ctx.get("panel"))),
        (fig_temporal_resource_trends, (ctx.get("panel"), ctx.get("panel_trends"))),
        (fig_risk_distribution, (df, ctx.get("risk_meta", {}))),
        (fig_risk_components, (df, ctx.get("risk_meta", {}))),
        (fig_risk_validation, (df, ctx.get("risk_meta", {}).get("validation", {}))),
        (fig_anomaly_distribution, (df, ctx.get("anomaly", {}))),
        (fig_anomaly_over_time, (df,)),
        (fig_machine_anomaly_rate, (df,)),
        (fig_anomaly_resource_comparison, (df,)),
        (fig_model_comparison, (ctx.get("model_comparison"),)),
        (fig_machine_risk_ranking, (ctx.get("machines"),)),
        (fig_confusion_matrices, (ctx.get("model_out", {}),)),
        (fig_curves, (ctx.get("model_out", {}),)),
        (fig_feature_importance, (ctx.get("explain", {}),)),
        (fig_shap_summary, (ctx.get("explain", {}), df)),
        (fig_leakage_demo, (ctx.get("model_comparison"),)),
        (fig_priority_and_class, (df,)),
    ]
    for fn, args in jobs:
        try:
            fn(*args)
        except Exception as exc:
            warn(f"{fn.__name__} failed: {exc}")
    info(f"{len(_MADE)} figures written to {config.FIG_DIR}")
    return figures_made()
