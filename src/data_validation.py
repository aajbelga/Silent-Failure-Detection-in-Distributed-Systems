"""
data_validation.py
==================
STAGE 1 -- automated dataset profiling and integrity validation.

Produces `outputs/reports/data_profile.txt` plus machine-readable tables.

The distinctive part of this module is :func:`detect_target_leakage`, which
*searches for* columns that determine the target rather than assuming which
ones do. On this dataset it independently rediscovers that ``event``,
``instance_events_type`` and ``collections_events_type`` are perfect
re-encodings of ``failed`` -- i.e. the leakage finding is produced by the
pipeline itself, not hard-coded by the author.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from . import config
from .utils import (banner, fmt, fmt_int, info, mem_mb, save_table, save_text,
                    section)


# ---------------------------------------------------------------------------
# Core profiling primitives
# ---------------------------------------------------------------------------

def missing_value_table(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    m = df.isna().sum()
    out = pd.DataFrame({
        "column": m.index,
        "dtype": [str(df[c].dtype) for c in m.index],
        "missing": m.values,
        "missing_pct": (100.0 * m.values / max(n, 1)).round(4),
        "non_null": n - m.values,
    })
    return out.sort_values("missing", ascending=False).reset_index(drop=True)


def numerical_summary(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return pd.DataFrame()
    desc = num.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    desc["missing"] = num.isna().sum()
    desc["n_unique"] = num.nunique()
    desc["skew"] = num.skew(numeric_only=True)
    desc["kurtosis"] = num.kurtosis(numeric_only=True)
    desc["zeros"] = (num == 0).sum()
    desc["n_infinite"] = np.isinf(num.to_numpy(dtype="float64", na_value=np.nan)).sum(axis=0)
    return desc.reset_index().rename(columns={"index": "column"})


def categorical_summary(df: pd.DataFrame, max_examples: int = 5) -> pd.DataFrame:
    cats = df.select_dtypes(include=["object", "category", "bool"])
    rows = []
    for c in cats.columns:
        s = cats[c]
        vc = s.value_counts(dropna=True)
        rows.append({
            "column": c,
            "dtype": str(s.dtype),
            "n_unique": int(s.nunique(dropna=True)),
            "missing": int(s.isna().sum()),
            "top_value": str(vc.index[0])[:70] if len(vc) else "",
            "top_count": int(vc.iloc[0]) if len(vc) else 0,
            "top_pct": round(100.0 * vc.iloc[0] / max(len(df), 1), 3) if len(vc) else 0.0,
            "examples": " | ".join(str(v)[:28] for v in vc.index[:max_examples]),
        })
    return pd.DataFrame(rows)


def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Outlier counts using the IQR rule and a robust (MAD) rule.

    Both are reported because on this trace they disagree sharply: the resource
    columns are so skewed that the IQR rule flags a large share of rows. That
    disagreement is itself the evidence for *not* trimming outliers, since the
    "outliers" are the heavy tail of a genuine workload distribution -- and the
    heavy tail is exactly where degradation lives.
    """
    num = df.select_dtypes(include=[np.number])
    rows = []
    for c in num.columns:
        s = pd.to_numeric(num[c], errors="coerce").dropna()
        if len(s) < 10:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_iqr = int(((s < lo) | (s > hi)).sum())
        med = s.median()
        mad = (s - med).abs().median()
        n_mad = int((((s - med).abs() / (1.4826 * mad)) > 3.5).sum()) if mad > 0 else 0
        rows.append({
            "column": c,
            "n": len(s),
            "iqr_outliers": n_iqr,
            "iqr_outlier_pct": round(100.0 * n_iqr / len(s), 3),
            "mad_outliers": n_mad,
            "mad_outlier_pct": round(100.0 * n_mad / len(s), 3),
            "min": s.min(), "max": s.max(), "median": med,
        })
    return pd.DataFrame(rows).sort_values("iqr_outlier_pct", ascending=False)


def duplicate_report(df: pd.DataFrame) -> dict:
    """Duplicates at several definitions of 'the same record'."""
    cols = [c for c in df.columns if c not in config.INDEX_ARTIFACT_COLS]
    out = {"full_row_duplicates": int(df.duplicated(subset=cols).sum())}

    def _dups(keys, label):
        keys = [k for k in keys if k in df.columns]
        if keys:
            out[label] = int(df.duplicated(subset=keys).sum())
            out[label + "_keys"] = keys

    _dups(["time", "collection_id", "instance_index", "machine_id"], "dup_on_event_key")
    _dups(["collection_id", "instance_index", "start_time", "end_time"], "dup_on_usage_key")
    return out


def index_artifact_report(df: pd.DataFrame) -> dict:
    """Detect a pandas-written index column masquerading as data."""
    out = {}
    for c in config.INDEX_ARTIFACT_COLS:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce").to_numpy()
            out[c] = {
                "n_unique": int(df[c].nunique()),
                "is_rangeindex": bool(len(vals) == len(df)
                                     and np.array_equal(vals, np.arange(len(df)))),
            }
    return out


# ---------------------------------------------------------------------------
# Leakage discovery  (the important bit)
# ---------------------------------------------------------------------------

def detect_target_leakage(df: pd.DataFrame, target: str = config.TARGET) -> pd.DataFrame:
    """
    Search every column for a deterministic relationship with the target.

    Method: group rows by the candidate column's value and ask whether the
    target is constant inside every group. If it is, the column *determines*
    the target -- a classifier can invert the mapping exactly, so any accuracy
    it reports is meaningless.

    We report ``purity`` = the share of rows lying in groups that are pure.
    purity == 1.0 means perfect, exact leakage.
    """
    if target not in df.columns:
        return pd.DataFrame()

    y = df[target]
    n = len(df)
    rows = []
    for c in df.columns:
        if c == target:
            continue
        nu = df[c].nunique(dropna=False)
        # A column with as many values as rows is an identifier, not a leak in
        # the "re-encoded target" sense; skip to keep the table readable.
        if nu <= 1 or nu > 5_000:
            continue
        g = y.groupby(df[c].astype(str), dropna=False)
        sizes = g.size()
        # A group is pure when the target never varies within it.
        pure_mask = (g.nunique(dropna=False) <= 1)
        pure_rows = int(sizes[pure_mask].sum())
        purity = pure_rows / max(n, 1)
        if purity >= 0.90:
            rows.append({
                "column": c,
                "n_unique": int(nu),
                "target_purity": round(purity, 6),
                "pure_groups": int(pure_mask.sum()),
                "total_groups": int(len(sizes)),
                "verdict": "PERFECT LEAKAGE (target re-encoding)" if purity >= 0.999999
                           else "SEVERE LEAKAGE RISK",
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("target_purity", ascending=False).reset_index(drop=True)
    return out


def detect_identical_columns(df: pd.DataFrame) -> list[tuple[str, str]]:
    """
    Find exactly-equal column pairs (redundancy).

    Rediscovers that instance_events_type and collections_events_type are the
    same column stored twice.
    """
    pairs = []
    cols = list(df.columns)
    # Fingerprint first so we only deep-compare plausible candidates: O(n) each.
    fp: dict[tuple, list[str]] = {}
    for c in cols:
        s = df[c]
        key = (str(s.dtype), int(s.isna().sum()), int(s.nunique(dropna=False)))
        fp.setdefault(key, []).append(c)
    for group in fp.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if df[a].equals(df[b]):
                    pairs.append((a, b))
    return pairs


def replication_report(df: pd.DataFrame) -> dict:
    """
    Quantify how often the resource-usage block repeats.

    This is the structural finding behind the whole leakage argument: the CSV is
    a join of a small usage table onto a large event table, so the same usage
    vector reappears many times. If those repeats straddle a random train/test
    split, the model can memorise instead of generalise.
    """
    usage_block = [c for c in [
        "start_time", "end_time", "average_usage", "maximum_usage",
        "random_sample_usage", "assigned_memory", "page_cache_memory",
        "cycles_per_instruction", "memory_accesses_per_instruction",
        "sample_rate", "cpu_usage_distribution", "tail_cpu_usage_distribution",
    ] if c in df.columns]
    if not usage_block:
        return {}
    sizes = df.groupby(usage_block, dropna=False, sort=False).size()
    out = {
        "usage_block_columns": usage_block,
        "n_rows": int(len(df)),
        "n_unique_usage_vectors": int(len(sizes)),
        "mean_replication": float(sizes.mean()),
        "median_replication": float(sizes.median()),
        "max_replication": int(sizes.max()),
        "pct_rows_in_replicated_vectors": float(100.0 * sizes[sizes > 1].sum() / len(df)),
    }
    # How target-constant is a collection? Drives the choice of split.
    if "collection_id" in df.columns and config.TARGET in df.columns:
        nu = df.groupby("collection_id", sort=False)[config.TARGET].nunique()
        out["n_collections"] = int(len(nu))
        out["pct_collections_target_constant"] = float(100.0 * (nu <= 1).mean())
    return out


def time_sentinel_report(df: pd.DataFrame) -> dict:
    """Characterise the Borg time sentinels and their association with the target."""
    if "time" not in df.columns:
        return {}
    t = pd.to_numeric(df["time"], errors="coerce")
    zero = (t == config.TIME_SENTINEL_MIN)
    maxi = (t == config.TIME_SENTINEL_MAX)
    valid = ~(zero | maxi) & t.notna()
    out = {
        "n_time_zero": int(zero.sum()),
        "n_time_int64max": int(maxi.sum()),
        "n_time_valid": int(valid.sum()),
        "pct_time_valid": float(100.0 * valid.mean()),
    }
    if valid.any():
        out["valid_min_us"] = int(t[valid].min())
        out["valid_max_us"] = int(t[valid].max())
        out["valid_span_days"] = float((t[valid].max() - t[valid].min()) / config.US_PER_SEC / 86400)
    if config.TARGET in df.columns:
        y = df[config.TARGET]
        out["fail_rate_time_zero"] = float(y[zero].mean()) if zero.any() else None
        out["fail_rate_time_valid"] = float(y[valid].mean()) if valid.any() else None
        out["fail_rate_overall"] = float(y.mean())
    return out


def temporal_support_report(df: pd.DataFrame) -> dict:
    """
    Measure how many observations each candidate entity actually has.

    This is what licenses (or forbids) rolling-window analysis per entity, and
    is the evidence used to move degradation analysis off the machine grain.
    """
    out = {}
    if "time" in df.columns:
        t = pd.to_numeric(df["time"], errors="coerce")
        valid = df[(t > config.TIME_SENTINEL_MIN) & (t < config.TIME_SENTINEL_MAX)]
    else:
        valid = df
    for key, label in [("machine_id", "machine"), ("collection_id", "collection")]:
        if key not in df.columns:
            continue
        g = df.groupby(key, sort=False).size()
        gv = valid.groupby(key, sort=False).size() if len(valid) else g
        out[label] = {
            "n_entities": int(len(g)),
            "events_mean": float(g.mean()),
            "events_median": float(g.median()),
            "events_max": int(g.max()),
            "entities_ge_3_valid_time": int((gv >= 3).sum()),
            "entities_ge_10_valid_time": int((gv >= 10).sum()),
            "entities_ge_50_valid_time": int((gv >= 50).sum()),
            "rolling_window_viable": bool((gv >= config.MIN_EVENTS_FOR_COLLECTION_TREND).sum() >= 100),
        }
    return out


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def profile_dataset(df: pd.DataFrame, physical_lines: int | None = None) -> dict:
    """Run every check and write the human + machine readable outputs."""
    buf = io.StringIO()

    def w(line: str = ""):
        buf.write(line + "\n")

    n, k = df.shape
    w("=" * 78)
    w(" AUTOMATED DATA PROFILING REPORT")
    w(" Silent Failure Detection in Distributed Systems -- Google Borg Traces")
    w("=" * 78)
    w()
    w(f"Source file             : {config.RAW_CSV.name}")
    w(f"File size               : {config.RAW_CSV.stat().st_size / 1024**2:,.1f} MB")
    w(f"Logical records (rows)  : {n:,}")
    w(f"Columns                 : {k}")
    if physical_lines:
        w(f"Physical newlines       : {physical_lines:,}")
        w("   NOTE: the physical line count exceeds the record count because")
        w("   cpu_usage_distribution / tail_cpu_usage_distribution store NumPy")
        w("   array reprs that wrap across lines inside quoted fields. Any")
        w("   line-based row count of this file is therefore wrong.")
    w(f"In-memory size          : {mem_mb(df):,.1f} MB")
    w()

    # -- columns / dtypes --------------------------------------------------
    w("-" * 78)
    w(" 1. COLUMNS, DTYPES, MISSINGNESS, CARDINALITY")
    w("-" * 78)
    w(f"{'#':>3} {'column':34s} {'dtype':9s} {'missing':>9s} {'miss%':>7s} {'unique':>9s}")
    for i, c in enumerate(df.columns):
        s = df[c]
        w(f"{i:>3} {c[:34]:34s} {str(s.dtype):9s} {s.isna().sum():>9,} "
          f"{100 * s.isna().mean():>6.2f}% {s.nunique(dropna=True):>9,}")
    w()

    # -- duplicates --------------------------------------------------------
    dup = duplicate_report(df)
    w("-" * 78)
    w(" 2. DUPLICATE ANALYSIS")
    w("-" * 78)
    w(f"Exact full-row duplicates                      : {dup.get('full_row_duplicates', 0):,}")
    if "dup_on_event_key" in dup:
        w(f"Duplicates on {dup['dup_on_event_key_keys']}")
        w(f"                                               : {dup['dup_on_event_key']:,}")
    if "dup_on_usage_key" in dup:
        w(f"Duplicates on {dup['dup_on_usage_key_keys']}")
        w(f"                                               : {dup['dup_on_usage_key']:,}")
    idx = index_artifact_report(df)
    for c, meta in idx.items():
        w(f"'{c}' is an exact RangeIndex                     : {meta['is_rangeindex']}")
        w("   -> pandas index artefact, carries no information, will be dropped.")
    w()

    # -- target ------------------------------------------------------------
    w("-" * 78)
    w(" 3. TARGET VARIABLE")
    w("-" * 78)
    if config.TARGET in df.columns:
        vc = df[config.TARGET].value_counts(dropna=False)
        w(f"Target column           : '{config.TARGET}'")
        w(f"Distinct values         : {df[config.TARGET].nunique(dropna=False)}")
        for kk, vv in vc.items():
            w(f"   {config.TARGET} = {kk!s:12s} {vv:>10,}  ({100 * vv / n:6.3f}%)")
        w()
        w("INTERPRETATION: 'failed' is an EXPLICIT failure indicator -- it marks")
        w("rows whose instance event is a FAIL event. It is NOT a silent-failure")
        w("label. No ground truth for silent failure exists in this trace; that")
        w("is why silent degradation is handled as a separate, unsupervised")
        w("research component rather than a classification target.")
    else:
        w(f"Target column '{config.TARGET}' NOT PRESENT -- supervised stages skipped.")
    w()

    # -- leakage -----------------------------------------------------------
    leak = detect_target_leakage(df)
    ident = detect_identical_columns(df)
    w("-" * 78)
    w(" 4. TARGET-LEAKAGE DISCOVERY  (automated search, not hard-coded)")
    w("-" * 78)
    w("Method: for each candidate column, group rows by its value and test")
    w("whether the target is constant within every group. purity = 1.0 means")
    w("the column determines the target exactly.")
    w()
    if leak.empty:
        w("No column with target purity >= 0.90 was found.")
    else:
        w(f"{'column':32s} {'unique':>7s} {'purity':>9s}  verdict")
        for _, r in leak.iterrows():
            w(f"{r['column'][:32]:32s} {r['n_unique']:>7,} {r['target_purity']:>9.6f}  {r['verdict']}")
        w()
        perfect = leak[leak.target_purity >= 0.999999]["column"].tolist()
        if perfect:
            w("CONSEQUENCE: the following columns are re-encodings of the target and")
            w("are EXCLUDED from every supervised feature matrix:")
            for c in perfect:
                w(f"   - {c}")
            w("Including any one of them yields ~100% accuracy that measures")
            w("nothing but the model's ability to read the answer key.")
    w()
    if ident:
        w("Exactly identical column pairs (pure redundancy):")
        for a, b in ident:
            w(f"   {a}  ==  {b}")
        w("   -> one of each pair is dropped.")
    else:
        w("No exactly identical column pairs found.")
    w()

    # -- replication -------------------------------------------------------
    rep = replication_report(df)
    w("-" * 78)
    w(" 5. JOIN-INDUCED REPLICATION  (why a random split is invalid here)")
    w("-" * 78)
    if rep:
        w(f"Rows                                     : {rep['n_rows']:,}")
        w(f"Distinct resource-usage vectors          : {rep['n_unique_usage_vectors']:,}")
        w(f"Mean replication factor                  : {rep['mean_replication']:.1f}x")
        w(f"Median replication factor                : {rep['median_replication']:.0f}x")
        w(f"Max replication factor                   : {rep['max_replication']:,}x")
        w(f"Rows sharing a vector with another row   : {rep['pct_rows_in_replicated_vectors']:.2f}%")
        if "pct_collections_target_constant" in rep:
            w(f"Collections                              : {rep['n_collections']:,}")
            w(f"Collections where target never varies    : {rep['pct_collections_target_constant']:.1f}%")
        w()
        w("INTERPRETATION: this file is a join of a small usage table onto a large")
        w("event table, so each usage measurement is duplicated many times over.")
        w("Because the target is also near-constant within a collection, a random")
        w("row-level train/test split places near-identical rows on both sides.")
        w("The model then memorises usage vectors instead of learning behaviour.")
        w("The remedy used throughout this project is GROUPED splitting by")
        w("collection_id, so no collection appears in both train and test.")
    else:
        w("Usage-block columns absent; replication analysis not applicable.")
    w()

    # -- time --------------------------------------------------------------
    ts = time_sentinel_report(df)
    w("-" * 78)
    w(" 6. TEMPORAL COLUMN SEMANTICS AND SENTINELS")
    w("-" * 78)
    if ts:
        w(f"time == 0            (before trace window) : {ts['n_time_zero']:,}")
        w(f"time == 2**63-1      (after trace window)  : {ts['n_time_int64max']:,}")
        w(f"time  valid timestamps                     : {ts['n_time_valid']:,} "
          f"({ts['pct_time_valid']:.2f}%)")
        if "valid_span_days" in ts:
            w(f"Valid-time span                            : {ts['valid_span_days']:.2f} days")
        if ts.get("fail_rate_time_zero") is not None:
            w()
            w(f"Fail rate where time == 0                  : {ts['fail_rate_time_zero']:.4f}")
            w(f"Fail rate where time is a real timestamp   : {ts['fail_rate_time_valid']:.4f}")
            w(f"Fail rate overall                          : {ts['fail_rate_overall']:.4f}")
            w()
            w("INTERPRETATION: the sentinel value 0 is strongly associated with")
            w("failure, but that association is trace-boundary BOOKKEEPING, not")
            w("system behaviour: it means the event predates the observation")
            w("window. Feeding raw 'time' to a model would manufacture a powerful")
            w("but scientifically empty predictor. We therefore convert sentinels")
            w("to NaN and expose their status as an explicit flag.")
    else:
        w("No 'time' column present.")
    w()

    # -- temporal support --------------------------------------------------
    sup = temporal_support_report(df)
    w("-" * 78)
    w(" 7. TEMPORAL SUPPORT PER ENTITY  (what rolling analysis is licensed)")
    w("-" * 78)
    for label, meta in sup.items():
        w(f"[{label}]")
        w(f"   entities                          : {meta['n_entities']:,}")
        w(f"   events per entity  mean/median/max: {meta['events_mean']:.2f} / "
          f"{meta['events_median']:.0f} / {meta['events_max']:,}")
        w(f"   entities with >=3  valid-time obs  : {meta['entities_ge_3_valid_time']:,}")
        w(f"   entities with >=10 valid-time obs  : {meta['entities_ge_10_valid_time']:,}")
        w(f"   entities with >=50 valid-time obs  : {meta['entities_ge_50_valid_time']:,}")
        w(f"   rolling-window analysis viable     : {meta['rolling_window_viable']}")
    w()
    w("CONSEQUENCE: machines carry only a handful of observations each, so")
    w("per-machine rolling-window degradation curves are NOT statistically")
    w("supportable on this file. Degradation trends are therefore computed at")
    w("the collection and cluster-hour grain, while machines are profiled")
    w("cross-sectionally against their cluster peers.")
    w()

    # -- numeric / categorical --------------------------------------------
    nums = numerical_summary(df)
    cats = categorical_summary(df)
    outl = outlier_summary(df)

    w("-" * 78)
    w(" 8. NUMERICAL SUMMARY")
    w("-" * 78)
    if not nums.empty:
        show = nums[["column", "count", "mean", "std", "min", "50%", "max",
                     "n_unique", "missing", "skew"]]
        w(show.to_string(index=False, float_format=lambda v: f"{v:,.6g}"))
    else:
        w("No numeric columns.")
    w()

    w("-" * 78)
    w(" 9. CATEGORICAL SUMMARY")
    w("-" * 78)
    if not cats.empty:
        w(cats.to_string(index=False))
    else:
        w("No categorical columns.")
    w()

    w("-" * 78)
    w(" 10. OUTLIER STATISTICS (IQR rule vs robust MAD rule)")
    w("-" * 78)
    if not outl.empty:
        w(outl.head(25).to_string(index=False, float_format=lambda v: f"{v:,.6g}"))
        w()
        w("INTERPRETATION: the two rules disagree strongly on the resource")
        w("columns because those distributions are heavy-tailed by nature. The")
        w("tail is real workload behaviour -- and degradation lives in the tail --")
        w("so extreme values are RETAINED, not trimmed. Only physically")
        w("impossible values (negatives, infinities) are corrected.")
    w()

    # -- write out ---------------------------------------------------------
    text = buf.getvalue()
    save_text(text, config.REPORT_DIR / "data_profile.txt")
    save_table(missing_value_table(df), config.TABLE_DIR / "missing_values.csv")
    if not nums.empty:
        save_table(nums, config.TABLE_DIR / "numerical_summary.csv")
    if not cats.empty:
        save_table(cats, config.TABLE_DIR / "categorical_summary.csv")
    if not outl.empty:
        save_table(outl, config.TABLE_DIR / "outlier_summary.csv")
    if not leak.empty:
        save_table(leak, config.TABLE_DIR / "leakage_audit.csv")

    stats = pd.DataFrame([{
        "n_rows": n, "n_columns": k,
        "memory_mb": round(mem_mb(df), 2),
        "file_size_mb": round(config.RAW_CSV.stat().st_size / 1024 ** 2, 2),
        "physical_lines": physical_lines or -1,
        "full_row_duplicates": dup.get("full_row_duplicates", 0),
        "n_unique_usage_vectors": rep.get("n_unique_usage_vectors", -1),
        "mean_replication": round(rep.get("mean_replication", float("nan")), 3),
        "pct_collections_target_constant": round(rep.get("pct_collections_target_constant", float("nan")), 2),
        "failure_rate": round(float(df[config.TARGET].mean()), 6) if config.TARGET in df else None,
        "n_leaking_columns": int((leak.target_purity >= 0.999999).sum()) if not leak.empty else 0,
        "pct_time_valid": round(ts.get("pct_time_valid", float("nan")), 3),
        "n_machines": int(df["machine_id"].nunique()) if "machine_id" in df else -1,
        "n_collections": int(df["collection_id"].nunique()) if "collection_id" in df else -1,
        "n_clusters": int(df["cluster"].nunique()) if "cluster" in df else -1,
    }])
    save_table(stats, config.METRICS_DIR / "dataset_statistics.csv")

    return {
        "shape": (n, k),
        "duplicates": dup,
        "leakage": leak,
        "identical_pairs": ident,
        "replication": rep,
        "time_sentinels": ts,
        "temporal_support": sup,
        "numeric_summary": nums,
        "categorical_summary": cats,
        "outliers": outl,
        "stats": stats,
    }
