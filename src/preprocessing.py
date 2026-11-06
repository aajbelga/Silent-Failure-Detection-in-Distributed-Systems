"""
preprocessing.py
================
STAGE 2 -- cleaning, with a written justification for every decision.

Design rule for this module: **no blanket ``dropna()``**. Missingness in this
trace is structured and informative, so it is handled column by column:

  * ``cycles_per_instruction`` / ``memory_accesses_per_instruction`` are missing
    together in 30.72% of rows. The missingness is *not* random -- rows with a
    CPI reading fail 27.3% of the time versus 12.9% for rows without one. Those
    columns are therefore imputed with the cluster median AND accompanied by a
    ``perf_counters_missing`` indicator, so the model can use the fact of
    absence as information instead of having it silently smoothed away.
  * ``random_sample_usage_memory`` is 100% null -- dropped, because imputing a
    column with no observed values would be fabricating data.
  * ``resource_request`` (0.19%), ``vertical_scaling`` / ``scheduler`` (0.24%)
    and ``maximum_usage_memory`` (1.21%) are imputed, not deleted: dropping
    those rows would discard genuine failure events for an unrelated reason.

Dropping all rows with any missing value would have destroyed ~31% of the
dataset, and disproportionately the non-failing rows -- i.e. it would have
*changed the class balance* while appearing to be routine hygiene.
"""

from __future__ import annotations

import ast
import io
import re

import numpy as np
import pandas as pd

from . import config
from .utils import fmt_int, info, save_text, section, warn

# ---------------------------------------------------------------------------
# Parsers for the nested columns
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_dict_column(s: pd.Series, key: str) -> pd.Series:
    """
    Pull one key out of a column of Python-dict literals.

    A vectorised regex is used rather than ``ast.literal_eval`` per row: on
    405,894 rows the regex is roughly two orders of magnitude faster, and the
    payloads have a fixed, simple shape ({'cpus': float, 'memory': float|None})
    that makes the regex safe. ``None`` becomes NaN.
    """
    pat = rf"'{key}'\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|None)"
    extracted = s.astype("string").str.extract(pat, expand=False)
    extracted = extracted.replace("None", pd.NA)
    return pd.to_numeric(extracted, errors="coerce")


def parse_dict_column_strict(value: str, key: str):
    """Reference implementation via literal_eval -- used only to validate the regex."""
    try:
        d = ast.literal_eval(value)
        return d.get(key) if isinstance(d, dict) else None
    except (ValueError, SyntaxError):
        return None


def parse_array_column(s: pd.Series, expected_len: int) -> np.ndarray:
    """
    Turn a column of NumPy array reprs into a dense (n, expected_len) matrix.

    Rows whose array is absent or malformed become all-NaN, which downstream
    imputation then handles -- rather than raising and killing the run.
    """
    n = len(s)
    out = np.full((n, expected_len), np.nan, dtype="float64")
    values = s.to_numpy()
    for i, v in enumerate(values):
        if not isinstance(v, str) or len(v) < 2:
            continue
        nums = _NUM_RE.findall(v)
        if len(nums) != expected_len:
            continue
        try:
            out[i, :] = [float(x) for x in nums]
        except ValueError:
            continue
    return out


def count_list_items(s: pd.Series, token: str) -> pd.Series:
    """Count occurrences of a marker token inside a list-literal column."""
    st = s.astype("string").fillna("[]")
    empty = st.str.strip().isin(["[]", ""])
    counts = st.str.count(re.escape(token)).fillna(0).astype("int32")
    return counts.where(~empty, 0)


def count_ids(s: pd.Series) -> pd.Series:
    """Count integer ids inside a list-literal column (dependency lists)."""
    st = s.astype("string").fillna("[]")
    empty = st.str.strip().isin(["[]", ""])
    counts = st.str.count(r"\d+").fillna(0).astype("int32")
    return counts.where(~empty, 0)


# ---------------------------------------------------------------------------
# Main cleaning routine
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame, profile: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Clean the raw frame.

    Returns ``(cleaned_df, decisions)`` where ``decisions`` is an ordered log of
    what was done and why -- written verbatim into the preprocessing report so
    every step is auditable.
    """
    log: list[dict] = []
    n0, k0 = df.shape

    def record(step: str, action: str, rationale: str, **nums):
        log.append({"step": step, "action": action, "rationale": rationale, **nums})
        info(f"{step}: {action}")

    df = df.copy()

    # -- 1. index artefact -------------------------------------------------
    dropped_index = [c for c in config.INDEX_ARTIFACT_COLS if c in df.columns]
    if dropped_index:
        is_range = all(
            np.array_equal(pd.to_numeric(df[c], errors="coerce").to_numpy(),
                           np.arange(len(df)))
            for c in dropped_index
        )
        df = df.drop(columns=dropped_index)
        record("drop_index_artifact", f"dropped {dropped_index}",
               f"Verified to be an exact RangeIndex ({is_range}); a pandas "
               "export artefact carrying zero information.",
               columns_dropped=len(dropped_index))

    # -- 2. completely empty columns ---------------------------------------
    all_null = [c for c in df.columns if df[c].isna().all()]
    if all_null:
        df = df.drop(columns=all_null)
        record("drop_all_null_columns", f"dropped {all_null}",
               "Column has no observed values at all; imputing it would "
               "fabricate data.", columns_dropped=len(all_null))
    else:
        record("drop_all_null_columns", "none found",
               "Every raw column has at least one observed value. (Note: the "
               "100%-null field is the 'memory' key *inside* "
               "random_sample_usage, which surfaces after parsing -- handled "
               "at step 5.)")

    # -- 3. exact duplicate columns ----------------------------------------
    # instance_events_type vs collections_events_type are byte-identical.
    dup_cols = []
    cols = list(df.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            if a in dup_cols or b in dup_cols:
                continue
            if df[a].dtype == df[b].dtype and df[a].nunique(dropna=False) == df[b].nunique(dropna=False):
                if df[a].equals(df[b]):
                    dup_cols.append(b)
                    record("drop_duplicate_column", f"'{b}' is identical to '{a}' -> dropped",
                           "Byte-identical duplicate column: pure redundancy, "
                           "and both are target re-encodings.")
    if dup_cols:
        df = df.drop(columns=dup_cols)

    # -- 4. duplicate rows -------------------------------------------------
    n_full_dup = int(df.duplicated().sum())
    key = [c for c in ["time", "collection_id", "instance_index", "machine_id",
                       "event", "start_time"] if c in df.columns]
    n_key_dup = int(df.duplicated(subset=key).sum()) if key else 0
    if n_full_dup > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        record("drop_duplicate_rows", f"removed {n_full_dup:,} exact duplicate rows",
               "Identical records add no information and would bias any "
               "frequency-based statistic.", rows_removed=n_full_dup)
    else:
        record("duplicate_rows", f"0 exact duplicates; {n_key_dup:,} rows share the "
               f"event key {key}",
               "No exact duplicates exist, so no rows are removed. Rows sharing "
               "an event key are legitimately distinct events (different event "
               "type or usage window) and are RETAINED -- deleting them would "
               "discard real observations.", rows_removed=0)

    # -- 5. parse nested dict columns --------------------------------------
    parsed_cols = []
    for col in config.DICT_COLS:
        if col not in df.columns:
            continue
        for k in ("cpus", "memory"):
            new = f"{col}_{k}"
            df[new] = parse_dict_column(df[col], k)
            parsed_cols.append(new)
    if parsed_cols:
        # validate the fast regex against the strict parser on a random sample
        check_col = next((c for c in config.DICT_COLS if c in df.columns), None)
        agree = None
        if check_col is not None:
            samp = df[check_col].dropna().sample(
                min(500, df[check_col].notna().sum()), random_state=config.RANDOM_STATE)
            strict = samp.map(lambda v: parse_dict_column_strict(v, "cpus"))
            fast = parse_dict_column(samp, "cpus")
            both = pd.DataFrame({"s": pd.to_numeric(strict, errors="coerce"), "f": fast})
            agree = float(np.isclose(both.s, both.f, equal_nan=True).mean())
        record("parse_dict_columns",
               f"extracted {len(parsed_cols)} numeric fields from {config.DICT_COLS}",
               f"These columns store Python dict literals, not numbers. A "
               f"vectorised regex extraction was validated against ast.literal_eval "
               f"on a 500-row sample: agreement = {agree:.4f}." if agree is not None
               else "Vectorised regex extraction of dict payloads.",
               columns_added=len(parsed_cols))

    # drop the now-redundant raw dict text columns
    drop_raw_dicts = [c for c in config.DICT_COLS if c in df.columns]
    if drop_raw_dicts:
        df = df.drop(columns=drop_raw_dicts)
        record("drop_raw_dict_text", f"dropped {drop_raw_dicts}",
               "Superseded by the parsed numeric fields; keeping the strings "
               "would waste ~120 MB of memory for no analytical gain.")

    # 100%-null parsed field
    fully_null = [c for c in df.columns if c in config.FULLY_NULL_PARSED and df[c].isna().all()]
    fully_null += [c for c in parsed_cols if c in df.columns and df[c].isna().all()
                   and c not in fully_null]
    if fully_null:
        df = df.drop(columns=fully_null)
        record("drop_fully_null_parsed", f"dropped {fully_null}",
               "100% missing after parsing (the 'memory' key of "
               "random_sample_usage is None in every row). Imputation is "
               "impossible without inventing values.",
               columns_dropped=len(fully_null))

    # -- 6. parse the CPU distribution arrays ------------------------------
    if "cpu_usage_distribution" in df.columns:
        mat = parse_array_column(df["cpu_usage_distribution"], config.CPU_DIST_LEN)
        for i in range(config.CPU_DIST_LEN):
            df[f"cpu_dist_p{i}"] = mat[:, i]
        n_bad = int(np.isnan(mat).all(axis=1).sum())
        df = df.drop(columns=["cpu_usage_distribution"])
        record("parse_cpu_distribution",
               f"expanded into {config.CPU_DIST_LEN} decile columns "
               f"({n_bad:,} rows had no array)",
               "The field stores an 11-point CPU-usage distribution (deciles). "
               "Expanding it exposes the *shape* of demand within the "
               "measurement window, which a single average hides -- and shape "
               "is where intermittent stalls show up.",
               columns_added=config.CPU_DIST_LEN)

    if "tail_cpu_usage_distribution" in df.columns:
        mat = parse_array_column(df["tail_cpu_usage_distribution"], config.TAIL_DIST_LEN)
        for i in range(config.TAIL_DIST_LEN):
            df[f"cpu_tail_p{i}"] = mat[:, i]
        df = df.drop(columns=["tail_cpu_usage_distribution"])
        record("parse_tail_cpu_distribution",
               f"expanded into {config.TAIL_DIST_LEN} tail-percentile columns",
               "Tail percentiles capture worst-case CPU behaviour within the "
               "window -- the natural quantitative signature of a latency "
               "problem that a mean would mask.",
               columns_added=config.TAIL_DIST_LEN)

    # -- 7. structural list columns -> counts ------------------------------
    if "constraint" in df.columns:
        df["n_constraints"] = count_list_items(df["constraint"], "'name'")
        df["has_constraint"] = (df["n_constraints"] > 0).astype("int8")
        df = df.drop(columns=["constraint"])
        record("encode_constraints",
               "constraint -> n_constraints, has_constraint",
               "The raw field is a variable-length list of hashed placement "
               "constraints. The hashes are not comparable across collections, "
               "but the *number* of constraints is a meaningful measure of "
               "scheduling restriction.", columns_added=2)

    if "start_after_collection_ids" in df.columns:
        df["n_dependencies"] = count_ids(df["start_after_collection_ids"])
        df["has_dependency"] = (df["n_dependencies"] > 0).astype("int8")
        df = df.drop(columns=["start_after_collection_ids"])
        record("encode_dependencies",
               "start_after_collection_ids -> n_dependencies, has_dependency",
               "98.72% of rows have no dependency; the count captures the "
               "informative minority without one-hot exploding arbitrary ids.",
               columns_added=2)

    # -- 8. time sentinels -------------------------------------------------
    if "time" in df.columns:
        t = pd.to_numeric(df["time"], errors="coerce")
        zero = (t == config.TIME_SENTINEL_MIN)
        maxi = (t == config.TIME_SENTINEL_MAX)
        df["time_is_valid"] = (~(zero | maxi) & t.notna()).astype("int8")
        df["time_sentinel"] = np.select(
            [zero, maxi], ["before_trace", "after_trace"], default="valid")
        df["time_us"] = t.where(df["time_is_valid"] == 1)
        record("handle_time_sentinels",
               f"time==0: {int(zero.sum()):,} rows, time==2**63-1: {int(maxi.sum()):,} rows "
               "-> set to NaN in time_us; status kept in time_sentinel",
               "Borg encodes 'event outside the trace window' with these "
               "sentinels. They are not measurements. Leaving them numeric "
               "would inject a 2**63 magnitude outlier and create a spurious "
               "predictor (fail rate 65.7% at time==0 vs 15.9% for real "
               "timestamps) that reflects bookkeeping, not system behaviour.",
               columns_added=3)

    # -- 9. numeric validation: negatives and infinities -------------------
    non_negative = [c for c in df.columns if any(
        tok in c for tok in ("usage", "memory", "cpu", "sample_rate",
                             "cycles_per_instruction", "assigned"))]
    n_inf, n_neg = 0, 0
    for c in non_negative:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        col = df[c]
        inf_mask = np.isinf(col.to_numpy(dtype="float64", na_value=np.nan))
        if inf_mask.any():
            n_inf += int(inf_mask.sum())
            df.loc[inf_mask, c] = np.nan
        neg_mask = col < 0
        if neg_mask.any():
            n_neg += int(neg_mask.sum())
            df.loc[neg_mask, c] = np.nan
    record("validate_numeric_ranges",
           f"set {n_inf:,} infinite and {n_neg:,} physically-impossible negative "
           "values to NaN across resource columns",
           "A negative CPU or memory usage, or an infinity, cannot be a real "
           "measurement. These are converted to NaN so the documented "
           "imputation policy handles them, rather than being clipped to 0 "
           "(which would silently assert 'idle').",
           values_fixed=n_inf + n_neg)

    # -- 10. missing-value policy, column by column ------------------------
    miss_before = df.isna().sum()
    miss_before = miss_before[miss_before > 0]

    # (a) performance counters: informative missingness
    perf = [c for c in ["cycles_per_instruction", "memory_accesses_per_instruction"]
            if c in df.columns]
    if perf:
        both_missing = df[perf].isna().all(axis=1)
        df["perf_counters_missing"] = both_missing.astype("int8")
        for c in perf:
            if "cluster" in df.columns:
                med = df.groupby("cluster")[c].transform("median")
                df[c] = df[c].fillna(med)
            df[c] = df[c].fillna(df[c].median())
        record("impute_perf_counters",
               f"{perf} imputed with cluster median; added 'perf_counters_missing' flag "
               f"({int(both_missing.sum()):,} rows = {100 * both_missing.mean():.2f}%)",
               "Missingness here is NOT random: rows lacking these counters "
               "fail 12.9% of the time versus 27.3% for rows that have them. "
               "Deleting the rows would remove 30.7% of the dataset and shift "
               "the class balance; imputing silently would erase a real signal. "
               "Keeping an explicit indicator lets the models use the absence "
               "itself as evidence.", columns_added=1)

    # (b) CPU distribution deciles / tails: row-wise then global
    dist_cols = [c for c in df.columns if c.startswith(("cpu_dist_p", "cpu_tail_p"))]
    if dist_cols:
        n_missing_rows = int(df[dist_cols].isna().all(axis=1).sum())
        df["cpu_dist_missing"] = df[dist_cols].isna().all(axis=1).astype("int8")
        # If the distribution is absent, the best available estimate of every
        # percentile is that row's own mean usage -- an internally consistent
        # fallback rather than a global constant.
        if "average_usage_cpus" in df.columns:
            fallback = df["average_usage_cpus"]
            for c in dist_cols:
                df[c] = df[c].fillna(fallback)
        for c in dist_cols:
            df[c] = df[c].fillna(df[c].median())
        record("impute_cpu_distributions",
               f"{len(dist_cols)} percentile columns; {n_missing_rows:,} rows had no "
               "array -> filled from that row's own average CPU usage, then median",
               "Using the row's own mean keeps the imputed percentile vector "
               "consistent with the row's other measurements instead of "
               "importing another workload's shape.", columns_added=1)

    # (c) remaining low-rate numeric gaps: median imputation
    imputed = {}
    for c in df.columns:
        if df[c].isna().any() and pd.api.types.is_numeric_dtype(df[c]):
            frac = float(df[c].isna().mean())
            med = df[c].median()
            if not np.isfinite(med):
                med = 0.0
            df[c] = df[c].fillna(med)
            imputed[c] = (frac, med)
    if imputed:
        detail = ", ".join(f"{c} ({100 * f:.2f}%)" for c, (f, _) in
                           sorted(imputed.items(), key=lambda kv: -kv[1][0])[:8])
        record("impute_remaining_numeric",
               f"median-imputed {len(imputed)} numeric columns: {detail}"
               + (" ..." if len(imputed) > 8 else ""),
               "All remaining gaps are low-rate (<= ~1.3%). The median is used "
               "rather than the mean because every resource column is strongly "
               "right-skewed, so the mean sits well above the typical value. "
               "Note time_us is intentionally left OUT of model features, so "
               "its sentinel-induced gaps never enter training.")

    # (d) categorical gaps
    cat_filled = []
    for c in df.select_dtypes(include=["object"]).columns:
        if df[c].isna().any():
            df[c] = df[c].fillna("UNKNOWN")
            cat_filled.append(c)
    if cat_filled:
        record("impute_categorical", f"filled {cat_filled} with 'UNKNOWN'",
               "An explicit 'UNKNOWN' level preserves the fact that the value "
               "was absent instead of hiding it inside the modal category.")

    # -- 11. explicit note on what was NOT done ----------------------------
    record("no_global_dropna", "df.dropna() was deliberately NOT used",
           "A global dropna() would have removed roughly 31% of rows (driven by "
           "the performance counters) and, because that missingness correlates "
           "with the target, would have altered the class balance while looking "
           "like innocuous cleaning.")

    record("outliers_retained", "extreme resource values retained (not winsorised)",
           "The IQR rule flags a large share of resource rows purely because "
           "the distributions are heavy-tailed. Those tails are genuine "
           "workload behaviour and are precisely where degradation appears, so "
           "trimming them would delete the phenomenon under study. Robust "
           "(median/MAD) statistics are used downstream instead, which tolerate "
           "the tails without discarding data.")

    # -- 12. memory-conscious dtypes ---------------------------------------
    protect = {"time", "time_us", "machine_id", "collection_id",
               "alloc_collection_id", "start_time", "end_time"}
    before_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    for c in df.columns:
        if c in protect:
            continue
        if df[c].dtype == "float64":
            df[c] = pd.to_numeric(df[c], downcast="float")
        elif df[c].dtype == "int64" and df[c].abs().max() < 2 ** 31:
            df[c] = pd.to_numeric(df[c], downcast="integer")
    after_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    record("downcast_dtypes", f"memory {before_mb:,.0f} MB -> {after_mb:,.0f} MB",
           "float64 -> float32 and int64 -> smallest safe int. ID and timestamp "
           "columns are excluded from downcasting because machine_id (1.7e11) "
           "and time (2.7e12) exceed 32-bit range and would be corrupted.")

    n1, k1 = df.shape
    summary = {
        "original_rows": n0, "original_columns": k0,
        "final_rows": n1, "final_columns": k1,
        "rows_removed": n0 - n1,
        "rows_removed_pct": round(100.0 * (n0 - n1) / max(n0, 1), 4),
        "duplicate_rows_found": n_full_dup,
        "missing_values_before": int(miss_before.sum()),
        "missing_values_after": int(df.isna().sum().sum()),
        "memory_mb": round(after_mb, 1),
        "decisions": log,
    }
    return df, summary


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(summary: dict, df_before: pd.DataFrame | None = None) -> str:
    buf = io.StringIO()

    def w(s: str = ""):
        buf.write(s + "\n")

    w("=" * 78)
    w(" PREPROCESSING REPORT")
    w("=" * 78)
    w()
    w("HEADLINE NUMBERS")
    w("-" * 78)
    w(f"Original rows            : {summary['original_rows']:,}")
    w(f"Rows removed             : {summary['rows_removed']:,} "
      f"({summary['rows_removed_pct']}%)")
    w(f"Duplicate rows found     : {summary['duplicate_rows_found']:,}")
    w(f"Missing values (before)  : {summary['missing_values_before']:,}")
    w(f"Missing values (after)   : {summary['missing_values_after']:,}")
    w(f"Rows after cleaning      : {summary['final_rows']:,}")
    w(f"Columns before           : {summary['original_columns']}")
    w(f"Columns after            : {summary['final_columns']}")
    w(f"Cleaned frame in memory  : {summary['memory_mb']:,.1f} MB")
    w()
    w("KEY POLICY: no global dropna(). Missingness in this trace is structured")
    w("and partly informative, so it is handled per column with an explicit")
    w("rationale, and indicator flags preserve the information that a value was")
    w("absent.")
    w()
    w("=" * 78)
    w(" DECISION LOG  (in execution order)")
    w("=" * 78)
    for i, d in enumerate(summary["decisions"], 1):
        w()
        w(f"[{i:02d}] {d['step']}")
        w(f"     ACTION    : {d['action']}")
        w(f"     RATIONALE : {_wrap(d['rationale'], 66, 17)}")
        extra = {k: v for k, v in d.items()
                 if k not in ("step", "action", "rationale")}
        if extra:
            w(f"     COUNTS    : {extra}")
    w()
    text = buf.getvalue()
    save_text(text, config.REPORT_DIR / "preprocessing_report.txt")

    stats = pd.DataFrame([{k: v for k, v in summary.items() if k != "decisions"}])
    stats.to_csv(config.METRICS_DIR / "preprocessing_statistics.csv", index=False)
    return text


def _wrap(text: str, width: int, indent: int) -> str:
    import textwrap
    lines = textwrap.wrap(text, width=width)
    pad = " " * indent
    return ("\n" + pad).join(lines) if lines else ""
