"""
feature_engineering.py
======================
STAGE 3 -- construct scientifically meaningful features.

Two ideas govern this module.

**1. Every feature is tagged, and the tag is enforced.**
Each engineered column is registered with a group and a ``model_safe`` flag.
Anything derived from the target (e.g. a machine's historical failure rate) is
tagged ``model_safe=False``: it remains available for descriptive ranking and
dashboards, but :func:`model_feature_columns` refuses to hand it to a
classifier. This makes leakage control a property of the code rather than a
promise in a comment.

**2. Features are only created where the data supports them.**
The brief asks for per-machine rolling means and machine degradation scores.
Inspection showed machines carry a median of 4 events (max 20), so rolling
windows and trend slopes per machine would be numerically computable but
statistically empty. Rolling/trend features are therefore built at the
**collection** grain (median 6 events, up to 9,828; 1,320 collections have
>= 10 timestamped events), and machines are characterised **cross-sectionally**
against their cluster peers -- a machine-level statistic that its sample size
can actually carry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .utils import coefficient_of_variation, info, robust_z, safe_div, section, slope, warn

# ---------------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------------

#: name -> (group, model_safe, description)
REGISTRY: dict[str, tuple[str, bool, str]] = {}


def register(name: str, group: str, model_safe: bool, description: str) -> str:
    REGISTRY[name] = (group, model_safe, description)
    return name


def registry_frame() -> pd.DataFrame:
    rows = [{"feature": k, "group": v[0], "model_safe": v[1], "description": v[2]}
            for k, v in REGISTRY.items()]
    return pd.DataFrame(rows).sort_values(["group", "feature"]).reset_index(drop=True)


_EPS = 1e-9


# ---------------------------------------------------------------------------
# 1. CPU features
# ---------------------------------------------------------------------------

def add_cpu_features(df: pd.DataFrame) -> pd.DataFrame:
    """CPU demand level, shape, efficiency and variability."""
    avg = df.get("average_usage_cpus")
    mx = df.get("maximum_usage_cpus")
    req = df.get("resource_request_cpus")
    if avg is None or mx is None:
        warn("CPU usage columns absent -- CPU feature block skipped.")
        return df

    df[register("cpu_avg", "cpu", True,
                "Mean CPU usage within the 5-minute measurement window "
                "(normalised Borg CPU units).")] = avg

    df[register("cpu_max", "cpu", True,
                "Peak CPU usage within the window.")] = mx

    df[register("cpu_range", "cpu", True,
                "cpu_max - cpu_avg: absolute headroom between peak and mean "
                "demand; a wide range means bursty rather than steady work.")] = mx - avg

    df[register("cpu_burstiness", "cpu", True,
                "cpu_max / cpu_avg: how many times its own average a task peaks "
                "at. Scale-free, so a tiny task and a huge task are comparable. "
                "High values indicate spiky, contention-prone behaviour.")] = \
        safe_div(mx, avg + _EPS, fill=np.nan)

    df[register("cpu_efficiency", "cpu", True,
                "cpu_avg / cpu_max: the fraction of its own peak a task "
                "sustains. Low values mean the task rarely uses the capacity it "
                "actually demands -- the classic profile of stalled or blocked "
                "work rather than genuinely heavy computation.")] = \
        safe_div(avg, mx + _EPS, fill=np.nan)

    if req is not None:
        df[register("cpu_request", "cpu", True,
                    "CPU the collection asked the scheduler for.")] = req
        df[register("cpu_utilization_ratio", "cpu", True,
                    "cpu_avg / cpu_request: how much of the reserved CPU is "
                    "actually used. << 1 = over-provisioned or starved work; "
                    "> 1 = the task exceeds its request and is a candidate for "
                    "throttling.")] = safe_div(avg, req + _EPS, fill=np.nan)
        df[register("cpu_request_overshoot", "cpu", True,
                    "cpu_max - cpu_request: peak demand above the reservation. "
                    "Positive values are moments when the task outgrew its "
                    "contract.")] = mx - req

    # --- shape of demand, from the 11 decile columns ---------------------
    dist_cols = [c for c in df.columns if c.startswith("cpu_dist_p")]
    if len(dist_cols) >= 3:
        D = df[dist_cols].to_numpy(dtype="float64")
        d_std = np.nanstd(D, axis=1)
        d_mean = np.nanmean(D, axis=1)
        df[register("cpu_dist_std", "cpu", True,
                    "Standard deviation across the 11 within-window CPU "
                    "deciles: dispersion of demand *inside* one measurement "
                    "window, which a single average cannot show.")] = d_std
        df[register("cpu_dist_cv", "cpu", True,
                    "Coefficient of variation of the decile curve "
                    "(std / mean): scale-free instability of CPU demand.")] = \
            safe_div(d_std, d_mean + _EPS, fill=np.nan)
        df[register("cpu_dist_iqr", "cpu", True,
                    "Inter-decile spread (p8 - p2) of the within-window CPU "
                    "distribution: a robust width measure that ignores the two "
                    "extreme deciles.")] = D[:, 8] - D[:, 2]
        df[register("cpu_dist_skew", "cpu", True,
                    "(p10 - p5) - (p5 - p0): asymmetry of the demand curve. "
                    "Positive = occasional spikes above a calm baseline.")] = \
            (D[:, 10] - D[:, 5]) - (D[:, 5] - D[:, 0])
        df[register("cpu_p50_to_p100", "cpu", True,
                    "Median-to-maximum decile ratio: how far the worst moment "
                    "sits above the typical moment within the window.")] = \
            safe_div(D[:, 5], D[:, 10] + _EPS, fill=np.nan)

    # --- tail behaviour ---------------------------------------------------
    tail_cols = [c for c in df.columns if c.startswith("cpu_tail_p")]
    if len(tail_cols) >= 3 and avg is not None:
        T = df[tail_cols].to_numpy(dtype="float64")
        tail_max = np.nanmax(T, axis=1)
        df[register("cpu_tail_max", "cpu", True,
                    "Highest tail-percentile CPU value recorded in the window.")] = tail_max
        df[register("cpu_tail_gap", "cpu", True,
                    "cpu_tail_max - cpu_avg: the amount of CPU behaviour hidden "
                    "behind a healthy-looking average. This is the quantitative "
                    "signature of a latency problem that mean-based monitoring "
                    "misses -- i.e. a silent one.")] = tail_max - avg.to_numpy()
        df[register("cpu_tail_ratio", "cpu", True,
                    "cpu_tail_max / cpu_avg: scale-free version of the tail "
                    "gap.")] = safe_div(tail_max, avg.to_numpy() + _EPS, fill=np.nan)

    if "random_sample_usage_cpus" in df.columns:
        df[register("cpu_sample_deviation", "cpu", True,
                    "|random_sample_usage - cpu_avg|: disagreement between an "
                    "instantaneous probe and the window mean, an independent "
                    "read on volatility.")] = (df["random_sample_usage_cpus"] - avg).abs()
    return df


# ---------------------------------------------------------------------------
# 2. Memory features
# ---------------------------------------------------------------------------

def add_memory_features(df: pd.DataFrame) -> pd.DataFrame:
    avg_m = df.get("average_usage_memory")
    max_m = df.get("maximum_usage_memory")
    assigned = df.get("assigned_memory")
    cache = df.get("page_cache_memory")
    req_m = df.get("resource_request_memory")
    if avg_m is None:
        warn("Memory usage columns absent -- memory feature block skipped.")
        return df

    df[register("mem_avg", "memory", True, "Mean memory usage in the window.")] = avg_m
    if max_m is not None:
        df[register("mem_max", "memory", True, "Peak memory usage in the window.")] = max_m
        df[register("mem_range", "memory", True,
                    "mem_max - mem_avg: growth headroom used within a single "
                    "window; a persistent positive gap is the shape of a leak.")] = max_m - avg_m
        df[register("mem_efficiency", "memory", True,
                    "mem_avg / mem_max: how steadily memory is held. Unlike "
                    "CPU, memory is rarely released, so a low ratio points to "
                    "transient allocation spikes.")] = safe_div(avg_m, max_m + _EPS, fill=np.nan)

    if assigned is not None:
        df[register("mem_assigned", "memory", True,
                    "Memory the scheduler assigned to the instance.")] = assigned
        df[register("memory_pressure", "memory", True,
                    "mem_avg / assigned_memory: fraction of the assigned "
                    "allocation actually resident. Values approaching 1 mean "
                    "the instance is operating near its memory ceiling, where "
                    "OOM kills and allocation stalls occur.")] = \
            safe_div(avg_m, assigned + _EPS, fill=np.nan)
        if max_m is not None:
            df[register("memory_peak_pressure", "memory", True,
                        "mem_max / assigned_memory: worst-case occupancy of the "
                        "assignment; the value that actually triggers an OOM.")] = \
                safe_div(max_m, assigned + _EPS, fill=np.nan)

    if cache is not None:
        df[register("page_cache", "memory", True, "Page-cache memory held.")] = cache
        df[register("cache_ratio", "memory", True,
                    "page_cache / mem_avg: share of memory serving as file "
                    "cache. A collapsing cache ratio under constant load is a "
                    "recognised early sign of memory starvation, because the "
                    "kernel evicts cache before it kills the process.")] = \
            safe_div(cache, avg_m + _EPS, fill=np.nan)
        if assigned is not None:
            df[register("cache_to_assigned", "memory", True,
                        "page_cache / assigned_memory: cache footprint relative "
                        "to the allocation.")] = safe_div(cache, assigned + _EPS, fill=np.nan)

    if req_m is not None:
        df[register("mem_request", "memory", True, "Memory requested from the scheduler.")] = req_m
        df[register("mem_utilization_ratio", "memory", True,
                    "mem_avg / mem_request: realised versus reserved memory.")] = \
            safe_div(avg_m, req_m + _EPS, fill=np.nan)

    # cross-resource
    if "average_usage_cpus" in df.columns:
        df[register("mem_to_cpu_ratio", "memory", True,
                    "mem_avg / cpu_avg: workload character. A rising ratio at "
                    "constant CPU means memory is accumulating without extra "
                    "work being done -- the textbook profile of a leak.")] = \
            safe_div(avg_m, df["average_usage_cpus"] + _EPS, fill=np.nan)

    if "memory_accesses_per_instruction" in df.columns:
        df[register("memory_access_intensity", "memory", True,
                    "Memory accesses per instruction (from hardware counters): "
                    "how memory-bound the work is. High MAI with low CPU "
                    "efficiency indicates memory-bandwidth starvation.")] = \
            df["memory_accesses_per_instruction"]
        if "cycles_per_instruction" in df.columns:
            df[register("stall_intensity", "memory", True,
                        "CPI x MAI: a combined index of cycles lost to memory "
                        "stalls. Elevated values mean the CPU is waiting on "
                        "memory rather than computing.")] = \
                df["cycles_per_instruction"] * df["memory_accesses_per_instruction"]
    return df


# ---------------------------------------------------------------------------
# 3. Performance-counter features
# ---------------------------------------------------------------------------

def add_perf_features(df: pd.DataFrame) -> pd.DataFrame:
    if "cycles_per_instruction" in df.columns:
        cpi = df["cycles_per_instruction"]
        df[register("cpi", "perf", True,
                    "Cycles per instruction. The direct microarchitectural "
                    "measure of execution efficiency: identical work taking "
                    "more cycles means the machine is degrading (cache "
                    "contention, memory pressure, thermal throttling, or a "
                    "noisy neighbour).")] = cpi
        if "cluster" in df.columns:
            grp = df.groupby("cluster")["cycles_per_instruction"]
            med = grp.transform("median")
            df[register("cpi_vs_cluster", "perf", True,
                        "cpi / cluster-median cpi: this instance's execution "
                        "efficiency relative to its own cell. Comparing within "
                        "a cluster controls for hardware-generation differences "
                        "between cells, so the residual reflects local "
                        "conditions rather than fleet heterogeneity.")] = \
                safe_div(cpi, med + _EPS, fill=np.nan)
    if "sample_rate" in df.columns:
        sr = df["sample_rate"]
        df[register("sample_rate", "perf", True,
                    "Fraction of expected telemetry samples actually collected.")] = sr
        df[register("telemetry_loss", "perf", True,
                    "1 - sample_rate: share of monitoring samples that never "
                    "arrived. This is itself a silent-failure signature -- the "
                    "observability layer degrading without raising an alarm.")] = 1.0 - sr
        df[register("telemetry_degraded", "perf", True,
                    "Flag for sample_rate < 1.0 (7.34% of rows).")] = (sr < 1.0).astype("int8")
    if "perf_counters_missing" in df.columns:
        register("perf_counters_missing", "perf", True,
                 "Indicator that hardware counters were unavailable for this "
                 "row (30.72%). Retained as a feature because the missingness "
                 "is informative rather than random.")
    if "cpu_dist_missing" in df.columns:
        register("cpu_dist_missing", "perf", True,
                 "Indicator that the within-window CPU distribution was absent.")
    return df


# ---------------------------------------------------------------------------
# 4. Task / scheduling features
# ---------------------------------------------------------------------------

def add_task_features(df: pd.DataFrame) -> pd.DataFrame:
    if "start_time" in df.columns and "end_time" in df.columns:
        dur = (df["end_time"] - df["start_time"]).astype("float64")
        df[register("window_duration_sec", "task", True,
                    "Length of the usage-measurement window in seconds. The "
                    "nominal Borg window is 300 s; a truncated window means the "
                    "instance stopped mid-measurement.")] = dur / config.US_PER_SEC
        df[register("window_truncated", "task", True,
                    "Flag for a window shorter than the nominal 300 s "
                    "(35.6% of rows) -- i.e. the measurement was cut short.")] = \
            (dur < config.NOMINAL_WINDOW_US).astype("int8")
        df[register("window_completeness", "task", True,
                    "duration / 300 s: fraction of a full measurement window "
                    "that was actually observed.")] = dur / config.NOMINAL_WINDOW_US

    if "priority" in df.columns:
        pr = df["priority"]
        df[register("priority", "task", True, "Raw Borg priority value.")] = pr
        tier = np.select(
            [pr.between(lo, hi) for lo, hi, _, _ in config.PRIORITY_TIERS],
            [t for _, _, t, _ in config.PRIORITY_TIERS],
            default=-1)
        df[register("priority_tier", "task", True,
                    "Ordinal Borg priority band (0 free, 1 best-effort batch, "
                    "2 mid, 3 production, 4 monitoring). Encodes the documented "
                    "scheduler semantics instead of treating priority as an "
                    "arbitrary integer: free-tier work is evictable by design, "
                    "production work is not.")] = tier.astype("int8")
        df[register("is_production", "task", True,
                    "Flag for production-or-above priority (>= 120), where a "
                    "failure has real service impact.")] = (pr >= 120).astype("int8")
        df[register("is_evictable", "task", True,
                    "Flag for free-tier priority (< 100), which Borg may evict "
                    "without it indicating a fault.")] = (pr < 100).astype("int8")

    for c, desc in [
        ("scheduling_class", "Latency sensitivity 0-3 (3 = most latency-sensitive)."),
        ("collection_type", "0 = job, 1 = alloc set."),
        ("vertical_scaling", "Vertical autoscaling mode selected for the collection."),
        ("scheduler", "Which scheduler variant placed the work."),
    ]:
        if c in df.columns:
            register(c, "task", True, desc)

    for c, desc in [
        ("n_constraints", "Number of placement constraints attached to the request."),
        ("has_constraint", "Whether any placement constraint exists."),
        ("n_dependencies", "Number of collections that must start first."),
        ("has_dependency", "Whether the collection has start-after dependencies."),
    ]:
        if c in df.columns:
            register(c, "task", True, desc)

    # combined resource intensity
    if {"cpu_avg", "mem_avg"} <= set(df.columns):
        df[register("resource_intensity", "task", True,
                    "sqrt(cpu_avg^2 + mem_avg^2): overall resource footprint as "
                    "a single magnitude in CPU-memory space.")] = \
            np.sqrt(df["cpu_avg"].astype("float64") ** 2 + df["mem_avg"].astype("float64") ** 2)

    if {"cpu_utilization_ratio", "mem_utilization_ratio"} <= set(df.columns):
        df[register("provisioning_mismatch", "task", True,
                    "|1 - cpu_util_ratio| + |1 - mem_util_ratio|: total "
                    "divergence between what was requested and what is used. "
                    "Large values mean the instance is starved or grossly "
                    "over-provisioned -- both are unhealthy states that no "
                    "explicit alarm reports.")] = \
            (1 - df["cpu_utilization_ratio"]).abs() + (1 - df["mem_utilization_ratio"]).abs()
    return df


# ---------------------------------------------------------------------------
# 5. Temporal features
# ---------------------------------------------------------------------------

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calendar / position-in-trace features derived from valid timestamps only.

    Sentinel rows (time == 0 or 2**63-1) receive NaN here and are then median-
    imputed, so they cannot masquerade as a real point in time.
    """
    if "time_us" not in df.columns:
        warn("No usable time column -- temporal block skipped.")
        return df

    t = pd.to_numeric(df["time_us"], errors="coerce")
    t0 = t.min()
    df[register("t_hours_from_start", "temporal", True,
                "Hours since the first observed event in the trace.")] = \
        (t - t0) / config.US_PER_HOUR
    df[register("hour_of_day", "temporal", True,
                "Hour of day (0-23) derived from the trace clock: captures "
                "diurnal load cycles in the cluster.")] = \
        ((t / config.US_PER_HOUR) % 24).astype("float32")
    df[register("day_index", "temporal", True,
                "Day number within the 31-day trace window.")] = \
        (t / (24 * config.US_PER_HOUR)).astype("float32")
    df[register("is_weekend_like", "temporal", True,
                "Flag for days 5-6 of each 7-day cycle in trace time. Labelled "
                "'like' because the trace is anonymised and its absolute "
                "calendar alignment is unknown -- so this is a periodicity "
                "probe, not a verified weekday claim.")] = \
        (((t / (24 * config.US_PER_HOUR)).fillna(0).astype("int64") % 7) >= 5).astype("int8")

    if "time_is_valid" in df.columns:
        register("time_is_valid", "temporal", True,
                 "Whether this row carries a real timestamp rather than a "
                 "trace-boundary sentinel. Kept as an honest flag; the raw "
                 "sentinel magnitude is never fed to a model.")
    return df


# ---------------------------------------------------------------------------
# 6. Collection-level rolling / trend features
# ---------------------------------------------------------------------------

def add_collection_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling and change features at the **collection** grain.

    Justification for this grain (from measured support, in
    ``outputs/reports/data_profile.txt``):
        machines    -> median 4 events   => rolling windows meaningless
        collections -> median 6, max 9,828; 1,320 have >= 10 timestamped events

    All rolling statistics use ``shift(1)`` so a row never sees its own value.
    That is what makes them usable as predictors: the feature describes the
    *past* of the collection, never the present row.
    """
    if "collection_id" not in df.columns:
        warn("No collection_id -- collection temporal block skipped.")
        return df

    sort_key = "time_us" if "time_us" in df.columns else "start_time"
    df = df.sort_values(["collection_id", sort_key], kind="mergesort").reset_index(drop=True)
    g = df.groupby("collection_id", sort=False)

    base = [c for c in ["cpu_avg", "mem_avg", "cpu_efficiency", "memory_pressure", "cpi"]
            if c in df.columns]
    W = config.ROLLING_WINDOW

    for c in base:
        s = df[c].astype("float64")
        shifted = g[c].shift(1)

        df[register(f"{c}_roll_mean", "temporal_collection", True,
                    f"Rolling mean of {c} over the previous {W} events of the "
                    "same collection (shifted, so the current row is excluded). "
                    "Represents the collection's recent baseline.")] = \
            shifted.groupby(df["collection_id"]).transform(
                lambda x: x.rolling(W, min_periods=2).mean())

        df[register(f"{c}_roll_std", "temporal_collection", True,
                    f"Rolling standard deviation of {c} over the previous {W} "
                    "events: recent instability of this collection.")] = \
            shifted.groupby(df["collection_id"]).transform(
                lambda x: x.rolling(W, min_periods=2).std())

        df[register(f"{c}_roll_median", "temporal_collection", True,
                    f"Rolling median of {c} (outlier-resistant baseline).")] = \
            shifted.groupby(df["collection_id"]).transform(
                lambda x: x.rolling(W, min_periods=2).median())

        df[register(f"{c}_delta", "temporal_collection", True,
                    f"{c} minus its value at the previous event of the same "
                    "collection: step change between consecutive observations.")] = \
            s - shifted

        df[register(f"{c}_pct_change", "temporal_collection", True,
                    f"Relative change in {c} versus the previous event "
                    "(scale-free).")] = safe_div(s - shifted, shifted.abs() + _EPS, fill=np.nan)

        df[register(f"{c}_vs_baseline", "temporal_collection", True,
                    f"{c} divided by its own rolling mean: deviation from the "
                    "collection's recent normal. Values persistently above 1 "
                    "are the definition of drift used in this study.")] = \
            safe_div(s, df[f"{c}_roll_mean"] + _EPS, fill=np.nan)

        df[register(f"{c}_zscore_within_collection", "temporal_collection", True,
                    f"({c} - rolling mean) / rolling std: how many standard "
                    "deviations the current value sits from the collection's "
                    "own recent behaviour. Self-referential, so it needs no "
                    "cross-workload threshold.")] = \
            safe_div(s - df[f"{c}_roll_mean"], df[f"{c}_roll_std"] + _EPS, fill=np.nan)

    df[register("collection_event_index", "temporal_collection", True,
                "Position of this event within its collection's history "
                "(0-based). Distinguishes a first observation from a "
                "long-running one.")] = g.cumcount()

    df[register("collection_size", "temporal_collection", True,
                "Total number of events recorded for this collection.")] = \
        g["collection_id"].transform("size")

    if sort_key in df.columns:
        gap = g[sort_key].diff() / config.US_PER_SEC
        df[register("time_since_prev_event", "temporal_collection", True,
                    "Seconds since the previous event of the same collection. "
                    "A shrinking gap means events are arriving faster, which "
                    "typically accompanies instability.")] = gap
    return df


# ---------------------------------------------------------------------------
# 7. Machine features  (cross-sectional -- see module docstring)
# ---------------------------------------------------------------------------

def add_machine_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Characterise machines by comparison with their cluster peers.

    ``machine_failure_rate`` is computed because the brief asks for it and it is
    genuinely useful for the machine-ranking deliverable -- but it is derived
    from the target, so it is registered with ``model_safe=False`` and is
    mechanically excluded from every supervised feature matrix.
    """
    if "machine_id" not in df.columns:
        warn("No machine_id -- machine block skipped.")
        return df

    gm = df.groupby("machine_id", sort=False)

    df[register("machine_event_count", "machine", True,
                "Number of trace events observed on this machine (median 4).")] = \
        gm["machine_id"].transform("size")

    for src, name, desc in [
        ("cpu_avg", "machine_cpu_mean", "Mean CPU usage across everything observed on this machine."),
        ("mem_avg", "machine_mem_mean", "Mean memory usage across everything observed on this machine."),
        ("cpi", "machine_cpi_mean", "Mean cycles-per-instruction on this machine -- a hardware-level "
                                    "efficiency signature."),
    ]:
        if src in df.columns:
            df[register(name, "machine", True, desc)] = gm[src].transform("mean")

    for src, name, desc in [
        ("cpu_avg", "machine_cpu_std", "Dispersion of CPU usage on this machine."),
        ("mem_avg", "machine_mem_std", "Dispersion of memory usage on this machine."),
    ]:
        if src in df.columns:
            df[register(name, "machine", True, desc)] = gm[src].transform("std")

    if {"machine_cpu_std", "machine_cpu_mean"} <= set(df.columns):
        df[register("machine_cpu_cv", "machine", True,
                    "Coefficient of variation of CPU on this machine: "
                    "scale-free instability of the machine's workload mix.")] = \
            safe_div(df["machine_cpu_std"], df["machine_cpu_mean"] + _EPS, fill=np.nan)

    # --- peer comparison within the cluster ------------------------------
    if "cluster" in df.columns:
        for src, name, desc in [
            ("cpi", "machine_cpi_vs_peers",
             "Machine mean CPI divided by the median CPI of its cluster. This "
             "is the key cross-sectional degradation signal: a machine that "
             "needs systematically more cycles for the same instruction stream "
             "than its peers is under-performing, even while every task on it "
             "still reports success."),
            ("cpu_efficiency", "machine_efficiency_vs_peers",
             "Machine mean CPU efficiency relative to its cluster median."),
            ("memory_pressure", "machine_mem_pressure_vs_peers",
             "Machine mean memory pressure relative to its cluster median."),
        ]:
            if src not in df.columns:
                continue
            mmean = gm[src].transform("mean")
            cmed = df.groupby("cluster")[src].transform("median")
            df[register(name, "machine", True, desc)] = safe_div(mmean, cmed + _EPS, fill=np.nan)

    # --- target-derived: descriptive ONLY --------------------------------
    if config.TARGET in df.columns:
        df[register("machine_failure_rate", "machine", False,
                    "Share of this machine's observed events that are explicit "
                    "failures. TARGET-DERIVED: excluded from all model features "
                    "(it would leak the label); used only for the descriptive "
                    "machine-risk ranking.")] = gm[config.TARGET].transform("mean")
        df[register("machine_failure_count", "machine", False,
                    "Absolute number of explicit failures on this machine. "
                    "TARGET-DERIVED -- descriptive use only.")] = \
            gm[config.TARGET].transform("sum")
    return df


# ---------------------------------------------------------------------------
# 8. Cluster context
# ---------------------------------------------------------------------------

def add_cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    if "cluster" not in df.columns:
        return df
    # The bare cell identifier is registered as NOT model-safe: it is an
    # arbitrary label, and letting a model key off "cluster == 3" memorises this
    # deployment instead of learning behaviour. Its *derived* statistics below
    # (peer means, and each row's deviation from them) are the legitimate way to
    # use cell membership.
    register("cluster", "cluster", False, "Borg cell identifier (descriptive only).")
    gc = df.groupby("cluster", sort=False)
    for src, name, desc in [
        ("cpu_avg", "cluster_cpu_mean", "Mean CPU usage across the whole cell -- ambient load context."),
        ("mem_avg", "cluster_mem_mean", "Mean memory usage across the whole cell."),
        ("cpi", "cluster_cpi_median", "Median CPI of the cell (hardware-generation baseline)."),
    ]:
        if src in df.columns:
            df[register(name, "cluster", True, desc)] = gc[src].transform(
                "median" if "median" in name else "mean")
    if {"cpu_avg", "cluster_cpu_mean"} <= set(df.columns):
        df[register("cpu_vs_cluster", "cluster", True,
                    "This instance's CPU usage relative to its cell average: "
                    "separates 'the whole cell is busy' from 'this instance is "
                    "behaving abnormally'.")] = \
            safe_div(df["cpu_avg"], df["cluster_cpu_mean"] + _EPS, fill=np.nan)
    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every feature block and return ``(df, registry_frame)``."""
    REGISTRY.clear()
    n_before = df.shape[1]

    section("CPU features");                 df = add_cpu_features(df)
    section("Memory features");              df = add_memory_features(df)
    section("Performance-counter features"); df = add_perf_features(df)
    section("Task / scheduling features");   df = add_task_features(df)
    section("Temporal features");            df = add_temporal_features(df)
    section("Collection-level temporal features"); df = add_collection_temporal_features(df)
    section("Machine features");             df = add_machine_features(df)
    section("Cluster features");             df = add_cluster_features(df)

    # Clean up numerical artefacts created by ratios.
    eng = [c for c in REGISTRY if c in df.columns]
    arr = df[eng].to_numpy(dtype="float64", na_value=np.nan)
    n_inf = int(np.isinf(arr).sum())
    if n_inf:
        df[eng] = df[eng].replace([np.inf, -np.inf], np.nan)
        info(f"replaced {n_inf:,} infinite values produced by ratio features with NaN")

    # Median-fill engineered NaNs (rolling warm-up rows, undefined ratios).
    n_nan = int(df[eng].isna().sum().sum())
    for c in eng:
        if df[c].isna().any():
            med = df[c].median()
            df[c] = df[c].fillna(0.0 if not np.isfinite(med) else med)
    info(f"median-filled {n_nan:,} NaNs in engineered features "
         f"(mostly rolling warm-up rows at the start of each collection)")

    # Downcast the engineered block only; keep IDs/timestamps at 64-bit.
    protect = {"time", "time_us", "machine_id", "collection_id",
               "alloc_collection_id", "start_time", "end_time"}
    for c in eng:
        if c in protect:
            continue
        if df[c].dtype == "float64":
            df[c] = pd.to_numeric(df[c], downcast="float")

    reg = registry_frame()
    n_safe = int(reg.model_safe.sum())
    info(f"engineered {len(reg)} features "
         f"({n_safe} model-safe, {len(reg) - n_safe} descriptive-only) "
         f"| columns {n_before} -> {df.shape[1]}")
    return df, reg


def model_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    The whitelist handed to every supervised model.

    Enforces, in code:
      * only registered engineered features (nothing raw slips through),
      * ``model_safe=True`` only (no target-derived columns),
      * never a leakage column, an identity hash, or a structural key,
      * numeric and non-constant.
    """
    banned = set(config.TARGET_LEAKAGE_COLS) | set(config.IDENTITY_COLS) \
        | set(config.KEY_COLS) | {config.TARGET, "time", "time_us", "time_sentinel",
                                  "start_time", "end_time", "instance_index"}
    cols = []
    for name, (_grp, safe, _desc) in REGISTRY.items():
        if not safe or name in banned or name not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[name]):
            continue
        if df[name].nunique(dropna=False) <= 1:
            continue
        cols.append(name)
    return sorted(set(cols))
