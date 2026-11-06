"""
tests/conftest.py
=================
Shared fixtures. The synthetic frame below mimics the *real* Borg schema --
including the two things that make this dataset awkward: Python-literal dict
columns and NumPy-array reprs stored as strings.

Nothing here touches the real 313 MB CSV, so the suite runs in seconds and works
on a machine that has never downloaded the trace.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from src import config


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(config.RANDOM_STATE)


#: The real file's exact 34 columns, in order, verified against
#: data/raw/borg_traces_data.csv. The fixture below must reproduce this list or
#: the suite stops testing the real pipeline.
REAL_COLUMNS = [
    "Unnamed: 0", "time", "instance_events_type", "collection_id",
    "scheduling_class", "collection_type", "priority", "alloc_collection_id",
    "instance_index", "machine_id", "resource_request", "constraint",
    "collections_events_type", "user", "collection_name",
    "collection_logical_name", "start_after_collection_ids", "vertical_scaling",
    "scheduler", "start_time", "end_time", "average_usage", "maximum_usage",
    "random_sample_usage", "assigned_memory", "page_cache_memory",
    "cycles_per_instruction", "memory_accesses_per_instruction", "sample_rate",
    "cpu_usage_distribution", "tail_cpu_usage_distribution", "cluster", "event",
    "failed",
]


@pytest.fixture
def raw_like(rng) -> pd.DataFrame:
    """A 400-row frame with the real column names, dtypes and quirks."""
    n = 400
    coll = rng.integers(0, 20, n)          # 20 collections -> group structure
    mach = rng.integers(0, 40, n)

    # Borg time sentinels: 0 = before the trace window, 2**63-1 = after.
    t = rng.integers(6_000_000_000, 7_000_000_000, n).astype("int64")
    t[:12] = 0
    t[12:20] = config.TIME_SENTINEL_MAX

    def arr_repr(k: int) -> list[str]:
        # Matches the real format: bracketed, space-separated, sorted ascending.
        return ["[" + " ".join(f"{v:.8f}" for v in np.sort(rng.random(k))) + "]"
                for _ in range(n)]

    avg_cpu = rng.random(n) * 0.05
    avg_mem = rng.random(n) * 0.05

    df = pd.DataFrame({
        "Unnamed: 0": np.arange(n),        # the index artifact the real file has
        "time": t,
        "instance_events_type": rng.choice([0, 1, 4, 5], n),
        "collection_id": coll,
        "scheduling_class": rng.integers(0, 4, n),
        "collection_type": rng.choice([0, 1], n),
        "priority": rng.choice([0, 25, 103, 115, 119, 200, 360], n),
        "alloc_collection_id": rng.integers(0, 5, n),
        "instance_index": rng.integers(0, 100, n),
        "machine_id": mach,
        "resource_request": [
            "{'cpus': %.8f, 'memory': %.8f}" % (rng.random() * .1, rng.random() * .1)
            for _ in range(n)],
        "constraint": rng.choice(["[]", "[{'name': 'x'}]"], n),
        "collections_events_type": rng.choice([0, 1, 4, 5], n),
        "user": rng.choice(["userA", "userB", "userC"], n),
        "collection_name": rng.choice(["jobA", "jobB"], n),
        "collection_logical_name": rng.choice(["lA", "lB"], n),
        "start_after_collection_ids": rng.choice(["[]", "[1, 2, 3]"], n),
        "vertical_scaling": rng.choice([0, 1, 2], n),
        "scheduler": rng.choice([0, 1], n),
        "start_time": t,
        "end_time": t + rng.integers(1, 10_000_000, n),
        "average_usage": [
            "{'cpus': %.8f, 'memory': %.8f}" % (avg_cpu[i], avg_mem[i])
            for i in range(n)],
        "maximum_usage": [
            "{'cpus': %.8f, 'memory': %.8f}" % (avg_cpu[i] * 2, avg_mem[i] * 2)
            for i in range(n)],
        "random_sample_usage": [
            "{'cpus': %.8f}" % (avg_cpu[i] * 1.1) for i in range(n)],
        "assigned_memory": rng.random(n) * 0.05,
        "page_cache_memory": rng.random(n) * 0.01,
        # ~40% null together: they share one hardware-counter collection path,
        # which is why the pipeline flags rather than drops them.
        "cycles_per_instruction": np.where(rng.random(n) < 0.4, np.nan,
                                           rng.random(n) * 3),
        "memory_accesses_per_instruction": np.where(rng.random(n) < 0.4, np.nan,
                                                    rng.random(n) * 0.02),
        "sample_rate": rng.random(n),
        "cpu_usage_distribution": arr_repr(config.CPU_DIST_LEN),
        "tail_cpu_usage_distribution": arr_repr(config.TAIL_DIST_LEN),
        "cluster": rng.integers(1, 9, n),        # real dtype is int64, 1-8
        "event": rng.choice(["SUBMIT", "SCHEDULE", "FINISH", "FAIL", "KILL",
                             "EVICT", "LOST", "ENABLE"], n),
    })
    # Reproduce the documented equivalence exactly:
    #     failed == 1  <=>  event == 'FAIL'  <=>  *_events_type == 2
    df["failed"] = (df["event"] == "FAIL").astype("int8")
    df.loc[df["failed"] == 1, "instance_events_type"] = 2
    df.loc[df["failed"] == 1, "collections_events_type"] = 2
    return df[REAL_COLUMNS]


@pytest.fixture
def cleaned(raw_like):
    from src import preprocessing
    out, summary = preprocessing.clean(raw_like.copy(), profile=None)
    return out, summary


@pytest.fixture
def featured(cleaned):
    from src import feature_engineering as fe
    df, _ = cleaned
    out, registry = fe.build_features(df.copy())
    return out, registry
