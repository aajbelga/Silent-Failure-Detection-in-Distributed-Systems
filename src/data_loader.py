"""
data_loader.py
==============
Loads the raw Borg trace CSV.

Two non-obvious things this module handles, both discovered by inspection:

1.  The CSV contains **multi-line quoted fields**. ``cpu_usage_distribution``
    stores a NumPy array repr such as::

        "[0.00314331 0.00381088 ... 0.01194763]"

    which NumPy wraps across several physical lines. Consequently ``wc -l``
    reports 1,324,695 lines for a file that holds only 405,894 logical records.
    Any line-based reader (or a naive `nrows` estimate) will mis-parse it;
    pandas' C parser handles the embedded newlines correctly.

2.  ``machine_id`` and ``time`` hold values up to 1.7e11 and 2.7e12, so they
    must stay 64-bit. We never downcast them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .utils import PipelineError, info, mem_mb, timer, warn


def check_raw_exists(path: Path | None = None) -> Path:
    """Fail gracefully and helpfully when the dataset is absent."""
    path = Path(path or config.RAW_CSV)
    if not path.exists():
        raise PipelineError(
            f"Raw dataset not found at:\n    {path}\n\n"
            f"Place the Google Borg trace CSV there (expected name: "
            f"'{config.RAW_FILENAME}') and re-run."
        )
    if path.stat().st_size == 0:
        raise PipelineError(f"Raw dataset at {path} is empty (0 bytes).")
    return path


def true_row_count(path: Path | None = None, chunksize: int = 200_000) -> int:
    """
    Count logical records by streaming the file.

    Used in the profiling report to make the point that the physical line count
    (1,324,695) is not the record count (405,894), because of the multi-line
    array fields.
    """
    path = check_raw_exists(path)
    n = 0
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        n += len(chunk)
    return n


def physical_line_count(path: Path | None = None) -> int:
    """Raw newline count, for the contrast noted above."""
    path = check_raw_exists(path)
    n = 0
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(1024 * 1024)
            if not buf:
                break
            n += buf.count(b"\n")
    return n


def load_raw(path: Path | None = None, nrows: int | None = None,
             sample_rows: int | None = None) -> pd.DataFrame:
    """
    Read the raw CSV into a DataFrame.

    Parameters
    ----------
    nrows
        Hard cap passed to pandas -- use for smoke tests only.
    sample_rows
        If set, take a reproducible random subsample *after* reading. Sampling
        after the read (not with ``nrows``) preserves the temporal and
        cluster spread of the trace; ``nrows`` would take only the first N rows,
        which are ordered and therefore biased.
    """
    path = check_raw_exists(path)
    sample_rows = sample_rows if sample_rows is not None else config.SAMPLE_ROWS

    with timer("read_csv"):
        df = pd.read_csv(path, nrows=nrows, low_memory=False)

    if df.empty:
        raise PipelineError(f"Parsed 0 rows from {path}.")

    info(f"loaded {len(df):,} rows x {df.shape[1]} columns  ({mem_mb(df):.0f} MB)")

    if sample_rows and sample_rows < len(df):
        df = df.sample(n=sample_rows, random_state=config.RANDOM_STATE)
        df = df.sort_index().reset_index(drop=True)
        warn(f"SAMPLING ACTIVE: using {len(df):,} randomly sampled rows "
             f"(SFD_SAMPLE_ROWS). Results are subsample estimates.")

    return df


def required_columns_present(df: pd.DataFrame, required: list[str]) -> list[str]:
    """Return the list of missing columns (empty list == all present)."""
    return [c for c in required if c not in df.columns]


def assert_minimum_schema(df: pd.DataFrame) -> None:
    """
    Guard the handful of columns without which the project has no meaning.

    Everything else is treated as optional and detected at runtime, so the
    pipeline degrades gracefully on a differently-shaped Borg export instead of
    crashing with a KeyError.
    """
    minimum = [config.TARGET, "collection_id", "machine_id", "time"]
    missing = required_columns_present(df, minimum)
    if missing:
        raise PipelineError(
            "Dataset is missing column(s) essential to this study: "
            f"{missing}. Present columns: {list(df.columns)}"
        )
    if df[config.TARGET].nunique(dropna=True) < 2:
        raise PipelineError(
            f"Target '{config.TARGET}' has < 2 distinct values "
            f"({df[config.TARGET].unique().tolist()}); supervised learning is "
            "impossible. Explicit-failure experiments will be skipped."
        )


def load_processed(which: str = "feature") -> pd.DataFrame:
    """Read back a processed artefact produced by an earlier stage."""
    paths = {
        "cleaned": config.CLEANED_CSV,
        "feature": config.FEATURE_CSV,
        "model": config.MODEL_CSV,
    }
    if which not in paths:
        raise PipelineError(f"Unknown processed dataset '{which}'. Options: {list(paths)}")
    p = paths[which]
    if not p.exists():
        raise PipelineError(
            f"{p.name} not found. Run the pipeline first:  python run_project.py"
        )
    return pd.read_csv(p, low_memory=False)
