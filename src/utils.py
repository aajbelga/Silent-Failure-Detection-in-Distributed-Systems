"""
utils.py
========
Small shared helpers: logging, timing, robust statistics, safe IO.

Kept deliberately plain -- every function here should be explainable in one
sentence during a viva.
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import config

# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

_WIDTH = 78


def banner(text: str, char: str = "=") -> None:
    print(char * _WIDTH)
    print(f" {text}")
    print(char * _WIDTH)


def section(text: str) -> None:
    print(f"\n--- {text} " + "-" * max(0, _WIDTH - len(text) - 6))


def step(i: int, n: int, text: str) -> None:
    print(f"[{i}/{n}] {text}", flush=True)


def info(text: str) -> None:
    print(f"    {text}", flush=True)


def warn(text: str) -> None:
    print(f"    [!] {text}", flush=True)


@contextmanager
def timer(label: str):
    """Print how long a block took. Used to show the pipeline is efficient."""
    t0 = time.time()
    yield
    print(f"    ({label} took {time.time() - t0:.1f}s)", flush=True)


# ---------------------------------------------------------------------------
# Robust statistics
# ---------------------------------------------------------------------------

def robust_z(s: pd.Series) -> pd.Series:
    """
    Median / MAD based z-score.

    Chosen over the standard z-score because every resource column in this
    trace is heavily right-skewed (e.g. maximum_usage_cpus has max 1.27 versus
    a median of 0.005). A mean/std z-score would be dominated by a handful of
    extreme rows, so the "normal" band would be meaningless.

    1.4826 rescales MAD to be a consistent estimator of sigma for Gaussian data.
    """
    s = pd.to_numeric(s, errors="coerce")
    med = s.median()
    mad = (s - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        # Degenerate spread: fall back to std, then to all-zeros.
        std = s.std()
        if not np.isfinite(std) or std == 0:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - med) / std
    return (s - med) / (1.4826 * mad)


def rank_normalise(s: pd.Series) -> pd.Series:
    """
    Map a series to [0, 1] by average rank.

    Used for the risk-score components. Rank-normalisation is scale-free and
    outlier-proof, so components measured in wildly different units (CPI in
    'cycles', sample_rate as a fraction) become directly comparable without
    inventing a conversion factor.
    """
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    r = s.rank(method="average", na_option="keep")
    denom = r.max() - r.min()
    if not np.isfinite(denom) or denom == 0:
        out = pd.Series(np.zeros(len(s)), index=s.index)
    else:
        out = (r - r.min()) / denom
    # A missing component contributes the neutral value (median rank 0.5)
    # rather than silently pulling the composite toward 0.
    return out.fillna(0.5)


def safe_div(a, b, fill: float = np.nan):
    """
    Division that yields `fill` instead of inf/NaN on a zero denominator.

    Always returns a positionally-indexed float64 ndarray for array-like input
    (callers assign it straight into a DataFrame column, so returning a Series
    here would introduce pandas index alignment where none is wanted), and a
    plain float when both arguments are scalars.
    """
    scalar = np.isscalar(a) and np.isscalar(b)
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(a, b)
    # np.divide on two scalars returns a 0-d array, which does not support item
    # assignment -- atleast_1d makes the masking below work for both cases.
    out = np.atleast_1d(out).astype("float64", copy=True)
    out[~np.isfinite(out)] = fill
    return float(out[0]) if scalar else out


def slope(y: Iterable[float], x: Iterable[float] | None = None) -> float:
    """
    Least-squares slope of y against x (or against position if x is None).

    Returns NaN when there are fewer than 3 finite points -- fitting a trend to
    two points always succeeds numerically but says nothing, and this project
    should not report meaningless trends.
    """
    y = np.asarray(list(y), dtype="float64")
    if x is None:
        x = np.arange(len(y), dtype="float64")
    else:
        x = np.asarray(list(x), dtype="float64")
    m = np.isfinite(y) & np.isfinite(x)
    if m.sum() < 3 or np.ptp(x[m]) == 0:
        return np.nan
    return float(np.polyfit(x[m], y[m], 1)[0])


def coefficient_of_variation(s: pd.Series) -> float:
    """std / |mean|; NaN when the mean is ~0 so the ratio is undefined."""
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 2:
        return np.nan
    m = s.mean()
    if abs(m) < 1e-12:
        return np.nan
    return float(s.std() / abs(m))


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def downcast(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Shrink numeric dtypes in place (float64 -> float32, int64 -> smallest int).

    Cuts the feature matrix roughly in half. Applied only to engineered
    features, never to the raw 64-bit ID and timestamp columns, where a
    narrower dtype would corrupt values (machine_id reaches 1.7e11 and
    time reaches 2.7e12).
    """
    before = df.memory_usage(deep=True).sum() / 1024 ** 2
    for c in df.columns:
        if df[c].dtype == "float64":
            df[c] = pd.to_numeric(df[c], downcast="float")
        elif df[c].dtype == "int64" and df[c].abs().max() < 2 ** 31:
            df[c] = pd.to_numeric(df[c], downcast="integer")
    after = df.memory_usage(deep=True).sum() / 1024 ** 2
    if verbose:
        info(f"memory {before:.0f} MB -> {after:.0f} MB")
    return df


def mem_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum() / 1024 ** 2)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def save_table(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    info(f"saved {path.relative_to(config.ROOT)}  ({len(df):,} rows)")
    return path


def save_text(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    info(f"saved {path.relative_to(config.ROOT)}")
    return path


def save_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return str(o)

    path.write_text(json.dumps(obj, indent=2, default=_default), encoding="utf-8")
    return path


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fmt(x, nd: int = 4) -> str:
    """Format a number for a report; never crash on None/NaN."""
    try:
        if x is None:
            return "n/a"
        xf = float(x)
        if not np.isfinite(xf):
            return "n/a"
        return f"{xf:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def fmt_int(x) -> str:
    try:
        return f"{int(x):,}"
    except (TypeError, ValueError):
        return str(x)


class PipelineError(RuntimeError):
    """Raised for expected, explainable failures (missing file, empty data)."""


def die(msg: str, code: int = 1):
    """Fail loudly but gracefully -- no stack trace for user-input problems."""
    print(f"\n[FATAL] {msg}\n", file=sys.stderr)
    raise PipelineError(msg)
