"""
config.py
=========
Single source of truth for paths, column groups, and tunable parameters.

Every design decision recorded here was derived from an *empirical inspection*
of `data/raw/borg_traces_data.csv` (405,894 rows x 34 columns), not assumed.
The inspection evidence for each decision is cited in the comments so the
choices can be defended in a viva.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. PATHS
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODEL_DIR = OUTPUT_DIR / "models"
PRED_DIR = OUTPUT_DIR / "predictions"
REPORT_DIR = OUTPUT_DIR / "reports"
TABLE_DIR = OUTPUT_DIR / "tables"

# The raw file keeps its original name so provenance is unambiguous.
RAW_FILENAME = "borg_traces_data.csv"
RAW_CSV = RAW_DIR / RAW_FILENAME

CLEANED_CSV = PROCESSED_DIR / "cleaned_dataset.csv"
FEATURE_CSV = PROCESSED_DIR / "feature_dataset.csv"
MODEL_CSV = PROCESSED_DIR / "model_dataset.csv"

ALL_DIRS = [
    RAW_DIR, PROCESSED_DIR, FIG_DIR, METRICS_DIR,
    MODEL_DIR, PRED_DIR, REPORT_DIR, TABLE_DIR,
]


def ensure_dirs() -> None:
    """Create every output directory. Idempotent."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 2. REPRODUCIBILITY
# ---------------------------------------------------------------------------

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 3. RAW SCHEMA  (verified against the actual file)
# ---------------------------------------------------------------------------

TARGET = "failed"

#: Pandas writes an unnamed index column; inspection confirmed it is exactly
#: ``np.arange(len(df))``, so it carries zero information.
INDEX_ARTIFACT_COLS = ["Unnamed: 0"]

#: ---------------------------------------------------------------------------
#: THE LEAKAGE BLACKLIST  -- the most important object in this project.
#: ---------------------------------------------------------------------------
#: Inspection proved, on all 405,894 rows with zero exceptions:
#:     failed == (event == "FAIL")
#:     failed == (instance_events_type == 2)
#:     failed == (collections_events_type == 2)
#: and additionally ``instance_events_type`` is byte-identical to
#: ``collections_events_type``.
#:
#: These columns are *re-encodings of the target itself*. Giving them to a
#: classifier yields ~100% accuracy that measures nothing. They are therefore
#: banned from every supervised feature matrix. They are still kept in the
#: cleaned dataset for descriptive analysis (event-type breakdowns), which is
#: legitimate because that analysis makes no predictive claim.
TARGET_LEAKAGE_COLS = [
    "event",
    "instance_events_type",
    "collections_events_type",
]

#: Hashed identifiers. ``collection_name`` maps to a single collection 89.2% of
#: the time and ``collection_logical_name`` 79.3% of the time, so encoding them
#: would smuggle collection identity (and hence the collection-constant target)
#: into the features. They are dropped from the model matrix.
IDENTITY_COLS = [
    "collection_name",
    "collection_logical_name",
    "user",
]

#: Structural keys: needed for grouping / joining / auditing, never used as
#: predictive features (they are arbitrary integers).
KEY_COLS = [
    "collection_id",
    "instance_index",
    "machine_id",
    "alloc_collection_id",
    "cluster",
]

#: Columns holding Python-literal dicts of the form {'cpus': x, 'memory': y}.
DICT_COLS = [
    "resource_request",
    "average_usage",
    "maximum_usage",
    "random_sample_usage",
]

#: Columns holding NumPy array reprs (may contain embedded newlines).
#: Verified lengths: cpu_usage_distribution -> 11 values (deciles),
#: tail_cpu_usage_distribution -> 9 values (tail percentiles).
ARRAY_COLS = ["cpu_usage_distribution", "tail_cpu_usage_distribution"]
CPU_DIST_LEN = 11
TAIL_DIST_LEN = 9

#: List-valued structural columns -> converted to counts.
LIST_COUNT_COLS = ["constraint", "start_after_collection_ids"]

#: ``random_sample_usage`` has a 'memory' key whose value is ``None`` in
#: 405,894 / 405,894 rows (100% null). It is dropped rather than imputed:
#: imputing a column with no observed values invents data.
FULLY_NULL_PARSED = ["random_sample_usage_memory"]

# ---------------------------------------------------------------------------
# 4. TIME SEMANTICS
# ---------------------------------------------------------------------------

#: Borg encodes "event happened before the trace window" as 0 and "after the
#: trace window" as 2**63-1. Inspection: 56,452 rows have time == 0 and 3 rows
#: have time == 2**63-1.
#:
#: These are NOT timestamps. Critically, fail-rate is 65.7% when time == 0
#: versus 15.9% for real timestamps -- so feeding raw ``time`` to a model
#: creates a strong pseudo-predictor that encodes trace-boundary bookkeeping
#: rather than system behaviour. We therefore:
#:   * replace sentinels with NaN in the numeric time column, and
#:   * expose the sentinel status as an explicit, honest categorical flag.
TIME_SENTINEL_MAX = 2 ** 63 - 1
TIME_SENTINEL_MIN = 0

#: Microseconds -> seconds / hours.
US_PER_SEC = 1_000_000
US_PER_HOUR = 3_600 * US_PER_SEC

#: The standard Borg usage-measurement window is 300 s.
NOMINAL_WINDOW_US = 300 * US_PER_SEC

# ---------------------------------------------------------------------------
# 5. BORG PRIORITY TIERS
# ---------------------------------------------------------------------------

#: Documented Google Borg priority bands. Used to build an ordinal, physically
#: meaningful feature instead of treating ``priority`` as a bare integer.
PRIORITY_TIERS = [
    (0, 99, 0, "free"),
    (100, 115, 1, "best_effort_batch"),
    (116, 119, 2, "mid"),
    (120, 359, 3, "production"),
    (360, 10_000, 4, "monitoring"),
]

# ---------------------------------------------------------------------------
# 6. TEMPORAL-ANALYSIS GRAIN  (chosen from measured support, not by habit)
# ---------------------------------------------------------------------------

#: Measured event counts per entity:
#:   machine_id : 96,174 entities, median 4 events,   max 20  -> rolling NOT viable
#:   collection : 4,057 entities,  median 6 events,   max 9,828 -> viable
#:   cluster-hr : 5,946 cells,     median 31 events            -> strongly viable
#:
#: Consequence: per-machine rolling-window degradation analysis is not
#: statistically supportable on this file. We therefore run rolling/trend
#: analysis at the collection and cluster-hour grain, and profile machines
#: *cross-sectionally* against their cluster peers.
MIN_EVENTS_FOR_MACHINE_STATS = 2      # machine appears at least twice
MIN_EVENTS_FOR_MACHINE_SLOPE = 3      # a straight line needs >= 3 points to be meaningful
MIN_EVENTS_FOR_COLLECTION_TREND = 10  # 1,320 collections qualify
ROLLING_WINDOW = 5                    # events, for collection-level rolling stats
TIME_BIN_HOURS = 1                    # cluster-hour panel

# ---------------------------------------------------------------------------
# 7. SILENT-FAILURE RISK SCORE
# ---------------------------------------------------------------------------

#: The six degradation indicators. Each is a *symptom of stress that is not
#: itself an explicit failure*, which is precisely what "silent" means here.
#: None of them is derived from ``failed`` -- that independence is what makes
#: validating the score against ``failed`` a genuine test rather than a
#: tautology.
RISK_COMPONENTS = {
    "cpu_stall_pressure": "Cycles-per-instruction elevated vs cluster peers "
                          "(memory stalls / cache contention -> work takes more cycles).",
    "memory_pressure": "Live memory occupying a high share of assigned memory, "
                       "with page-cache displacement (approaching OOM territory).",
    "resource_variability": "Coefficient of variation / burstiness of CPU demand "
                            "(unstable, thrashing behaviour).",
    "tail_latency_pressure": "Gap between tail CPU percentiles and mean usage "
                             "(long stalls hidden behind a healthy average).",
    "telemetry_incompleteness": "sample_rate below 1.0 -- the monitoring system "
                                "itself failed to collect samples, a classic "
                                "silent-failure signature.",
    "provisioning_mismatch": "Divergence between requested and actually used "
                             "resources (stuck, starved, or misprovisioned work).",
}

#: Weighting policy. The brief forbids inventing weights, so we do not.
#: We compute the composite three ways and *report their agreement*:
#:   equal  -- principle of indifference: no evidence justifies unequal weights
#:   pca    -- data-driven weights taken from the first principal component
#:   mahal  -- multivariate outlyingness, weight-free by construction
#: If the three rankings agree (Spearman rho), the conclusion is robust to the
#: weighting choice, which is a stronger statement than any single weight set.
RISK_WEIGHT_SCHEMES = ("equal", "pca", "mahalanobis")
PRIMARY_RISK_SCHEME = "equal"

#: Risk bands are *distributional quantiles of this trace*, not absolute
#: clinical thresholds. Naming them this way keeps the claim honest.
RISK_BANDS = [
    (0.00, 0.50, "NORMAL"),
    (0.50, 0.80, "LOW_RISK"),
    (0.80, 0.95, "MEDIUM_RISK"),
    (0.95, 1.01, "HIGH_RISK"),
]

# ---------------------------------------------------------------------------
# 8. ANOMALY DETECTION
# ---------------------------------------------------------------------------

ANOMALY_CONTAMINATION = 0.05   # 5% expected anomaly budget, held constant across detectors
IFOREST_ESTIMATORS = 200
LOF_NEIGHBORS = 20

#: LOF is O(n log n) with a tree but memory-heavy on wide data; One-Class SVM is
#: O(n^2)-ish. Both are therefore fitted on a stratified subsample while
#: Isolation Forest runs on the full dataset. Sampling here is a *compute*
#: concession, declared openly, not a statistical shortcut.
LOF_SAMPLE_SIZE = 60_000
OCSVM_SAMPLE_SIZE = 20_000

# ---------------------------------------------------------------------------
# 9. SUPERVISED LEARNING
# ---------------------------------------------------------------------------

TEST_SIZE = 0.20
CV_FOLDS = 5

#: Three evaluation protocols are run on identical features and models so the
#: *only* thing that varies is the split. This is the project's core experiment.
SPLIT_PROTOCOLS = ("random", "grouped", "grouped_temporal")
HONEST_PROTOCOL = "grouped"   # the one whose numbers we report as the real result

#: SVM, KNN and Gradient Boosting do not scale to 400k rows in a
#: student-runnable time budget. They are trained on a stratified subsample;
#: this is recorded in the metrics table so the comparison is never silently
#: unfair.
#:
#: The size was chosen by measurement, not by taste. On a 32,000-row training
#: fold with 113 features, one RBF SVM fit took ~25 minutes and one Gradient
#: Boosting fit took 90s; three protocols made stage 7 a 54-minute stage. At
#: 12,000 rows the same three protocols finish in a few minutes. The kernel SVM
#: is O(n^2)-O(n^3) in samples, so this is a quadratic saving, and Gradient
#: Boosting is strictly sequential in its 200 boosting rounds (it cannot use
#: n_jobs). Logistic Regression and Random Forest still see every training row.
HEAVY_MODEL_SAMPLE = 12_000
HEAVY_MODELS = ("SVM (RBF)", "KNN", "Gradient Boosting")

SHAP_SAMPLE_SIZE = 3_000
PERMUTATION_REPEATS = 5

# ---------------------------------------------------------------------------
# 10. RUNTIME / PERFORMANCE
# ---------------------------------------------------------------------------

#: Set SFD_SAMPLE_ROWS=200000 to run the whole pipeline on a subsample.
#: Default None = use every row.
SAMPLE_ROWS = os.environ.get("SFD_SAMPLE_ROWS")
SAMPLE_ROWS = int(SAMPLE_ROWS) if SAMPLE_ROWS else None

N_JOBS = -1
FIG_DPI = 140
FIG_FORMAT = "png"
