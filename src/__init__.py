"""
Silent Failure Detection in Distributed Systems -- Google Borg Cluster Traces
============================================================================

Package layout (each module maps to stages of the study):

    config              paths, constants, every tunable in one place
    utils               small shared helpers (robust stats, IO, logging)
    data_loader         raw CSV loading, row counting, schema guards
    data_validation     Stage 1  -- profiling + LEAKAGE AUDIT
    preprocessing       Stage 2  -- cleaning, parsing, per-column NA policy
    feature_engineering Stage 3  -- feature construction + safety registry
    target_engineering  Stage 4  -- silent-failure degradation index (CORE)
    anomaly_detection   Stage 5  -- IsolationForest / LOF / One-Class SVM
    models              Stages 6-7 -- classifiers + split protocols
    evaluation          metrics, confusion matrices, honest interpretation
    explainability      Stage 8  -- importance, permutation, SHAP
    temporal_analysis   Stages 9-10 -- trends, degradation candidates
    visualization       Stage 12 -- all figures
"""

__version__ = "1.0.0"
__all__ = [
    "config", "utils", "data_loader", "data_validation", "preprocessing",
    "feature_engineering", "target_engineering", "anomaly_detection",
    "temporal_analysis", "models", "evaluation", "explainability",
    "visualization",
]
