"""
tests/test_pipeline.py
======================
STAGE 17 -- the test suite.

Covers the six required areas (loading, preprocessing, feature generation,
target construction, model training, evaluation) plus the five required
graceful-failure cases.

    pytest -q

Two classes of test here, and the distinction matters at a viva:

  * **mechanical** -- does the function run and return the right shape?
  * **scientific** -- does it protect a claim the report makes? The leakage
    tests are the important ones: if `test_target_leakage_columns_excluded`
    ever fails, every accuracy number in the report becomes meaningless.
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
from src.utils import PipelineError


# ===========================================================================
# 1. LOADING
# ===========================================================================

class TestLoading:

    def test_required_columns_present_detects_gaps(self):
        from src.data_loader import required_columns_present
        df = pd.DataFrame({"a": [1], "b": [2]})
        assert required_columns_present(df, ["a", "b"]) == []
        assert required_columns_present(df, ["a", "zzz"]) == ["zzz"]

    def test_minimum_schema_accepts_valid_frame(self, raw_like):
        from src.data_loader import assert_minimum_schema
        assert_minimum_schema(raw_like)          # must not raise

    def test_fixture_matches_real_schema_width(self, raw_like):
        # The real file has exactly 34 columns; the fixture must not silently
        # drift away from it or these tests stop testing the real pipeline.
        from conftest import REAL_COLUMNS
        assert list(raw_like.columns) == REAL_COLUMNS
        assert raw_like.shape[1] == 34, raw_like.shape

    def test_load_processed_rejects_unknown_name(self):
        from src.data_loader import load_processed
        with pytest.raises((PipelineError, ValueError, KeyError)):
            load_processed("not_a_real_dataset")


# ===========================================================================
# 2. PREPROCESSING
# ===========================================================================

class TestPreprocessing:

    def test_dict_column_parsed_to_numeric(self, raw_like):
        from src.preprocessing import parse_dict_column
        out = parse_dict_column(raw_like["average_usage"], "cpus")
        assert pd.api.types.is_numeric_dtype(out)
        assert out.notna().all()
        assert (out >= 0).all()

    def test_dict_column_survives_garbage(self):
        from src.preprocessing import parse_dict_column
        s = pd.Series(["{'cpus': 0.5}", "not a dict", None, "{}", "[]"])
        out = parse_dict_column(s, "cpus")
        assert out.iloc[0] == pytest.approx(0.5)
        assert out.iloc[1:].isna().all()          # bad input -> NaN, not a crash

    def test_array_column_expands_to_fixed_width(self, raw_like):
        from src.preprocessing import parse_array_column
        arr = parse_array_column(raw_like["cpu_usage_distribution"],
                                 config.CPU_DIST_LEN)
        assert arr.shape == (len(raw_like), config.CPU_DIST_LEN)
        assert np.isfinite(arr).all()

    def test_array_column_rejects_wrong_length_rows(self):
        """A distribution vector of the wrong length is malformed, so the parser
        returns all-NaN for that row rather than padding it. Padding would
        fabricate percentile values that were never measured."""
        from src.preprocessing import parse_array_column
        arr = parse_array_column(
            pd.Series(["[0.1 0.2 0.3 0.4 0.5]", "[0.1 0.2]", "[]", None]), 5)
        assert arr.shape == (4, 5)
        assert arr[0, 0] == pytest.approx(0.1)      # correct length -> parsed
        assert np.isnan(arr[1]).all()               # too short -> all NaN
        assert np.isnan(arr[2]).all()               # empty -> all NaN
        assert np.isnan(arr[3]).all()               # None -> all NaN

    def test_clean_returns_frame_and_summary(self, cleaned):
        df, s = cleaned
        assert isinstance(df, pd.DataFrame) and len(df) > 0
        for k in ["original_rows", "rows_removed", "duplicate_rows_found",
                  "final_rows", "original_columns", "final_columns",
                  "missing_values_before", "missing_values_after"]:
            assert k in s, f"summary missing required key {k}"

    def test_clean_does_not_dropna_the_dataset(self, raw_like, cleaned):
        """The spec forbids a global dropna. cycles_per_instruction is ~40%
        null in the fixture; a global dropna would destroy most rows."""
        df, s = cleaned
        assert len(df) >= 0.75 * len(raw_like), (
            f"lost {len(raw_like) - len(df)} of {len(raw_like)} rows -- "
            "this looks like a global dropna")

    def test_clean_records_a_decision_for_every_step(self, cleaned):
        _, s = cleaned
        assert len(s.get("decisions", [])) >= 8

    def test_time_sentinels_are_not_treated_as_real_times(self, cleaned):
        df, _ = cleaned
        assert "time_is_valid" in df.columns
        assert df["time_is_valid"].sum() < len(df)     # the fixture plants 20
        if "time_clean" in df.columns:
            valid = df.loc[df["time_is_valid"], "time_clean"]
            assert (valid < config.TIME_SENTINEL_MAX).all()

    def test_missingness_indicator_created(self, cleaned):
        df, _ = cleaned
        flags = [c for c in df.columns if c.endswith("_missing")]
        assert flags, "informative missingness must be preserved as a flag"

    def test_clean_is_idempotent_on_row_count(self, cleaned):
        from src import preprocessing
        df, _ = cleaned
        again, s2 = preprocessing.clean(df.copy(), profile=None)
        assert s2["duplicate_rows_found"] == 0, \
            "cleaning twice should find nothing left to remove"


# ===========================================================================
# 3. FEATURE GENERATION
# ===========================================================================

class TestFeatureGeneration:

    def test_build_features_adds_columns_and_registry(self, cleaned, featured):
        before, _ = cleaned
        after, registry = featured
        assert after.shape[1] > before.shape[1]
        assert len(after) == len(before), "features must not change the row count"
        assert not registry.empty
        assert set(registry.columns) >= {"feature", "group", "model_safe",
                                         "description"}

    def test_every_registered_feature_has_a_rationale(self, featured):
        _, registry = featured
        assert registry["description"].str.len().min() > 10, \
            "a feature with no stated rationale cannot be defended at a viva"

    def test_no_infinities_leak_into_features(self, featured):
        df, registry = featured
        safe = [c for c in registry.loc[registry.model_safe, "feature"]
                if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        bad = [c for c in safe if np.isinf(df[c].to_numpy(dtype="float64",
                                                          na_value=np.nan)).any()]
        assert not bad, f"infinite values in {bad} (division without safe_div?)"

    def test_model_feature_columns_are_all_numeric(self, featured):
        from src import feature_engineering as fe
        df, _ = featured
        cols = fe.model_feature_columns(df)
        assert cols
        non_num = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
        assert not non_num, f"non-numeric features in the model matrix: {non_num}"

    # ---- the leakage guards --------------------------------------------

    def test_target_leakage_columns_excluded(self, featured):
        """If this fails, every reported metric is invalid."""
        from src import feature_engineering as fe
        df, _ = featured
        cols = set(fe.model_feature_columns(df))
        for bad in config.TARGET_LEAKAGE_COLS + [config.TARGET]:
            assert bad not in cols, f"LEAKAGE: {bad} reached the model matrix"

    def test_identity_and_key_columns_excluded(self, featured):
        from src import feature_engineering as fe
        df, _ = featured
        cols = set(fe.model_feature_columns(df))
        for bad in config.IDENTITY_COLS + config.KEY_COLS:
            assert bad not in cols, f"{bad} would let the model memorise identity"

    def test_target_derived_features_are_not_model_safe(self, featured):
        _, registry = featured
        for name in ["machine_failure_rate", "machine_failure_count"]:
            row = registry[registry["feature"] == name]
            if not row.empty:
                assert not bool(row["model_safe"].iloc[0]), \
                    f"{name} is derived from the target and cannot be model-safe"

    def test_fixture_reproduces_the_documented_target_equivalence(self, raw_like):
        """The report claims failed <=> event == 'FAIL'. Assert the property the
        real audit found, so the claim is testable rather than anecdotal."""
        assert (raw_like["failed"] == (raw_like["event"] == "FAIL")).all()
        assert (raw_like["failed"] ==
                (raw_like["instance_events_type"] == 2)).all()


# ===========================================================================
# 4. TARGET / RISK CONSTRUCTION
# ===========================================================================

class TestTargetConstruction:

    def test_components_are_bounded_and_named(self, featured):
        from src import target_engineering as te
        df, _ = featured
        comp, prov = te.build_risk_components(df)
        assert not comp.empty
        # second return is provenance keyed by component name
        assert set(prov.keys()) == set(comp.columns)
        for c in comp.columns:
            v = comp[c].dropna()
            assert v.min() >= -1e-9 and v.max() <= 1 + 1e-9, \
                f"{c} is not rank-normalised to [0,1]"
            assert c in config.RISK_COMPONENTS
            assert prov[c]["sources"], f"{c} has no recorded source columns"

    def test_combination_produces_all_three_schemes(self, featured):
        from src import target_engineering as te
        df, _ = featured
        comp, _ = te.build_risk_components(df)
        scores, meta = te.combine_components(comp)
        for s in config.RISK_WEIGHT_SCHEMES:
            assert f"risk_{s}" in scores.columns
        assert "spearman_agreement" in meta

    def test_risk_score_never_uses_the_target(self, featured):
        """The whole validity argument rests on this independence."""
        from src import target_engineering as te
        df, _ = featured
        shuffled = df.copy()
        rng = np.random.default_rng(0)
        shuffled[config.TARGET] = rng.permutation(shuffled[config.TARGET].values)
        a, _ = te.build_risk_components(df)
        b, _ = te.build_risk_components(shuffled)
        common = [c for c in a.columns if c in b.columns]
        for c in common:
            pd.testing.assert_series_equal(a[c], b[c], check_names=False)

    def test_bands_are_ordered_and_cover_every_row(self, featured):
        from src import target_engineering as te
        df, _ = featured
        comp, _ = te.build_risk_components(df)
        scores, _ = te.combine_components(comp)
        bands = te.assign_risk_bands(scores["risk_equal"])
        assert bands.notna().all()
        assert set(bands.unique()) <= {b[2] for b in config.RISK_BANDS}

    def test_risk_bands_increase_with_score(self, featured):
        from src import target_engineering as te
        df, _ = featured
        comp, _ = te.build_risk_components(df)
        s = te.combine_components(comp)[0]["risk_equal"]
        means = pd.DataFrame({"s": s, "b": te.assign_risk_bands(s)}) \
            .groupby("b", observed=True)["s"].mean().dropna()
        assert list(means.values) == sorted(means.values), \
            "band means must increase monotonically with the score"

    def test_machine_profile_reports_support(self, featured):
        from src import target_engineering as te
        df, _ = featured
        comp, _ = te.build_risk_components(df)
        scores, _ = te.combine_components(comp)
        df = df.assign(silent_failure_risk_score=scores["risk_equal"])
        prof = te.machine_risk_profile(df, comp)
        assert "support" in prof.columns and "confidence" in prof.columns
        assert prof["support"].min() >= 1


# ===========================================================================
# 5. MODEL TRAINING
# ===========================================================================

class TestModelTraining:

    def test_grouped_split_has_no_shared_groups(self):
        from src import models
        rng = np.random.default_rng(0)
        X = rng.random((600, 4))
        y = (rng.random(600) < 0.3).astype(int)
        g = rng.integers(0, 30, 600)
        tr, te = models.split_grouped(X, y, g)
        assert len(set(g[tr]) & set(g[te])) == 0, "grouped split leaked a group"
        assert len(tr) + len(te) == 600

    def test_random_split_does_share_groups(self):
        """Documents the trap rather than only asserting the fix works."""
        from src import models
        rng = np.random.default_rng(0)
        X = rng.random((600, 4))
        y = (rng.random(600) < 0.3).astype(int)
        g = rng.integers(0, 10, 600)
        tr, te = models.split_random(X, y, g)
        assert len(set(g[tr]) & set(g[te])) > 0

    def test_grouped_temporal_split_puts_the_past_in_train(self):
        from src import models
        rng = np.random.default_rng(1)
        n = 600
        X = rng.random((n, 3))
        y = (rng.random(n) < 0.3).astype(int)
        g = np.repeat(np.arange(30), 20)
        times = np.repeat(np.arange(30) * 1000.0, 20)
        tr, te = models.split_grouped_temporal(X, y, g, times)
        assert len(set(g[tr]) & set(g[te])) == 0
        assert times[tr].max() <= times[te].min()

    def test_describe_split_flags_leakage(self):
        from src import models
        rng = np.random.default_rng(0)
        y = (rng.random(200) < 0.3).astype(int)
        g = rng.integers(0, 5, 200)
        d = models.describe_split("random", np.arange(150), np.arange(150, 200),
                                  y, g)
        assert d["group_leakage"] is True
        assert d["n_train"] == 150 and d["n_test"] == 50

    def test_build_models_returns_the_planned_family(self):
        from src import models
        m = models.build_models(5_000, 0.2)
        for name in ["Logistic Regression", "Decision Tree", "Random Forest",
                     "Gradient Boosting", "KNN", "SVM (RBF)"]:
            assert name in m
        assert all(hasattr(v, "fit") for v in m.values())

    def test_fit_predict_trains_and_scores(self):
        from src import models
        rng = np.random.default_rng(0)
        X = rng.random((400, 5))
        y = (X[:, 0] + rng.random(400) * 0.3 > 0.8).astype(int)
        tr, te = np.arange(300), np.arange(300, 400)
        clf = models.build_models(300, y.mean())["Decision Tree"]
        pred, proba, secs, sub = models.fit_predict(clf, X, y, tr, te,
                                                    "Decision Tree")
        assert pred.shape == (100,)
        assert proba is None or (0 <= np.nanmin(proba) and np.nanmax(proba) <= 1)
        assert secs >= 0

    def test_heavy_models_are_subsampled_and_disclosed(self):
        from src import models
        tr = np.arange(config.HEAVY_MODEL_SAMPLE * 3)
        y = np.zeros(len(tr), dtype=int)
        y[::4] = 1
        kept = models._subsample_train("SVM (RBF)", tr, y)
        assert len(kept) == config.HEAVY_MODEL_SAMPLE
        light = models._subsample_train("Logistic Regression", tr, y)
        assert len(light) == len(tr), "cheap models must see every training row"


# ===========================================================================
# 6. EVALUATION
# ===========================================================================

class TestEvaluation:

    def test_perfect_prediction_scores_one(self):
        from src.evaluation import evaluate
        y = np.array([0, 0, 1, 1, 0, 1])
        m = evaluate(y, y, y.astype(float))
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)
        assert m["roc_auc"] == pytest.approx(1.0)
        assert m["fp"] == 0 and m["fn"] == 0

    def test_all_negative_prediction_is_not_rewarded(self):
        """The reason PR-AUC, not accuracy, ranks the models."""
        from src.evaluation import evaluate
        y = np.array([0] * 95 + [1] * 5)
        m = evaluate(y, np.zeros(100, dtype=int), np.zeros(100))
        assert m["accuracy"] == pytest.approx(0.95)   # looks great
        assert m["recall"] == 0.0                     # is useless
        assert m["f1"] == 0.0
        assert m["missed_failures"] == 5

    def test_confusion_counts_sum_to_n(self):
        from src.evaluation import evaluate
        rng = np.random.default_rng(0)
        y = (rng.random(300) < 0.3).astype(int)
        p = (rng.random(300) < 0.3).astype(int)
        m = evaluate(y, p)
        assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == 300

    def test_metrics_degrade_gracefully_without_probabilities(self):
        from src.evaluation import evaluate
        y = np.array([0, 1, 0, 1])
        m = evaluate(y, np.array([0, 1, 0, 0]), None)
        assert np.isnan(m["roc_auc"]) and np.isnan(m["pr_auc"])
        assert not np.isnan(m["accuracy"])

    def test_single_class_input_does_not_crash(self):
        from src.evaluation import evaluate
        y = np.zeros(50, dtype=int)
        m = evaluate(y, y, np.zeros(50))
        assert np.isnan(m["roc_auc"]), "AUC is undefined with one class"
        assert m["accuracy"] == pytest.approx(1.0)


# ===========================================================================
# 7. GRACEFUL FAILURE -- the five required cases
# ===========================================================================

class TestGracefulFailure:

    def test_missing_dataset_gives_a_clear_message(self, tmp_path):
        from src.data_loader import check_raw_exists
        with pytest.raises(PipelineError) as e:
            check_raw_exists(tmp_path / "nope.csv")
        msg = str(e.value).lower()
        assert "nope.csv" in msg or "not found" in msg or "missing" in msg
        assert "traceback" not in msg

    def test_missing_column_is_named_not_a_keyerror(self):
        from src.data_loader import assert_minimum_schema
        df = pd.DataFrame({"failed": [0, 1], "collection_id": [1, 2]})
        with pytest.raises(PipelineError) as e:
            assert_minimum_schema(df)
        assert "machine_id" in str(e.value) or "time" in str(e.value)

    def test_empty_dataset_is_rejected_with_a_reason(self):
        from src.data_loader import assert_minimum_schema
        empty = pd.DataFrame(columns=["failed", "collection_id", "machine_id",
                                      "time"])
        with pytest.raises(PipelineError):
            assert_minimum_schema(empty)

    def test_numeric_conversion_failure_yields_nan_not_a_crash(self):
        from src.preprocessing import parse_array_column, parse_dict_column
        junk = pd.Series(["completely unparseable", "{{{", "", None])
        assert parse_dict_column(junk, "cpus").isna().all()
        arr = parse_array_column(junk, 4)
        assert arr.shape == (4, 4) and np.isnan(arr).all()

    def test_insufficient_temporal_data_reports_rather_than_guesses(self):
        from src import temporal_analysis as ta
        df = pd.DataFrame({
            "collection_id": [1, 1, 2],
            "machine_id": [1, 2, 3],
            "cluster": ["a", "a", "b"],
            "time_clean": [1.0, 2.0, 3.0],
            "time_is_valid": [True, True, True],
            "silent_failure_risk_score": [0.1, 0.2, 0.3],
            config.TARGET: [0, 0, 1]})
        trends = ta.collection_trends(df)
        assert trends.empty or len(trends) == 0, \
            "3 rows cannot support a trend; the module must decline, not guess"

    def test_slope_declines_with_too_few_points(self):
        from src.utils import slope
        assert np.isnan(slope([1.0]))
        assert np.isnan(slope([1.0, 2.0]))          # needs >= 3 points
        assert slope([1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_zero_division_returns_fill_not_inf(self):
        from src.utils import safe_div
        out = safe_div(pd.Series([1.0, 2.0]), pd.Series([0.0, 2.0]))
        assert np.isnan(out[0]) and out[1] == pytest.approx(1.0)

    def test_safe_div_handles_scalars(self):
        """Regression test: a 0-d numpy result does not support item assignment,
        which crashed the temporal stage on a real run."""
        from src.utils import safe_div
        assert safe_div(4.0, 2.0) == pytest.approx(2.0)
        assert np.isnan(safe_div(1.0, 0.0))
        assert np.isnan(safe_div(np.float64(1.0), np.float64(0.0)))

    def test_rank_normalise_handles_a_constant_column(self):
        from src.utils import rank_normalise
        out = rank_normalise(pd.Series([5.0] * 10))
        assert out.notna().all() and out.between(0, 1).all()

    def test_rank_normalise_handles_all_nan(self):
        from src.utils import rank_normalise
        out = rank_normalise(pd.Series([np.nan] * 5))
        assert len(out) == 5      # must not raise


# ===========================================================================
# 8. END-TO-END SMOKE
# ===========================================================================

def test_short_end_to_end_on_synthetic_data(raw_like):
    """Loading -> cleaning -> features -> risk -> anomaly -> model -> evaluate
    on 400 synthetic rows. Proves the modules compose, without the 313 MB file."""
    from src import anomaly_detection, evaluation, models
    from src import feature_engineering as fe
    from src import preprocessing
    from src import target_engineering as te

    df, summary = preprocessing.clean(raw_like.copy(), profile=None)
    assert summary["final_rows"] > 0

    df, registry = fe.build_features(df)
    cols = fe.model_feature_columns(df)
    assert len(cols) > 10

    comp, meta = te.build_risk_components(df)
    scores, _ = te.combine_components(comp)
    df["silent_failure_risk_score"] = scores["risk_equal"].values
    assert df["silent_failure_risk_score"].between(0, 1).all()

    X = np.nan_to_num(df[cols].to_numpy("float32"), nan=0.0, posinf=0.0,
                      neginf=0.0)
    y = df[config.TARGET].to_numpy("int8")
    g = df["collection_id"].to_numpy()
    tr, teidx = models.split_grouped(X, y, g)
    assert len(set(g[tr]) & set(g[teidx])) == 0

    clf = models.build_models(len(tr), y.mean())["Decision Tree"]
    pred, proba, _, _ = models.fit_predict(clf, X, y, tr, teidx, "Decision Tree")
    m = evaluation.evaluate(y[teidx], pred, proba)
    assert 0.0 <= m["accuracy"] <= 1.0
    assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == len(teidx)
