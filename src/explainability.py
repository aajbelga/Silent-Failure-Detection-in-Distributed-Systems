"""
explainability.py
=================
STAGE 8 -- why does the model say what it says?

Three complementary views, in increasing cost and fidelity:

  1. Built-in importance   -- tree impurity gain / linear |coefficient|.
                              Free, but impurity importance is biased toward
                              high-cardinality features, so it is never the
                              only evidence presented.
  2. Permutation importance -- measured on the TEST fold, so it reports what
                              actually generalises rather than what the model
                              fitted. Model-agnostic.
  3. SHAP                  -- per-prediction attributions, computed on a
                              subsample because TreeExplainer over 400k rows is
                              needlessly expensive.

If SHAP is not installed or fails, that is reported as a limitation.
**No SHAP numbers are ever synthesised.**
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from . import config
from .utils import info, save_table, save_text, section, timer, warn


# ---------------------------------------------------------------------------
# 1. built-in importance
# ---------------------------------------------------------------------------

def native_importance(model, feature_cols: list[str]) -> pd.DataFrame:
    """Impurity gain for trees, |coefficient| for linear models."""
    est = model
    if hasattr(model, "named_steps"):
        est = model.named_steps.get("clf", model)

    if hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_, dtype="float64")
        kind = "impurity_gain"
    elif hasattr(est, "coef_"):
        imp = np.abs(np.asarray(est.coef_, dtype="float64")).ravel()
        kind = "abs_coefficient"
    else:
        return pd.DataFrame()

    if len(imp) != len(feature_cols):
        warn(f"importance length {len(imp)} != {len(feature_cols)} features; skipped")
        return pd.DataFrame()

    out = pd.DataFrame({"feature": feature_cols, "importance": imp, "kind": kind})
    out["importance_normalised"] = out.importance / max(out.importance.sum(), 1e-12)
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. permutation importance (on the held-out fold)
# ---------------------------------------------------------------------------

def permutation_importance_test(model, X, y, te, feature_cols: list[str],
                                sample: int = 20_000) -> pd.DataFrame:
    """
    Permutation importance on the TEST fold, scored by average precision.

    Uses a subsample of the test fold when it is large -- permutation cost is
    n_features x n_repeats refits of the scoring pass, and the ranking is stable
    well before the full fold is used.
    """
    from sklearn.inspection import permutation_importance

    idx = te
    if len(idx) > sample:
        rng = np.random.RandomState(config.RANDOM_STATE)
        # keep all positives; they are scarce and drive average precision
        pos = idx[y[idx] == 1]
        neg = idx[y[idx] == 0]
        take_neg = min(len(neg), max(sample - len(pos), 1))
        idx = np.sort(np.concatenate([pos, rng.choice(neg, take_neg, replace=False)]))
        warn(f"permutation importance on {len(idx):,} test rows "
             f"(all {len(pos):,} positives retained)")

    try:
        with timer("permutation_importance"):
            r = permutation_importance(
                model, X[idx], y[idx], n_repeats=config.PERMUTATION_REPEATS,
                random_state=config.RANDOM_STATE, scoring="average_precision",
                n_jobs=1)
    except Exception as exc:
        warn(f"permutation importance failed: {exc}")
        return pd.DataFrame()

    out = pd.DataFrame({"feature": feature_cols,
                        "perm_importance_mean": r.importances_mean,
                        "perm_importance_std": r.importances_std})
    # A feature whose mean drop is within one sd of zero has not demonstrated
    # any effect; flagging this prevents over-reading a long tail of noise.
    out["significant"] = out.perm_importance_mean > out.perm_importance_std
    return out.sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. SHAP
# ---------------------------------------------------------------------------

def shap_values(model, X, te, feature_cols: list[str],
                sample: int = None) -> tuple[pd.DataFrame, dict]:
    """
    TreeExplainer SHAP on a subsample of the test fold.

    Returns ``(mean_abs_shap_table, info)``. On any failure the table is empty
    and ``info['status']`` explains why -- SHAP results are never fabricated.
    """
    sample = sample or config.SHAP_SAMPLE_SIZE
    meta: dict = {"requested_sample": sample}
    try:
        import shap
    except Exception as exc:
        meta["status"] = f"shap not installed ({exc}); SHAP analysis omitted"
        warn(meta["status"])
        return pd.DataFrame(), meta

    est = model.named_steps.get("clf", model) if hasattr(model, "named_steps") else model
    if not hasattr(est, "feature_importances_"):
        meta["status"] = ("TreeExplainer requires a tree model; the selected best "
                          "model is not tree-based, so SHAP was omitted rather "
                          "than approximated with a slow KernelExplainer.")
        warn(meta["status"])
        return pd.DataFrame(), meta

    rng = np.random.RandomState(config.RANDOM_STATE)
    idx = te if len(te) <= sample else np.sort(rng.choice(te, sample, replace=False))
    meta["n_explained"] = int(len(idx))

    try:
        with timer("shap.TreeExplainer"):
            expl = shap.TreeExplainer(est)
            sv = expl.shap_values(X[idx], check_additivity=False)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        sv = np.asarray(sv)
        if sv.ndim == 3:                     # (n, features, classes)
            sv = sv[:, :, -1]
        if sv.shape[1] != len(feature_cols):
            meta["status"] = (f"SHAP returned shape {sv.shape}, incompatible with "
                              f"{len(feature_cols)} features; omitted.")
            warn(meta["status"])
            return pd.DataFrame(), meta
    except Exception as exc:
        meta["status"] = f"SHAP computation failed ({exc}); omitted, not faked"
        warn(meta["status"])
        return pd.DataFrame(), meta

    mean_abs = np.abs(sv).mean(axis=0)
    # signed mean tells us the DIRECTION of each feature's typical push
    mean_signed = sv.mean(axis=0)
    out = pd.DataFrame({"feature": feature_cols,
                        "mean_abs_shap": mean_abs,
                        "mean_signed_shap": mean_signed})
    out["share_of_total"] = out.mean_abs_shap / max(out.mean_abs_shap.sum(), 1e-12)
    out["pushes_toward"] = np.where(out.mean_signed_shap > 0, "failure", "no-failure")
    meta["status"] = "ok"
    meta["shap_matrix"] = sv
    meta["explained_index"] = idx
    return out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True), meta


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def run(model_out: dict, df: pd.DataFrame, model_name: str, model) -> dict:
    from . import models as models_mod

    section(f"Explaining: {model_name}")
    cols = model_out["feature_cols"]
    X = df[cols].to_numpy(dtype="float32")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df[config.TARGET].to_numpy(dtype="int8")
    te = model_out["test_index"].get(model_name, np.arange(len(y)))

    res: dict = {"model": model_name}

    nat = native_importance(model, cols)
    if not nat.empty:
        save_table(nat, config.METRICS_DIR / "feature_importance.csv")
        res["native"] = nat
        info(f"top native: {nat.feature.head(5).tolist()}")

    perm = permutation_importance_test(model, X, y, te, cols)
    if not perm.empty:
        save_table(perm, config.METRICS_DIR / "permutation_importance.csv")
        res["permutation"] = perm
        info(f"top permutation: {perm.feature.head(5).tolist()}")

    sh, shmeta = shap_values(model, X, te, cols)
    res["shap_meta"] = {k: v for k, v in shmeta.items()
                        if k not in {"shap_matrix", "explained_index"}}
    if not sh.empty:
        save_table(sh, config.METRICS_DIR / "shap_importance.csv")
        res["shap"] = sh
        res["shap_matrix"] = shmeta.get("shap_matrix")
        res["shap_index"] = shmeta.get("explained_index")
        info(f"top SHAP: {sh.feature.head(5).tolist()}")

    # consensus across the three views -- agreement is what makes an
    # explanation trustworthy, and disagreement is worth reporting
    ranks = []
    if "native" in res:
        ranks.append(res["native"].set_index("feature")["importance"].rank(ascending=False)
                     .rename("rank_native"))
    if "permutation" in res:
        ranks.append(res["permutation"].set_index("feature")["perm_importance_mean"]
                     .rank(ascending=False).rename("rank_permutation"))
    if "shap" in res:
        ranks.append(res["shap"].set_index("feature")["mean_abs_shap"]
                     .rank(ascending=False).rename("rank_shap"))
    if len(ranks) >= 2:
        cons = pd.concat(ranks, axis=1)
        cons["mean_rank"] = cons.mean(axis=1)
        cons = cons.sort_values("mean_rank").reset_index()
        save_table(cons, config.TABLE_DIR / "importance_consensus.csv")
        res["consensus"] = cons
        from scipy import stats as _st
        pairs = {}
        cs = [c for c in cons.columns if c.startswith("rank_")]
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                rho = _st.spearmanr(cons[cs[i]], cons[cs[j]]).statistic
                pairs[f"{cs[i]}__vs__{cs[j]}"] = round(float(rho), 6)
        res["rank_agreement"] = pairs

    _write_report(res, model_out)
    return res


def _write_report(res: dict, model_out: dict) -> None:
    buf = io.StringIO()

    def w(s: str = ""):
        buf.write(s + "\n")

    w("=" * 78)
    w(" EXPLAINABILITY REPORT (STAGE 8)")
    w("=" * 78)
    w()
    w(f"Model explained : {res['model']}")
    w(f"Protocol        : {config.HONEST_PROTOCOL} (the honest split)")
    w(f"Features        : {len(model_out['feature_cols'])}")
    w()
    w("THREE INDEPENDENT VIEWS")
    w("-" * 78)
    w("1. Built-in importance  -- free, but impurity gain is biased toward")
    w("   high-cardinality features, so it is never the sole evidence.")
    w("2. Permutation importance -- measured on the HELD-OUT fold, so it reports")
    w("   what generalises, not what was fitted. Model-agnostic.")
    w("3. SHAP -- per-prediction attributions with direction.")
    w()

    if "native" in res:
        w("=" * 78)
        w(" 1. BUILT-IN FEATURE IMPORTANCE (top 20)")
        w("=" * 78)
        w(res["native"].head(20).to_string(index=False,
                                           float_format=lambda v: f"{v:.6f}"))
        w()

    if "permutation" in res:
        w("=" * 78)
        w(" 2. PERMUTATION IMPORTANCE ON THE TEST FOLD (top 20)")
        w("=" * 78)
        w("Metric: drop in average precision when the feature is shuffled.")
        w("'significant' = mean drop exceeds one standard deviation, i.e. the")
        w("effect is distinguishable from permutation noise.")
        w()
        w(res["permutation"].head(20).to_string(index=False,
                                                float_format=lambda v: f"{v:.6f}"))
        nsig = int(res["permutation"]["significant"].sum())
        w()
        w(f"{nsig} of {len(res['permutation'])} features show an effect above noise.")
        if nsig == 0:
            w("No feature's contribution is distinguishable from noise on the")
            w("held-out fold. Insufficient evidence to name any single driver.")
        w()

    w("=" * 78)
    w(" 3. SHAP")
    w("=" * 78)
    st = res.get("shap_meta", {}).get("status", "not attempted")
    if "shap" in res:
        w(f"Computed on {res['shap_meta'].get('n_explained', 0):,} held-out rows "
          f"with TreeExplainer.")
        w()
        w(res["shap"].head(20).to_string(index=False,
                                         float_format=lambda v: f"{v:.6f}"))
        w()
        w("'pushes_toward' is the sign of the mean SHAP value: which way the")
        w("feature moves the prediction on average.")
    else:
        w(f"SHAP NOT AVAILABLE -- {st}")
        w()
        w("This is recorded as a limitation. No SHAP values were invented to fill")
        w("the gap; the permutation importances above are the model-agnostic")
        w("substitute and were computed on held-out data.")
    w()

    if "rank_agreement" in res:
        w("=" * 78)
        w(" AGREEMENT BETWEEN THE EXPLANATION METHODS")
        w("=" * 78)
        w("Spearman rank correlation between the importance rankings:")
        for k, v in res["rank_agreement"].items():
            w(f"   {k:44s} rho = {v:+.4f}")
        w()
        vals = list(res["rank_agreement"].values())
        if vals and min(vals) >= 0.7:
            w("The methods broadly agree, so the identified drivers are not an")
            w("artefact of one importance definition.")
        else:
            w("The methods DISAGREE materially. Impurity importance and held-out")
            w("permutation importance measure different things (fit vs")
            w("generalisation); where they diverge, the permutation ranking is the")
            w("one to trust, and that divergence is reported rather than smoothed")
            w("over.")
        w()
        w("CONSENSUS TOP 15 (mean rank across available methods)")
        w("-" * 78)
        w(res["consensus"].head(15).to_string(index=False,
                                              float_format=lambda v: f"{v:.2f}"))
    w()
    save_text(buf.getvalue(), config.REPORT_DIR / "explainability_report.txt")
