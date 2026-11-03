"""Paired evaluation of frozen predictions; no fitting or outcome-based selection.

The primary unit is supplied by the experiment, normally one completed window
per user/released prediction day. Bootstrap draws are Poisson(1) multiplicities
of unchanged global user IDs, shared by every dataset, model and contrast.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import pandas as pd

GROUP = ["dataset", "split", "cohort", "model", "representation"]
BASE = ["dataset", "split", "cohort", "model"]
DATASETS = ["NonEnc_desktop", "NonEnc_smartphone", "P2P_desktop", "P2P_smartphone"]
CONTRASTS = [("R2_STATIC", "R1_HISTORY"), ("R3_ADAPTIVE", "R2_STATIC"),
             ("R4_DRIFT", "R3_ADAPTIVE"), ("R4_DRIFT", "R1_HISTORY")]


def _validate(predictions: pd.DataFrame, config: dict) -> pd.DataFrame:
    required = set(GROUP + ["user_id", "row_id", "label", "score"])
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction fields: {sorted(missing)}")
    if predictions.empty:
        raise ValueError("No predictions supplied")
    frame = predictions.copy(deep=True)
    if frame[list(required)].isna().any().any():
        raise ValueError("Prediction keys, labels and scores must not be missing")
    for column in GROUP + ["user_id", "row_id"]:
        if not frame[column].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
            raise ValueError(f"{column} must contain unchanged nonempty strings")
    if not frame.label.isin([0, 1]).all():
        raise ValueError("Evaluation labels must be observed binary 0/1")
    if not pd.api.types.is_numeric_dtype(frame.score) or not np.isfinite(frame.score.to_numpy(dtype=float)).all():
        raise ValueError("Every prediction score must be finite and numeric")
    if frame.duplicated(GROUP + ["row_id"]).any():
        raise ValueError("Duplicate row_id inside a prediction group")
    allowed_cohorts = config.get("cohorts", ["KNOWN", "UNSEEN"])
    if not frame.cohort.isin(allowed_cohorts).all():
        raise ValueError("Unexpected evaluation cohort")
    if frame.groupby(["dataset", "split", "user_id"], sort=False).cohort.nunique().gt(1).any():
        raise ValueError("The same user appears in both evaluation cohorts within a dataset/split")
    # A repeated prediction is allowed; changing its observed identity is not.
    identities = frame.groupby(["dataset", "row_id"], sort=False)[["user_id", "label"]].nunique()
    if identities.gt(1).any().any():
        raise ValueError("A row has conflicting user identity or ground truth")
    time_column = config.get("time_column", "available_at")
    if time_column in frame:
        frame["_prediction_time"] = pd.to_datetime(frame[time_column], errors="raise")
        if frame["_prediction_time"].isna().any():
            raise ValueError("Missing prediction availability time")
        try:
            frame["_prediction_day"] = frame["_prediction_time"].dt.date
        except AttributeError as exc:
            raise ValueError("Prediction times must use one declared released clock") from exc
        clock_conflicts = frame.groupby(["dataset", "row_id"], sort=False)["_prediction_time"].nunique()
        if clock_conflicts.gt(1).any():
            raise ValueError("A paired row has inconsistent availability time")
    elif config.get("primary_population", "USER_DAY_PRIMARY") == "USER_DAY_PRIMARY":
        raise ValueError("USER_DAY_PRIMARY requires available_at or configured time_column")
    if config.get("primary_population", "USER_DAY_PRIMARY") == "USER_DAY_PRIMARY":
        if frame.duplicated(GROUP + ["user_id", "_prediction_day"]).any():
            raise ValueError("USER_DAY_PRIMARY contains multiple observations per user/day")
    return frame


def latest_user_day_rows(frame: pd.DataFrame, time_column: str = "available_at") -> pd.DataFrame:
    """Label-free sensitivity selection, with deterministic row-ID ties.

    Selection is within each prediction group, user and supplied-clock day.
    This helper must not be used to decide which outcomes are retained.
    """
    selected = frame.copy(deep=True)
    selected["_latest_time"] = pd.to_datetime(selected[time_column], errors="raise")
    if selected["_latest_time"].isna().any():
        raise ValueError("Missing availability time for user-day selection")
    selected["_latest_day"] = selected["_latest_time"].dt.date
    selected = selected.sort_values(["_latest_time", "row_id"], ascending=[False, True], kind="stable")
    selected = selected.drop_duplicates(GROUP + ["user_id", "_latest_day"], keep="first")
    return selected.drop(columns=["_latest_time", "_latest_day"]).sort_index()


def _rank_metrics(y: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """AP uses distinct-score thresholds, preserving exact ties like sklearn."""
    positive = float(y.sum())
    negative = float(len(y) - positive)
    if positive == 0 or negative == 0:
        return math.nan, math.nan
    order = np.argsort(-scores, kind="stable")
    ys, ss = y[order], scores[order]
    ends = np.r_[np.flatnonzero(ss[1:] != ss[:-1]), len(ss) - 1]
    tp = np.cumsum(ys, dtype=float)[ends]
    total = (ends + 1).astype(float)
    increments = np.diff(np.r_[0.0, tp])
    ap = float(np.sum(increments * tp / total) / positive)
    fp = total - tp
    neg_groups = np.diff(np.r_[0.0, fp])
    # Descending scores: lower-ranked negatives lose to each positive group;
    # tied positives/negatives contribute one half.
    auc = float(np.sum(increments * (negative - fp + 0.5 * neg_groups)) / (positive * negative))
    return ap, auc


def _capacity_metrics(frame: pd.DataFrame, fraction: float, daily: bool) -> dict:
    selected, true_positive = 0, 0
    batches = frame.groupby("_prediction_day", sort=False) if daily else [(None, frame)]
    for _, batch in batches:
        count = max(1, int(math.ceil(fraction * len(batch))))
        ranked = batch.sort_values(["score", "row_id"], ascending=[False, True], kind="stable")
        chosen = ranked.iloc[:count]
        selected += len(chosen)
        true_positive += int(chosen.label.sum())
    positives = int(frame.label.sum())
    return {"precision": true_positive / selected if selected else math.nan,
            "recall": true_positive / positives if positives else math.nan,
            "selected_rows": selected, "selected_positive_rows": true_positive,
            "realized_capacity": selected / len(frame)}


def _uncalibrated(model: str, config: dict) -> bool:
    names = config.get("uncalibrated_models", ["IF", "IsolationForest", "Isolation Forest"])
    normalize = lambda value: "".join(c for c in value.upper() if c.isalnum())
    return normalize(model) in {normalize(name) for name in names}


def _point_metrics(frame: pd.DataFrame, population: str, config: dict) -> list[dict]:
    rows = []
    capacities = config.get("capacities", [0.01, 0.05])
    if any(not 0 < float(v) <= 1 for v in capacities):
        raise ValueError("Capacity fractions must lie in (0,1]")
    scope = config.get("capacity_scope", "PER_PREDICTION_DAY" if "_prediction_day" in frame else "SPLIT_COHORT")
    if scope not in {"PER_PREDICTION_DAY", "SPLIT_COHORT"}:
        raise ValueError("Unknown capacity_scope")
    if scope == "PER_PREDICTION_DAY" and "_prediction_day" not in frame:
        raise ValueError("Daily capacity requires prediction availability times")
    for key, group in frame.groupby(GROUP, sort=True):
        identity = dict(zip(GROUP, key))
        y = group.label.to_numpy(dtype=float)
        scores = group.score.to_numpy(dtype=float)
        positives = int(y.sum())
        common = dict(identity, population=population, n_rows=len(group), n_users=int(group.user_id.nunique()),
                      positive_rows=positives, negative_rows=len(group) - positives,
                      positive_users=int(group.loc[group.label.eq(1), "user_id"].nunique()),
                      negative_users=int(group.loc[group.label.eq(0), "user_id"].nunique()),
                      label_prevalence=float(y.mean()), weighting="ONE_WEIGHT_PER_SUPPLIED_OBSERVATION")
        ap, auc = _rank_metrics(y, scores)
        for metric, value in [("AP", ap), ("ROC_AUC", auc)]:
            rows.append(dict(common, metric=metric, value=value, status="ESTIMABLE" if np.isfinite(value) else "NOT_ESTIMABLE",
                             reason="" if np.isfinite(value) else "BOTH_OBSERVED_CLASSES_REQUIRED", capacity_fraction=math.nan))
        brier_allowed = not _uncalibrated(identity["model"], config) and bool(((scores >= 0) & (scores <= 1)).all())
        rows.append(dict(common, metric="BRIER", value=float(np.mean((scores - y) ** 2)) if brier_allowed else math.nan,
                         status="ESTIMABLE" if brier_allowed else "NOT_APPLICABLE", reason="" if brier_allowed else "UNCALIBRATED_OR_UNBOUNDED_SCORE",
                         capacity_fraction=math.nan))
        for fraction in capacities:
            capacity = _capacity_metrics(group, float(fraction), scope == "PER_PREDICTION_DAY")
            for metric, value in [("PRECISION_AT_CAPACITY", capacity["precision"]), ("RECALL_AT_CAPACITY", capacity["recall"])]:
                rows.append(dict(common, metric=metric, value=value, status="ESTIMABLE" if np.isfinite(value) else "NOT_ESTIMABLE",
                                 reason="" if np.isfinite(value) else "NO_OBSERVED_POSITIVES", capacity_fraction=float(fraction),
                                 capacity_scope=scope, selected_rows=capacity["selected_rows"],
                                 selected_positive_rows=capacity["selected_positive_rows"], realized_capacity=capacity["realized_capacity"]))
    return rows


def _aligned(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = left.sort_values("row_id", kind="stable").reset_index(drop=True)
    right = right.sort_values("row_id", kind="stable").reset_index(drop=True)
    columns = ["row_id", "user_id", "label"]
    if "_prediction_time" in left and "_prediction_time" in right:
        columns.append("_prediction_time")
    if len(left) != len(right) or not left[columns].equals(right[columns]):
        raise ValueError("Paired representations must have exactly identical eligible rows, users, labels and availability times")
    return left, right


def _bootstrap_ap(frame: pd.DataFrame, weights: np.ndarray, user_index: dict[str, int], chunk_size: int) -> np.ndarray:
    """Exact multiplicity-weighted AP, computed in bounded NumPy chunks."""
    order = np.argsort(-frame.score.to_numpy(dtype=float), kind="stable")
    y = frame.label.to_numpy(dtype=float)[order]
    scores = frame.score.to_numpy(dtype=float)[order]
    users = np.array([user_index[u] for u in frame.user_id], dtype=np.int64)[order]
    ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(scores) - 1]
    result = np.full(len(weights), np.nan)
    for start in range(0, len(weights), chunk_size):
        stop = min(start + chunk_size, len(weights))
        row_weights = weights[start:stop, users].astype(float)
        cumulative_total = np.cumsum(row_weights, axis=1)[:, ends]
        cumulative_positive = np.cumsum(row_weights * y, axis=1)[:, ends]
        total_positive = cumulative_positive[:, -1]
        total_negative = cumulative_total[:, -1] - total_positive
        valid = (total_positive > 0) & (total_negative > 0)
        precision = np.divide(cumulative_positive, cumulative_total, out=np.zeros_like(cumulative_positive), where=cumulative_total > 0)
        increments = np.diff(cumulative_positive, axis=1, prepend=0.0)
        weighted_sum = np.sum(increments * precision, axis=1)
        values = np.divide(weighted_sum, total_positive, out=np.full(stop - start, np.nan), where=valid)
        result[start:stop] = values
    return result


def _dataset_names(config: dict) -> list[str]:
    names = config.get("datasets", DATASETS)
    result = [entry["dataset"] if isinstance(entry, dict) else entry for entry in names]
    if not result or len(set(result)) != len(result):
        raise ValueError("Primary dataset list must be nonempty and unique")
    return result


def evaluate_predictions(predictions_df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return long point metrics, strictly paired AP contrasts and audit metadata.

    Required config choices may override primary_models (LR,HGB), datasets,
    bootstrap_repeats (20,000), seed (42), capacities (.01,.05), and population.
    Empty/missing fits remain NOT_ESTIMABLE in the prespecified primary family.
    Bootstrap CIs condition on fitted predictions and do not include refit/HPO
    uncertainty. No p-values or significance flags are manufactured from CIs.
    """
    frame = _validate(predictions_df, config)
    population = config.get("primary_population", "USER_DAY_PRIMARY")
    models = list(config.get("primary_models", ["LR", "HGB"]))
    if not models or len(set(models)) != len(models):
        raise ValueError("primary_models must be nonempty and unique")
    datasets = _dataset_names(config)
    contrasts = [tuple(item) for item in config.get("primary_contrasts", CONTRASTS)]
    if not contrasts or len(set(contrasts)) != len(contrasts) or any(len(c) != 2 or c[0] == c[1] for c in contrasts):
        raise ValueError("Primary contrasts must be unique ordered representation pairs")
    secondary_contrasts = [tuple(item) for item in config.get("secondary_contrasts", [
        ("R4_DRIFT", "C_ACTIVITY"), ("R4_DRIFT", "C_SHUFFLED")])]
    if any(len(c) != 2 or c[0] == c[1] for c in secondary_contrasts):
        raise ValueError("Secondary contrasts must be ordered representation pairs")
    all_contrasts = list(dict.fromkeys(contrasts + secondary_contrasts))
    repeats = int(config.get("bootstrap_repeats", 20_000))
    seed = int(config.get("seed", 42))
    chunk = int(config.get("bootstrap_chunk_size", 64))
    min_fraction = float(config.get("minimum_valid_bootstrap_fraction", 0.95))
    alpha = float(config.get("familywise_alpha", 0.05))
    if repeats < 2 or chunk < 1 or not 0 < min_fraction <= 1 or not 0 < alpha < 1:
        raise ValueError("Invalid bootstrap settings")
    family_size = len(datasets) * len(models) * len(contrasts)
    primary_split, primary_cohort = config.get("primary_split", "FINAL"), config.get("primary_cohort", "UNSEEN")
    users = sorted(frame.user_id.unique())
    user_index = {user: index for index, user in enumerate(users)}
    weights = np.random.default_rng(seed).poisson(1.0, size=(repeats, len(users))).astype(np.uint16)
    metrics = _point_metrics(frame, population, config)
    sensitivity = "NOT_REQUIRED_PRIMARY_ALREADY_USER_DAY"
    time_column = config.get("time_column", "available_at")
    if population != "USER_DAY_PRIMARY" and time_column in frame:
        latest = latest_user_day_rows(frame, time_column)
        metrics.extend(_point_metrics(latest, "LATEST_USER_DAY_SENSITIVITY", config))
        sensitivity = "COMPUTED_LABEL_FREE_LATEST_USER_DAY"
    elif population != "USER_DAY_PRIMARY":
        sensitivity = "NOT_ESTIMABLE_NO_AVAILABILITY_TIME"
    grouped = {key: group for key, group in frame.groupby(GROUP, sort=False)}
    bases = {key[:4] for key in grouped}
    bases.update((dataset, primary_split, primary_cohort, model) for dataset in datasets for model in models)
    bootstrap_cache = {}
    paired = []
    warnings = []
    for base in sorted(bases):
        primary_base = base[0] in datasets and base[1] == primary_split and base[2] == primary_cohort and base[3] in models
        for higher, lower in all_contrasts:
            is_primary = primary_base and (higher, lower) in contrasts
            left, right = grouped.get(base + (higher,)), grouped.get(base + (lower,))
            if not is_primary and (left is None or right is None):
                continue
            record = dict(zip(BASE, base))
            record.update(population=population, metric="AP", contrast=f"{higher} - {lower}",
                          higher_representation=higher, lower_representation=lower, primary=is_primary,
                          family_size=family_size if is_primary else 0, effect=math.nan,
                          higher_value=math.nan, lower_value=math.nan, ci95_lower=math.nan, ci95_upper=math.nan,
                          bonferroni_lower=math.nan, bonferroni_upper=math.nan, bootstrap_draws=repeats if is_primary else 0,
                          valid_draws=0, valid_fraction=math.nan, n_rows=0, n_users=0,
                          status="NOT_ESTIMABLE", reason="MISSING_PREDICTION_GROUP")
            if left is None or right is None:
                paired.append(record)
                continue
            left, right = _aligned(left, right)
            record.update(n_rows=len(left), n_users=int(left.user_id.nunique()))
            ap_left = _rank_metrics(left.label.to_numpy(dtype=float), left.score.to_numpy(dtype=float))[0]
            ap_right = _rank_metrics(right.label.to_numpy(dtype=float), right.score.to_numpy(dtype=float))[0]
            if not np.isfinite(ap_left) or not np.isfinite(ap_right):
                record["reason"] = "BOTH_OBSERVED_CLASSES_REQUIRED"
                paired.append(record)
                continue
            record.update(higher_value=ap_left, lower_value=ap_right, effect=ap_left - ap_right,
                          status="POINT_ESTIMATE_ONLY", reason="NONPRIMARY_DESCRIPTIVE_COMPARISON")
            if is_primary:
                for representation, group in [(higher, left), (lower, right)]:
                    key = base + (representation,)
                    if key not in bootstrap_cache:
                        bootstrap_cache[key] = _bootstrap_ap(group, weights, user_index, chunk)
                differences = bootstrap_cache[base + (higher,)] - bootstrap_cache[base + (lower,)]
                valid = differences[np.isfinite(differences)]
                record.update(valid_draws=len(valid), valid_fraction=len(valid) / repeats)
                if len(valid) / repeats < min_fraction:
                    record.update(status="NOT_ESTIMABLE_UNCERTAINTY", reason="INSUFFICIENT_TWO_CLASS_CLUSTER_DRAWS")
                else:
                    lower_tail = alpha / (2 * family_size)
                    q = np.quantile(valid, [0.025, 0.975, lower_tail, 1 - lower_tail], method="linear")
                    record.update(ci95_lower=float(q[0]), ci95_upper=float(q[1]), bonferroni_lower=float(q[2]),
                                  bonferroni_upper=float(q[3]), status="ESTIMABLE", reason="")
            paired.append(record)
    tail_draws = repeats * alpha / (2 * family_size)
    if tail_draws < 20:
        warnings.append(f"Bonferroni tail quantiles use only {tail_draws:.2f} expected draws per tail; marginal bounds have Monte Carlo uncertainty.")
    metadata = {
        "primary_population": population, "primary_metric": "AP", "primary_split": primary_split,
        "primary_cohort": primary_cohort, "primary_models": models, "primary_datasets": datasets,
        "primary_contrasts": [list(c) for c in contrasts], "primary_family_size": family_size,
        "secondary_descriptive_contrasts": [list(c) for c in secondary_contrasts],
        "family_not_reduced_for_missing_or_unestimable_comparisons": True,
        "bootstrap_method": "PAIRED_GLOBAL_USER_POISSON_1_MULTIPLICITIES",
        "bootstrap_repeats": repeats, "seed": seed, "global_user_count": len(users),
        "user_order_sha256": hashlib.sha256("\n".join(users).encode("utf-8")).hexdigest(),
        "shared_draw_matrix_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
        "bootstrap_chunk_size": chunk, "minimum_valid_bootstrap_fraction": min_fraction,
        "pointwise_confidence_level": 0.95, "familywise_alpha": alpha,
        "bonferroni_quantiles": [alpha / (2 * family_size), 1 - alpha / (2 * family_size)],
        "expected_draws_per_bonferroni_tail": tail_draws, "quantile_method": "linear",
        "uncertainty_scope": "CONDITIONAL_ON_FROZEN_PREDICTIONS_NO_MODEL_REFITTING",
        "score_tie_policy_ap": "GROUP_EQUAL_SCORES_AT_COMMON_THRESHOLD",
        "capacity_tie_policy": "SCORE_DESCENDING_ROW_ID_ASCENDING",
        "capacity_rounding": "MAX_1_CEIL_FRACTION_TIMES_BATCH_ROWS",
        "capacity_scope": config.get("capacity_scope", "PER_PREDICTION_DAY" if "_prediction_day" in frame else "SPLIT_COHORT"),
        "daily_sensitivity": sensitivity, "models_fitted": 0, "p_values_computed": False,
        "significance_claims_generated": False, "warnings": warnings,
    }
    return pd.DataFrame(metrics), pd.DataFrame(paired), metadata
