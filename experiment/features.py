"""Causal behavioral-reference features for completed-window risk ranking.

Sources are thinned within each user by a deterministic greedy rule: retain
the first source, then retain a source only if its timestamp is strictly more
than2hours after the last retained source. A current query can use only these
retained sources whose timestamp+2hours is strictly before the query timestamp.
Thus historical source windows are mutually disjoint under the registered
conditional envelopes, and are disjoint from CURRENT. Labels are never read.

Static, adaptive, simple-history and drift blocks share exactly this source
sequence. No source is called safe or normal. Missing/nonfinite measurements
become NaN; all imputation, fitted scaling and train-specific feature filtering
belong to the downstream training pipeline. Fixed deviation normalization uses
the relevant *past-only within-user* population standard deviation plus1e-6
in that field's provided units; it is not a fitted global/train scaler.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


HOUR_NS = 3600 * 1_000_000_000
DEFAULTS = {
    "history_gap_hours": 2.0,
    "source_thinning_gap_hours": 2.0,
    "static_count": 3,
    "ewma_alpha": 0.1,
    "drift_window": 6,
    "min_history_count": 12,
    "min_history_dates": 3,
    "epsilon": 1e-6,
    "deviation_clip": 25.0,
    "min_static_finite": 2,
    "min_drift_window_finite": 3,
}


def _configuration(config: dict | None) -> dict:
    if config is not None and not isinstance(config, dict):
        raise ValueError("config must be a dict or None")
    supplied = config or {}
    cfg = {key: supplied.get(key, default) for key, default in DEFAULTS.items()}
    for key in ("static_count", "drift_window", "min_history_count", "min_history_dates", "min_static_finite", "min_drift_window_finite"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
        cfg[key] = int(value)
    for key in ("history_gap_hours", "source_thinning_gap_hours", "epsilon", "ewma_alpha"):
        value = float(cfg[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
        cfg[key] = value
    if cfg["ewma_alpha"] > 1:
        raise ValueError("ewma_alpha must be at most1")
    # Smaller gaps would violate the one-hour envelope contract.
    if cfg["history_gap_hours"] < 2 or cfg["source_thinning_gap_hours"] < 2:
        raise ValueError("Source/current and source/source gaps must each be at least2hours")
    if cfg["min_static_finite"] > cfg["static_count"]:
        raise ValueError("min_static_finite exceeds static_count")
    if cfg["min_drift_window_finite"] > cfg["drift_window"]:
        raise ValueError("min_drift_window_finite exceeds drift_window")
    clip = cfg["deviation_clip"]
    if clip is not None:
        clip = float(clip)
        if not math.isfinite(clip) or clip <= 0:
            raise ValueError("deviation_clip must be positive finite or None")
    cfg["deviation_clip"] = clip
    return cfg


def _greedy_sources(times_ns: np.ndarray, gap_ns: int) -> np.ndarray:
    if not len(times_ns):
        return np.empty(0, dtype=np.int64)
    retained = [0]
    last_time = int(times_ns[0])
    for position in range(1, len(times_ns)):
        current = int(times_ns[position])
        if current - last_time > gap_ns:
            retained.append(position)
            last_time = current
    return np.asarray(retained, dtype=np.int64)


def _absolute_z(current: np.ndarray, mean: np.ndarray, variance: np.ndarray, epsilon: float, clip: float | None) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        score = np.abs(current - mean) / (np.sqrt(np.maximum(variance, 0.0)) + epsilon)
    score[~np.isfinite(score)] = np.nan
    if clip is not None:
        np.minimum(score, clip, out=score)
    return score


def _finite_float32(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if len(finite) and np.max(np.abs(finite)) > np.finfo(np.float32).max:
        raise ValueError("A finite feature exceeds float32 range; silent overflow is forbidden")
    return values.astype(np.float32, copy=False)


def build_features(frame: pd.DataFrame, predictor_names: list[str], config: dict | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return all input rows' numeric float32 features and block/row metadata.

    Input must be contiguous by exact unchanged user ID and strictly increasing
    in timestamp inside each user. ``row_id`` must be unique/nonmissing. The
    returned index is the original row_id sequence, without cohort filtering.
    The input label column, when present, is deliberately never accessed.
    Config is the flat feature-parameter mapping; unspecified keys use DEFAULTS.
    """
    cfg = _configuration(config)
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("frame must be a pandas DataFrame")
    if not predictor_names or len(set(predictor_names)) != len(predictor_names) or any(not isinstance(name, str) or not name for name in predictor_names):
        raise ValueError("predictor_names must be a nonempty unique string list")
    if set(predictor_names) & {"row_id", "user_id", "timestamp", "label", "entity"}:
        raise ValueError("Identifiers, timestamps, metadata and labels cannot be predictors")
    missing = {"row_id", "user_id", "timestamp", *predictor_names} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame["row_id"].isna().any() or frame["row_id"].duplicated().any():
        raise ValueError("row_id must be nonmissing and unique")
    if frame["user_id"].isna().any() or not frame["user_id"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Exact nonempty original string user IDs are required")
    times = frame["timestamp"]
    if not is_datetime64_any_dtype(times.dtype) or times.isna().any():
        raise ValueError("timestamp must contain parsed nonmissing datetime values")
    time_index = pd.DatetimeIndex(times).as_unit("ns")
    all_times = time_index.asi8
    gap_ns = int(round(cfg["history_gap_hours"] * HOUR_NS))
    source_gap_ns = int(round(cfg["source_thinning_gap_hours"] * HOUR_NS))
    if len(all_times) and int(all_times.min()) <= np.iinfo(np.int64).min + gap_ns:
        raise ValueError("Timestamps too close to int64 lower bound")
    user_codes, unique_users = pd.factorize(frame["user_id"], sort=False)
    boundaries = np.r_[0, np.flatnonzero(user_codes[1:] != user_codes[:-1]) + 1, len(frame)] if len(frame) else np.array([0], dtype=np.int64)
    if len(boundaries) - 1 != len(unique_users):
        raise ValueError("Input must be contiguous by user; repeated user groups are forbidden")
    if len(frame) > 1 and np.any((all_times[1:] <= all_times[:-1]) & (user_codes[1:] == user_codes[:-1])):
        raise ValueError("Timestamps must be unique and strictly increasing within each user")
    try:
        values = frame[predictor_names].to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("All predictors must be numeric") from exc
    values[~np.isfinite(values)] = np.nan
    p, n = len(predictor_names), len(frame)
    blocks = {
        "CURRENT": ["current__" + name for name in predictor_names],
        "SIMPLE_HISTORY": ["history_mean__" + name for name in predictor_names] + ["history__log1p_count", "history__log1p_last_age_hours", "history__span_hours"],
        "STATIC_DEVIATION": ["static_abs_z__" + name for name in predictor_names],
        "ADAPTIVE_DEVIATION": ["adaptive_abs_z__" + name for name in predictor_names] + ["adaptive_movement_abs_z__" + name for name in predictor_names],
        "DRIFT": ["drift_abs_z__" + name for name in predictor_names] + ["drift_change_abs__" + name for name in predictor_names],
    }
    row_columns = ["meta__eligible_history_count", "meta__history_distinct_prediction_dates", "meta__history_span_hours", "meta__history_last_age_hours", "meta__common_cohort_eligible", "meta__accepted_source_indicator", "meta__minimum_finite_history_count"]
    columns = sum(blocks.values(), []) + row_columns
    matrix = np.full((n, len(columns)), np.nan, dtype=np.float32)
    slices = {}
    offset = 0
    for block, names in blocks.items():
        slices[block] = slice(offset, offset + len(names))
        offset += len(names)
    slices["ROW_METADATA"] = slice(offset, len(columns))
    matrix[:, slices["CURRENT"]] = _finite_float32(values)
    accepted_sources_total = 0
    epsilon, clip = cfg["epsilon"], cfg["deviation_clip"]
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        user_times = all_times[left:right]
        current = values[left:right]
        accepted_positions = _greedy_sources(user_times, source_gap_ns)
        accepted_sources_total += len(accepted_positions)
        accepted_times = user_times[accepted_positions]
        source = current[accepted_positions]
        eligible = np.searchsorted(accepted_times, user_times - gap_ns, side="left")
        m = len(source)
        valid = np.isfinite(source)
        cumulative_count = np.vstack([np.zeros((1, p), dtype=np.int64), np.cumsum(valid, axis=0)])
        cumulative_sum = np.vstack([np.zeros((1, p), dtype=np.float64), np.cumsum(np.where(valid, source, 0.0), axis=0)])
        past_count = cumulative_count[eligible]
        past_mean = np.full_like(current, np.nan)
        np.divide(cumulative_sum[eligible], past_count, out=past_mean, where=past_count > 0)
        last_positions = np.maximum(eligible - 1, 0)
        has_history = eligible > 0
        span = np.where(has_history, (accepted_times[last_positions] - accepted_times[0]) / HOUR_NS, np.nan)
        last_age = np.where(has_history, (user_times - accepted_times[last_positions]) / HOUR_NS, np.nan)
        source_dates = (time_index[left:right][accepted_positions] + pd.Timedelta(hours=1)).normalize().asi8
        new_date = np.r_[True, source_dates[1:] != source_dates[:-1]]
        date_prefix = np.r_[0, np.cumsum(new_date)]
        distinct_dates = date_prefix[eligible]
        simple = np.column_stack([past_mean, np.log1p(eligible), np.log1p(last_age), span])
        matrix[left:right, slices["SIMPLE_HISTORY"]] = _finite_float32(simple)

        static_mean = np.full(p, np.nan)
        static_var = np.full(p, np.nan)
        k = cfg["static_count"]
        if m >= k:
            anchor = source[:k]
            anchor_valid = np.isfinite(anchor)
            anchor_count = anchor_valid.sum(axis=0)
            good = anchor_count >= cfg["min_static_finite"]
            np.divide(np.where(anchor_valid, anchor, 0.0).sum(axis=0), anchor_count, out=static_mean, where=good)
            centered = np.where(anchor_valid, anchor - static_mean, 0.0)
            np.divide((centered * centered).sum(axis=0), anchor_count, out=static_var, where=good)
        static_score = _absolute_z(current, static_mean, static_var, epsilon, clip)
        static_score[eligible < k] = np.nan
        matrix[left:right, slices["STATIC_DEVIATION"]] = _finite_float32(static_score)

        source_frame = pd.DataFrame(source, copy=False)
        ewm = source_frame.ewm(alpha=cfg["ewma_alpha"], adjust=False, ignore_na=True)
        ewm_mean = np.vstack([np.full((1, p), np.nan), ewm.mean().to_numpy()])
        ewm_var = np.vstack([np.full((1, p), np.nan), ewm.var(bias=True).to_numpy()])
        adaptive_mean, adaptive_var = ewm_mean[eligible], ewm_var[eligible]
        adaptive_score = _absolute_z(current, adaptive_mean, adaptive_var, epsilon, clip)
        adaptive_score[past_count < 2] = np.nan
        movement = _absolute_z(adaptive_mean, static_mean, static_var, epsilon, clip)
        movement[(past_count < 2) | (eligible[:, None] < k)] = np.nan
        matrix[left:right, slices["ADAPTIVE_DEVIATION"]] = _finite_float32(np.column_stack([adaptive_score, movement]))

        w = cfg["drift_window"]
        rolling = source_frame.rolling(w, min_periods=cfg["min_drift_window_finite"])
        means = rolling.mean().to_numpy()
        variances = rolling.var(ddof=0).to_numpy()
        # drift_by_count[q] is computed only from the first q accepted sources.
        drift_by_count = np.full((m + 1, p), np.nan)
        if m >= 2 * w:
            q = np.arange(2 * w, m + 1)
            recent_end, older_end = q - 1, q - w - 1
            pooled_var = 0.5 * (variances[recent_end] + variances[older_end])
            drift_by_count[q] = _absolute_z(means[recent_end], means[older_end], pooled_var, epsilon, clip)
        drift = drift_by_count[eligible]
        trajectory = np.abs(drift - drift_by_count[np.maximum(eligible - 1, 0)])
        matrix[left:right, slices["DRIFT"]] = _finite_float32(np.column_stack([drift, trajectory]))
        accepted_indicator = np.zeros(right - left, dtype=np.float32)
        accepted_indicator[accepted_positions] = 1
        cohort = (eligible >= cfg["min_history_count"]) & (distinct_dates >= cfg["min_history_dates"])
        row_data = np.column_stack([eligible, distinct_dates, span, last_age, cohort, accepted_indicator, past_count.min(axis=1)])
        matrix[left:right, slices["ROW_METADATA"]] = _finite_float32(row_data)

    representations = {}
    accumulated = []
    for number, block in enumerate(("CURRENT", "SIMPLE_HISTORY", "STATIC_DEVIATION", "ADAPTIVE_DEVIATION", "DRIFT")):
        accumulated = accumulated + blocks[block]
        representations[f"R{number}"] = accumulated.copy()
    activity = blocks["SIMPLE_HISTORY"][-3:]
    blocks["ACTIVITY_HISTORY"] = activity.copy()
    features = pd.DataFrame(matrix, columns=columns, index=pd.Index(frame["row_id"].to_numpy(copy=True), name="row_id"), copy=False)
    metadata = {
        "version": "MEREI_CAUSAL_PROFILE_FEATURES_V1", "parameters": cfg,
        "representations": representations, "blocks": blocks,
        "control_representations": {"CURRENT_ACTIVITY_HISTORY": blocks["CURRENT"] + activity},
        "feature_columns": sum((blocks[b] for b in ("CURRENT", "SIMPLE_HISTORY", "STATIC_DEVIATION", "ADAPTIVE_DEVIATION", "DRIFT")), []),
        "row_metadata_columns": row_columns,
        "history_count_column": "meta__eligible_history_count", "history_dates_column": "meta__history_distinct_prediction_dates",
        "history_span_column": "meta__history_span_hours", "common_cohort_column": "meta__common_cohort_eligible",
        "predictor_names": predictor_names.copy(), "rows": int(n), "users": int(len(unique_users)),
        "accepted_source_rows": int(accepted_sources_total), "thinned_source_rows": int(n - accepted_sources_total),
        "common_cohort_rows": int(features["meta__common_cohort_eligible"].sum()),
        "source_thinning": "GREEDY_PER_USER_SOURCE_T_STRICTLY_GREATER_THAN_LAST_ACCEPTED_T_PLUS_CONFIG_GAP",
        "query_history_rule": "ACCEPTED_SOURCE_T_PLUS_CONFIG_GAP_STRICTLY_LESS_THAN_CURRENT_T",
        "prediction_date_rule": "SOURCE_TIMESTAMP_PLUS_1H_DATE_ON_SUPPLIED_CLOCK",
        "clock_policy": "PRESERVE_SUPPLIED_CLOCK_NO_ASSUMED_TIMEZONE",
        "static_definition": "FIRST_K_ACCEPTED_SOURCES; population variance; per-field finite subset subject to configured minimum",
        "adaptive_definition": "EWMA adjust=False ignore_na=True; central variance bias=True; alpha applied per finite observation",
        "adaptive_half_life_finite_observations": 0.0 if cfg["ewma_alpha"] == 1 else math.log(0.5) / math.log(1 - cfg["ewma_alpha"]),
        "movement_definition": "ABS(EWMA_MEAN-STATIC_MEAN)/(STATIC_POPULATION_STD+EPSILON)",
        "drift_definition": "ABS(MEAN_LAST_W-MEAN_PREVIOUS_W)/(SQRT((VAR_LAST_W+VAR_PREVIOUS_W)/2)+EPSILON)",
        "trajectory_definition": "ABS(DRIFT_AT_Q_ACCEPTED_SOURCES-DRIFT_AT_Q_MINUS_1); no current values or labels",
        "history_age_rule": "CURRENT_SOURCE_TIMESTAMP_MINUS_LAST_ELIGIBLE_SOURCE_TIMESTAMP_IN_HOURS",
        "history_span_scope": "LAST_MINUS_FIRST_ELIGIBLE_SOURCE_TIMESTAMP; not continuous observation coverage",
        "normalization_scope": "PAST_ONLY_WITHIN_USER_REFERENCE_UNITS; downstream fitted scalers remain train-only",
        "missing_policy": "NONFINITE_INPUT_AS_NAN; no global imputation; insufficient baseline/window support stays NaN",
        "source_label_used": False, "reference_is_assumed_safe": False, "input_rows_filtered": False,
        "output_dtype": "float32", "model_fits": 0,
    }
    return features, metadata
