"""Train-only preprocessing and frozen ranking models for the four-dataset study.

These utilities do not choose splits, tune parameters, or use evaluation labels.
Balanced supervised probabilities are uncalibrated ranking scores. Isolation
Forest uses benign training observations only and returns raw anomaly scores.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
from pathlib import Path
import platform
from typing import Any

import numpy as np


_ALIASES = {
    "lr": "LR", "logistic_regression": "LR", "logisticregression": "LR",
    "hgb": "HGB", "hist_gradient_boosting": "HGB",
    "histgradientboosting": "HGB",
    "rf": "RF", "random_forest": "RF", "randomforest": "RF",
    "xgb": "XGB", "xgboost": "XGB",
    "if": "IF", "isolation_forest": "IF", "isolationforest": "IF",
}


def supported_model_names() -> tuple[str, ...]:
    return ("LR", "HGB")


def _canonical_model_name(name: str) -> str:
    try:
        value = _ALIASES[str(name).lower().replace("-", "_")]
        if value not in ("LR", "HGB"): raise KeyError(value)
        return value
    except KeyError as exc:
        raise ValueError(f"Unknown model {name!r}; choose {supported_model_names()}") from exc


def _matrix_and_names(X: Any, feature_names: Any = None) -> tuple[np.ndarray, tuple[str, ...]]:
    if hasattr(X, "columns"):
        names = tuple(str(c) for c in X.columns)
        if feature_names is not None and tuple(feature_names) != names:
            raise ValueError("Configured feature_names differ from DataFrame columns")
    else:
        shape = getattr(X, "shape", None)
        if shape is None:
            shape = np.asarray(X).shape
        if len(shape) != 2:
            raise ValueError("X must be a two-dimensional numeric matrix")
        names = tuple(str(v) for v in feature_names) if feature_names is not None else tuple(
            f"x{i:03d}" for i in range(shape[1]))
    if len(names) != len(set(names)):
        raise ValueError("Feature names must be unique")
    matrix = np.asarray(X, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError("X shape and feature names do not match")
    if matrix.shape[1] == 0:
        raise ValueError("At least one input feature is required")
    return matrix, names


def _ordered_matrix(X: Any, names: tuple[str, ...]) -> np.ndarray:
    if hasattr(X, "columns"):
        incoming = tuple(str(c) for c in X.columns)
        if len(incoming) != len(set(incoming)):
            raise ValueError("Evaluation feature names must be unique")
        if set(incoming) != set(names):
            raise ValueError("Evaluation columns differ from the frozen training schema")
        positions = [incoming.index(name) for name in names]
        matrix = np.asarray(X, dtype=np.float64)[:, positions]
    else:
        matrix = np.asarray(X, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError("Evaluation matrix does not match the frozen training schema")
    return matrix


def _labels(y: Any, rows: int) -> np.ndarray:
    values = np.asarray(y)
    if values.ndim != 1 or len(values) != rows:
        raise ValueError("train_y must be one-dimensional and aligned with train_X")
    if not np.isin(values, [0, 1]).all():
        raise ValueError("Only known binary training labels 0 and 1 are allowed")
    return values.astype(np.int8, copy=False)


def _versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    for package in ("scikit-learn", "scipy", "xgboost"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


@dataclass
class FittedRankingPipeline:
    """Picklable train schema, preprocessing statistics, and fitted estimator."""

    model_name: str
    feature_names: tuple[str, ...]
    kept_indices: np.ndarray
    medians: np.ndarray
    estimator: Any
    scaler: Any = None
    thread_limit: int = 2

    def transform(self, X: Any) -> np.ndarray:
        matrix = _ordered_matrix(X, self.feature_names)
        selected = matrix[:, self.kept_indices].copy()
        bad_rows, bad_cols = np.where(~np.isfinite(selected))
        selected[bad_rows, bad_cols] = self.medians[bad_cols]
        if self.scaler is not None and len(selected):
            selected = self.scaler.transform(selected)
        return selected

    def predict_scores(self, X: Any) -> np.ndarray:
        return predict_scores(self, X)


def predict_scores(pipeline: FittedRankingPipeline, X: Any) -> np.ndarray:
    """Replay scores without refitting or modifying preprocessing statistics."""
    transformed = pipeline.transform(X)
    if len(transformed) == 0:
        return np.empty(0, dtype=np.float64)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=pipeline.thread_limit):
        if pipeline.model_name == "IF":
            scores = -pipeline.estimator.score_samples(transformed)
        else:
            classes = list(pipeline.estimator.classes_)
            scores = pipeline.estimator.predict_proba(transformed)[:, classes.index(1)]
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(transformed),) or not np.isfinite(scores).all():
        raise RuntimeError("Estimator returned invalid ranking scores")
    return scores


def _estimator(name: str, config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    seed = int(config.get("seed", 20261022))
    jobs = int(config.get("n_jobs", 2))
    if jobs < 1:
        raise ValueError("n_jobs must be positive")
    if name == "LR":
        from sklearn.linear_model import LogisticRegression
        cls = LogisticRegression
        params = dict(C=1.0, class_weight="balanced", max_iter=1000,
                      solver="lbfgs", random_state=seed)
    elif name == "HGB":
        from sklearn.ensemble import HistGradientBoostingClassifier
        cls = HistGradientBoostingClassifier
        params = dict(max_iter=120, max_leaf_nodes=15, l2_regularization=1.0,
                      early_stopping=False, random_state=seed)
    else:
        raise ValueError("Reviewer protocol supports LR/HGB only")
    overrides = dict(config.get("model_params", {}))
    unknown = set(overrides) - set(params)
    if unknown:
        raise ValueError(f"Unsupported parameter overrides for {name}: {sorted(unknown)}")
    params.update(overrides)
    # Evaluation-based stopping and changes to weighting would break this contract.
    if name == "HGB" and params["early_stopping"] is not False:
        raise ValueError("HGB early_stopping must remain False")
    if name in ("LR", "RF") and params["class_weight"] != "balanced":
        raise ValueError("Supervised class_weight must remain balanced")
    return cls(**params), params


def fit_predict(
    train_X: Any, train_y: Any, eval_X: Any, model_name: str,
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any], FittedRankingPipeline]:
    """Fit one fixed configuration on training rows, then score evaluation rows.

    config supports seed, n_jobs, feature_names (arrays), and model_params with
    explicit overrides of documented defaults. Parameter selection is external;
    this function never inspects evaluation labels or performs model selection.
    """
    config = dict(config or {})
    name = _canonical_model_name(model_name)
    train, names = _matrix_and_names(train_X, config.get("feature_names"))
    if len(train) == 0:
        raise ValueError("Training matrix must contain observations")
    labels = _labels(train_y, len(train))
    evaluation = _ordered_matrix(eval_X, names)
    if np.shares_memory(train, evaluation):
        raise ValueError("Training and evaluation matrices must not overlap in memory")
    counts = {str(label): int(np.count_nonzero(labels == label)) for label in (0, 1)}
    if name != "IF" and min(counts.values()) == 0:
        raise ValueError("Supervised training requires both binary classes")
    fit_mask = labels == 0 if name == "IF" else np.ones(len(labels), dtype=bool)
    if not fit_mask.any():
        raise ValueError("Isolation Forest requires benign training observations")
    fitting = train[fit_mask].copy()
    fitting[~np.isfinite(fitting)] = np.nan
    kept, dropped = [], []
    for j, feature in enumerate(names):
        observed = fitting[:, j]
        observed = observed[np.isfinite(observed)]
        if not len(observed):
            dropped.append({"feature": feature, "reason": "all_missing_in_training"})
        elif float(observed.min()) == float(observed.max()):
            dropped.append({"feature": feature, "reason": "constant_in_training"})
        else:
            kept.append(j)
    if not kept:
        raise ValueError("No nonconstant finite training features remain")
    kept_indices = np.asarray(kept, dtype=np.int64)
    selected = fitting[:, kept_indices].copy()
    medians = np.nanmedian(selected, axis=0)
    bad_rows, bad_cols = np.where(~np.isfinite(selected))
    selected[bad_rows, bad_cols] = medians[bad_cols]
    scaler = None
    if name == "LR":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(selected)
        selected = scaler.transform(selected)
    estimator, params = _estimator(name, config)
    thread_limit = int(config.get("n_jobs", 2))
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=thread_limit):
        if name in ("HGB", "XGB"):
            weights = np.asarray([len(labels) / (2.0 * counts[str(int(y))]) for y in labels])
            estimator.fit(selected, labels, sample_weight=weights)
        elif name == "IF":
            estimator.fit(selected)
        else:
            estimator.fit(selected, labels)
    pipeline = FittedRankingPipeline(name, names, kept_indices, medians, estimator, scaler, thread_limit)
    scores = predict_scores(pipeline, eval_X)
    metadata = {
        "model_name": name,
        "estimator_class": f"{type(estimator).__module__}.{type(estimator).__name__}",
        "model_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parameters": params,
        "seed": int(params["random_state"]),
        "thread_limit": thread_limit,
        "training_rows": int(len(train)),
        "fitting_rows": int(len(selected)),
        "preprocessing_fit_rows": int(len(selected)),
        "evaluation_rows": int(len(evaluation)),
        "class_counts": counts,
        "fit_class_counts": {"0": counts["0"], "1": 0 if name == "IF" else counts["1"]},
        "input_features": list(names),
        "retained_features": [names[j] for j in kept],
        "dropped_features": dropped,
        "training_medians": {names[j]: float(v) for j, v in zip(kept, medians)},
        "preprocessing": "train-only finite median imputation; train-only constant/all-missing filtering",
        "scaling": "train-only StandardScaler" if scaler is not None else "none",
        "weighting": "benign-only" if name == "IF" else "inverse-training-class-frequency balanced",
        "score_type": "raw_negative_score_samples" if name == "IF" else "uncalibrated_balanced_class_probability",
        "score_higher_is_riskier": True,
        "probability_calibrated": False,
        "evaluation_labels_used": False,
        "library_versions": _versions(),
    }
    return scores, metadata, pipeline
