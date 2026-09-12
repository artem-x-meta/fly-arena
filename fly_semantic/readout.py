"""Regularized linear multilabel readout trained on neural features only.

The caller owns whole-episode train/validation/test splits. ``fit`` accepts
training rows only; neither validation labels nor organism state are inference
arguments. Sigmoid outputs are model scores, not calibrated probabilities.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


def sigmoid(values: Any) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_negative = np.exp(values[~positive])
    result[~positive] = exp_negative / (1.0 + exp_negative)
    return result


def _array(values: Any, name: str, ndim: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != ndim or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return result


class LinearReadout:
    FORMAT = "fly_semantic.linear_multilabel.v1"

    def __init__(self, *, feature_digest: str, concept_ids: Any, mean: Any, scale: Any, weights: Any, bias: Any, seed: int = 42, diagnostics: dict | None = None):
        if not isinstance(feature_digest, str) or not feature_digest:
            raise ValueError("feature_digest must bind a nonempty feature manifest")
        raw_ids = np.asarray(concept_ids)
        if raw_ids.ndim != 1 or not raw_ids.size or raw_ids.dtype.kind not in "iu" or np.any(raw_ids <= 0) or np.unique(raw_ids).size != raw_ids.size:
            raise ValueError("concept_ids must be unique positive integer IDs")
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.feature_digest = feature_digest
        self.concept_ids = raw_ids.astype(np.int64, copy=True)
        self.mean = _array(mean, "mean", 1).copy()
        self.scale = _array(scale, "scale", 1).copy()
        self.weights = _array(weights, "weights", 2).copy()
        self.bias = _array(bias, "bias", 1).copy()
        self.n_features = int(self.mean.size)
        self.n_outputs = int(self.concept_ids.size)
        if not 1 <= self.n_features <= 1024 or self.scale.shape != self.mean.shape or np.any(self.scale <= 0) or self.weights.shape != (self.n_features, self.n_outputs) or self.bias.shape != (self.n_outputs,):
            raise ValueError("inconsistent readout dimensions or scaler")
        self.seed = int(seed)
        self.diagnostics = dict(diagnostics or {})
        json.dumps(self.diagnostics, allow_nan=False)
        for value in (self.concept_ids, self.mean, self.scale, self.weights, self.bias):
            value.flags.writeable = False

    @classmethod
    def fit(cls, X_train: Any, y_train: Any, *, feature_digest: str, concept_ids: Any = (1,), l2: float = 1e-3, max_iter: int = 500, seed: int = 42, tolerance: float = 1e-8) -> "LinearReadout":
        """Fit train-only mean/std and BCE + l2/2 * ||W||² using L-BFGS.

        Supported labels are independent columns; unsupported concepts must be
        omitted explicitly by the caller. Zero initialization is deterministic
        and does not consume any simulator RNG. Seed is recorded as provenance.
        """
        from scipy.optimize import minimize

        X = _array(X_train, "X_train", 2)
        y = np.asarray(y_train, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]
        y = _array(y, "y_train", 2)
        if X.shape[0] < 2 or not 1 <= X.shape[1] <= 1024 or y.shape[0] != X.shape[0] or y.shape[1] < 1 or np.any((y != 0) & (y != 1)):
            raise ValueError("training needs >=2 aligned rows and binary supported labels")
        if not np.isscalar(l2) or not np.isfinite(l2) or l2 < 0 or not np.isscalar(tolerance) or not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("l2 must be finite/nonnegative and tolerance finite/positive")
        if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        mean = X.mean(axis=0)
        scale = X.std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        normalized = (X - mean) / scale
        if not np.all(np.isfinite(normalized)):
            raise ValueError("training scaler overflowed")
        n, d = X.shape
        k = y.shape[1]
        initial = np.zeros((d + 1) * k, dtype=np.float64)
        prevalence = (y.sum(axis=0) + 0.5) / (n + 1.0)
        initial[d * k:] = np.log(prevalence / (1 - prevalence))

        def objective(parameters):
            weights = parameters[:d * k].reshape(d, k)
            bias = parameters[d * k:]
            logits = normalized @ weights + bias
            loss = np.mean(np.logaddexp(0.0, logits) - y * logits) + 0.5 * l2 * np.sum(weights * weights)
            residual = (sigmoid(logits) - y) / (n * k)
            grad_weights = normalized.T @ residual + l2 * weights
            gradient = np.concatenate((grad_weights.reshape(-1), residual.sum(axis=0)))
            return float(loss), gradient

        result = minimize(objective, initial, jac=True, method="L-BFGS-B", options={"maxiter": int(max_iter), "ftol": float(tolerance), "gtol": float(tolerance), "maxls": 40})
        final_loss, final_gradient = objective(result.x)
        if not np.isfinite(final_loss) or not np.all(np.isfinite(result.x)):
            raise RuntimeError("linear readout optimization produced nonfinite parameters")
        diagnostics = {
            "optimizer": "scipy.optimize.L-BFGS-B",
            "converged": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(result.nit),
            "objective": final_loss,
            "gradient_inf_norm": float(np.max(np.abs(final_gradient))),
            "train_rows": int(n),
            "feature_count": int(d),
            "positive_train_rows": y.sum(axis=0).astype(int).tolist(),
            "constant_label_columns": np.flatnonzero((y.sum(axis=0) == 0) | (y.sum(axis=0) == n)).tolist(),
            "l2": float(l2),
            "max_iter": int(max_iter),
            "tolerance": float(tolerance),
            "initialization": "zero_weights_smoothed_training_prevalence_bias",
            "scaler_fit": "training_rows_only",
            "score_calibration": "not_calibrated",
        }
        return cls(feature_digest=feature_digest, concept_ids=concept_ids, mean=mean, scale=scale, weights=result.x[:d * k].reshape(d, k), bias=result.x[d * k:], seed=seed, diagnostics=diagnostics)

    def predict_scores(self, neural_features: Any) -> np.ndarray:
        """Accept one vector or a batch; return independent concept scores."""
        features = np.asarray(neural_features, dtype=np.float64)
        if features.ndim not in (1, 2) or features.shape[-1] != self.n_features or not np.all(np.isfinite(features)):
            raise ValueError("inference accepts finite neural feature vectors with the trained dimension")
        logits = ((features - self.mean) / self.scale) @ self.weights + self.bias
        if not np.all(np.isfinite(logits)):
            raise ValueError("readout inference overflowed")
        return sigmoid(logits)

    def save(self, path: str | Path) -> None:
        """Atomically write numeric arrays and JSON metadata; never pickle."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = json.dumps({"format": self.FORMAT, "feature_digest": self.feature_digest, "seed": self.seed, "diagnostics": self.diagnostics}, sort_keys=True, allow_nan=False)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + ".", suffix=".tmp", delete=False) as stream:
                temporary = stream.name
                np.savez_compressed(stream, metadata=np.asarray(metadata), concept_ids=self.concept_ids, mean=self.mean, scale=self.scale, weights=self.weights, bias=self.bias)
            os.replace(temporary, destination)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path: str | Path, *, expected_feature_digest: str) -> "LinearReadout":
        if not isinstance(expected_feature_digest, str) or not expected_feature_digest:
            raise ValueError("loading requires the current feature identity digest")
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {"metadata", "concept_ids", "mean", "scale", "weights", "bias"}:
                raise ValueError("malformed readout checkpoint")
            metadata = json.loads(str(data["metadata"].item()))
            if not isinstance(metadata, dict) or set(metadata) != {"format", "feature_digest", "seed", "diagnostics"} or metadata["format"] != cls.FORMAT or metadata["feature_digest"] != expected_feature_digest:
                raise ValueError("readout checkpoint format or feature identity mismatch")
            return cls(feature_digest=metadata["feature_digest"], concept_ids=data["concept_ids"], mean=data["mean"], scale=data["scale"], weights=data["weights"], bias=data["bias"], seed=metadata["seed"], diagnostics=metadata["diagnostics"])
