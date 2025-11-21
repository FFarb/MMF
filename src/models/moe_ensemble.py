"""
Mixture-of-Experts ensemble that adapts routing based on market regimes.

Instead of relying on a single classifier, the model blends three specialists
whose strengths align with different regimes: Trend, Range, and Stress. A
lightweight gating network consumes chaos metrics (Hurst, Entropy, Volatility)
and emits softmax weights that determine how much influence each expert has per
observation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def _as_dataframe(X: pd.DataFrame | np.ndarray | Sequence[Sequence[float]]) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    array = np.asarray(X)
    if array.ndim != 2:
        raise ValueError("Input feature matrix must be 2-dimensional.")
    columns = [f"feature_{idx}" for idx in range(array.shape[1])]
    return pd.DataFrame(array, columns=columns)


def _ensure_columns(df: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required physics columns: {missing}")
    return list(columns)


def _derive_regime_targets(physics_matrix: np.ndarray) -> np.ndarray:
    hurst = physics_matrix[:, 0]
    entropy = physics_matrix[:, 1]
    volatility = physics_matrix[:, 2]
    labels = np.zeros(hurst.shape[0], dtype=int)
    stress_mask = (entropy > 0.85) | (volatility > np.median(volatility) + np.std(volatility))
    range_mask = (~stress_mask) & (hurst <= 0.55)
    labels[range_mask] = 1  # Range
    labels[stress_mask] = 2  # Stress
    return labels


@dataclass
class MixtureOfExpertsEnsemble(BaseEstimator, ClassifierMixin):
    """
    Adaptive ensemble that routes samples to experts based on chaos metrics.

    Parameters
    ----------
    physics_features : Sequence[str], optional
        Column names supplying [Hurst, Entropy, Volatility] for the gating net.
    random_state : int, optional
        Seed for all stochastic components.
    """

    physics_features: Sequence[str] = field(
        default_factory=lambda: ("hurst_200", "entropy_200", "volatility_200")
    )
    random_state: int = 42

    def __post_init__(self) -> None:
        self.trend_expert = GradientBoostingClassifier(
            learning_rate=0.05,
            n_estimators=300,
            max_depth=3,
            random_state=self.random_state,
        )
        self.range_expert = KNeighborsClassifier(n_neighbors=15, weights="distance")
        self.stress_expert = LogisticRegression(
            class_weight={0: 2.0, 1: 1.0},
            max_iter=500,
            solver="lbfgs",
        )
        self.gating_network = MLPClassifier(
            hidden_layer_sizes=(8,),
            activation="tanh",
            alpha=1e-3,
            max_iter=500,
            random_state=self.random_state,
        )
        self.feature_scaler = StandardScaler()
        self.physics_scaler = StandardScaler()
        self._fitted = False

    def fit(
        self,
        X: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
        y: Iterable[int | float],
    ) -> "MixtureOfExpertsEnsemble":
        """
        Fit the experts and gating network.
        """
        df = _as_dataframe(X)
        physics_cols = _ensure_columns(df, self.physics_features)

        numeric_df = df.select_dtypes(include=["number"])
        if numeric_df.empty:
            raise ValueError("MixtureOfExpertsEnsemble requires numeric features.")

        self.feature_columns_ = list(numeric_df.columns)
        features = numeric_df.to_numpy(dtype=float)
        scaled_features = self.feature_scaler.fit_transform(features)

        y_array = np.ravel(np.asarray(y))
        if y_array.size != scaled_features.shape[0]:
            raise ValueError("Mismatched samples between X and y.")

        # Train specialists
        self.trend_expert.fit(scaled_features, y_array)
        self.range_expert.fit(scaled_features, y_array)
        self.stress_expert.fit(scaled_features, y_array)

        # Train gating network on heuristic regimes
        physics_matrix = df.loc[:, physics_cols].to_numpy(dtype=float)
        scaled_physics = self.physics_scaler.fit_transform(physics_matrix)
        regime_labels = _derive_regime_targets(physics_matrix)
        self.gating_network.fit(scaled_physics, regime_labels)
        self._gate_classes_ = list(self.gating_network.classes_)

        self._fitted = True
        return self

    def _check_is_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("MixtureOfExpertsEnsemble must be fitted before use.")

    def _expert_probabilities(self, scaled_features: np.ndarray) -> List[np.ndarray]:
        trend_probs = self.trend_expert.predict_proba(scaled_features)
        range_probs = self.range_expert.predict_proba(scaled_features)
        stress_probs = self.stress_expert.predict_proba(scaled_features)
        return [trend_probs, range_probs, stress_probs]

    def _gating_weights(self, physics_matrix: np.ndarray) -> np.ndarray:
        scaled = self.physics_scaler.transform(physics_matrix)
        raw = self.gating_network.predict_proba(scaled)
        weights = np.full((scaled.shape[0], 3), 1.0 / 3.0, dtype=float)
        for idx, cls in enumerate(self._gate_classes_):
            if cls < weights.shape[1]:
                weights[:, cls] = raw[:, idx]
        weights /= weights.sum(axis=1, keepdims=True)
        return weights

    def predict_proba(
        self, X: pd.DataFrame | np.ndarray | Sequence[Sequence[float]]
    ) -> np.ndarray:
        """
        Predict blended class probabilities.
        """
        self._check_is_fitted()
        df = _as_dataframe(X)
        numeric_df = df[self.feature_columns_]
        features = numeric_df.to_numpy(dtype=float)
        scaled_features = self.feature_scaler.transform(features)

        physics_cols = _ensure_columns(df, self.physics_features)
        physics_matrix = df.loc[:, physics_cols].to_numpy(dtype=float)
        weights = self._gating_weights(physics_matrix)
        expert_probabilities = self._expert_probabilities(scaled_features)

        blended = np.zeros_like(expert_probabilities[0])
        for weight_idx, probs in enumerate(expert_probabilities):
            blended += weights[:, [weight_idx]] * probs
        return blended

    def predict(
        self, X: pd.DataFrame | np.ndarray | Sequence[Sequence[float]]
    ) -> np.ndarray:
        """
        Predict classes based on weighted probabilities.
        """
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)


__all__ = ["MixtureOfExpertsEnsemble"]
