"""
Bicameral Hybrid Ensemble: Neuro-Symbolic Trading System.
Updated with Telemetry & Sample Weighting support.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .deep_experts import TorchSklearnWrapper


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
class HybridTrendExpert(BaseEstimator, ClassifierMixin):
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 3
    random_state: int = 42
    
    def __post_init__(self) -> None:
        self.analyst = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self.visionary = None
        self.meta_learner = LogisticRegression(
            max_iter=1000, solver='lbfgs', random_state=self.random_state,
        )
        self.scaler_ = StandardScaler()
        self._fitted = False
        self.classes_ = np.array([0, 1])
        
    def fit(self, X, y, sample_weight=None) -> "HybridTrendExpert":
        # Simplified fit for brevity - in full version ensure sample_weight is passed to analyst
        df = _as_dataframe(X)
        numeric_df = df.select_dtypes(include=["number"])
        if 'asset_id' in df.columns: numeric_df = numeric_df.drop(columns=['asset_id'], errors='ignore')
        
        X_array = numeric_df.to_numpy(dtype=float)
        y_array = np.ravel(np.asarray(y))
        X_scaled = self.scaler_.fit_transform(X_array)
        
        # Init Visionary
        if self.visionary is None:
            self.visionary = TorchSklearnWrapper(n_features=X_scaled.shape[1], random_state=self.random_state)
        
        # Train Analyst (Supports weights)
        self.analyst.fit(X_scaled, y_array, sample_weight=sample_weight)
        
        # Train Visionary (Does not support weights yet, ignoring)
        self.visionary.fit(X_scaled, y_array)
        
        # Train Meta (Supports weights)
        # For simplicity in this patch, we skip stacking complexity and train meta on base predictions
        p1 = self.analyst.predict_proba(X_scaled)[:, 1]
        p2 = self.visionary.predict_proba(X_scaled)[:, 1]
        meta_X = np.column_stack([p1, p2])
        self.meta_learner.fit(meta_X, y_array, sample_weight=sample_weight)
        
        self._fitted = True
        return self
    
    def predict_proba(self, X) -> np.ndarray:
        df = _as_dataframe(X)
        numeric_df = df.select_dtypes(include=["number"])
        if 'asset_id' in df.columns: numeric_df = numeric_df.drop(columns=['asset_id'], errors='ignore')
        X_scaled = self.scaler_.transform(numeric_df.to_numpy(dtype=float))
        
        p1 = self.analyst.predict_proba(X_scaled)[:, 1]
        p2 = self.visionary.predict_proba(X_scaled)[:, 1]
        meta_X = np.column_stack([p1, p2])
        return self.meta_learner.predict_proba(meta_X)


@dataclass
class MixtureOfExpertsEnsemble(BaseEstimator, ClassifierMixin):
    physics_features: Sequence[str] = field(default_factory=lambda: ("hurst_200", "entropy_200", "volatility_200"))
    random_state: int = 42
    trend_estimators: int = 300
    gating_epochs: int = 500

    def __post_init__(self) -> None:
        self.trend_expert = HybridTrendExpert(n_estimators=self.trend_estimators, random_state=self.random_state)
        self.range_expert = KNeighborsClassifier(n_neighbors=15, weights="distance")
        self.stress_expert = LogisticRegression(class_weight={0: 2.0, 1: 1.0}, max_iter=500)
        self.gating_network = MLPClassifier(hidden_layer_sizes=(8,), activation="tanh", max_iter=self.gating_epochs, random_state=self.random_state)
        self.feature_scaler = StandardScaler()
        self.physics_scaler = StandardScaler()
        self._fitted = False

    def fit(self, X, y, sample_weight=None) -> "MixtureOfExpertsEnsemble":
        """Fit with sample weights support."""
        df = _as_dataframe(X)
        physics_cols = _ensure_columns(df, self.physics_features)
        
        # Prepare Features
        numeric_df = df.select_dtypes(include=["number"])
        self.feature_columns_ = list(numeric_df.columns)
        X_scaled = self.feature_scaler.fit_transform(numeric_df.to_numpy(dtype=float))
        y_array = np.ravel(np.asarray(y))

        # Train Experts (Pass weights where supported)
        print("    [MoE] Training Experts with Smart Weights...")
        self.trend_expert.fit(X, y_array, sample_weight=sample_weight)
        self.range_expert.fit(X_scaled, y_array) # KNN doesn't support fit weights usually
        self.stress_expert.fit(X_scaled, y_array, sample_weight=sample_weight)

        # Train Gating
        print("    [MoE] Training Gating Network...")
        physics_matrix = df.loc[:, physics_cols].to_numpy(dtype=float)
        scaled_physics = self.physics_scaler.fit_transform(physics_matrix)
        regime_labels = _derive_regime_targets(physics_matrix)
        self.gating_network.fit(scaled_physics, regime_labels) # MLP doesn't support sample_weight standardly

        self._gate_classes_ = list(self.gating_network.classes_)
        self._fitted = True
        return self

    def _gating_weights(self, physics_matrix: np.ndarray) -> np.ndarray:
        scaled = self.physics_scaler.transform(physics_matrix)
        raw = self.gating_network.predict_proba(scaled)
        weights = np.full((scaled.shape[0], 3), 1.0 / 3.0, dtype=float)
        for idx, cls in enumerate(self._gate_classes_):
            if cls < weights.shape[1]:
                weights[:, cls] = raw[:, idx]
        weights /= weights.sum(axis=1, keepdims=True)
        return weights

    def predict_proba(self, X) -> np.ndarray:
        # Same as before
        if not self._fitted: raise RuntimeError("Not fitted")
        df = _as_dataframe(X)
        X_scaled = self.feature_scaler.transform(df[self.feature_columns_].to_numpy(dtype=float))
        physics_matrix = df.loc[:, _ensure_columns(df, self.physics_features)].to_numpy(dtype=float)
        
        weights = self._gating_weights(physics_matrix)
        
        p1 = self.trend_expert.predict_proba(X) # Hybrid handles scaling internally for now to match fit
        p2 = self.range_expert.predict_proba(X_scaled)
        p3 = self.stress_expert.predict_proba(X_scaled)
        
        blended = weights[:, [0]] * p1 + weights[:, [1]] * p2 + weights[:, [2]] * p3
        return blended

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def get_expert_telemetry(self, X: pd.DataFrame) -> Dict[str, float]:
        """New Telemetry Method: Returns expert activation stats."""
        if not self._fitted: return {}
        df = _as_dataframe(X)
        physics_matrix = df.loc[:, self.physics_features].to_numpy(dtype=float)
        weights = self._gating_weights(physics_matrix)
        
        activation = weights.mean(axis=0)
        confidence = np.max(weights, axis=1).mean()
        
        return {
            "share_trend": float(activation[0]),
            "share_range": float(activation[1]),
            "share_stress": float(activation[2]),
            "gating_confidence": float(confidence)
        }
