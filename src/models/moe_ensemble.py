"""
Bicameral Hybrid Ensemble: Neuro-Symbolic Trading System.
Updated with Telemetry & Sample Weighting support.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Optional, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .cnn_temporal import CNNExpert
from .deep_experts import TorchSklearnWrapper
from ..config import (
    NUM_ASSETS,
    CNN_ARTIFACTS_DIR,
    CNN_BATCH_SIZE,
    CNN_C_MID,
    CNN_DROPOUT,
    CNN_EPOCHS,
    CNN_FILL_EARLY,
    CNN_HIDDEN,
    CNN_LATENT_PREFIX,
    CNN_LR,
    CNN_RANDOM_STATE,
    CNN_USE,
    CNN_WINDOW_L,
)

logger = logging.getLogger(__name__)


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
    fractal_dim = physics_matrix[:, 2]
    fractal_dim = np.nan_to_num(fractal_dim, nan=np.nanmean(fractal_dim))
    entropy = np.nan_to_num(entropy, nan=np.nanmean(entropy))
    labels = np.zeros(hurst.shape[0], dtype=int)
    high_fdi = np.nanpercentile(fractal_dim, 75)
    stress_mask = (entropy > 0.9) | (fractal_dim > high_fdi)
    range_mask = (~stress_mask) & (hurst <= 0.55)
    cnn_mask = (~stress_mask) & (~range_mask) & ((entropy >= 0.6) & (entropy <= 0.85))
    labels[range_mask] = 1  # Range
    labels[stress_mask] = 2  # Stress
    labels[cnn_mask] = 3    # CNN-preferring ambiguous churn
    return labels


@dataclass
class HybridTrendExpert(BaseEstimator, ClassifierMixin):
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 3
    random_state: int = 42
    n_assets: int = 1
    
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
        
        # Init Visionary with Global Context
        if self.visionary is None:
            # We assume X_scaled is (Batch * Assets, Sequence_Length * Features)
            # We need to derive n_features per asset.
            seq_len = 16 
            
            total_features = X_scaled.shape[1]
            if total_features % seq_len != 0:
                 # Fallback
                 n_features = total_features
            else:
                 n_features = total_features // seq_len
            
            self.visionary = TorchSklearnWrapper(
                n_features=n_features,
                n_assets=self.n_assets,
                sequence_length=seq_len,
                random_state=self.random_state
            )
        
        # Train Analyst (Supports weights) - Analyst sees each row independently
        self.analyst.fit(X_scaled, y_array, sample_weight=sample_weight)
        
        # Train Visionary (Neural Part) - Needs Global Context
        # We pass X_scaled directly. TorchSklearnWrapper.fit will handle reshaping if possible.
        # If X_scaled is just a pile of rows, reshaping might be wrong if order isn't preserved.
        # We assume the caller (run_deep_research.py) provides data in correct order:
        # Time-major or Asset-major blocks.
        # Given GlobalMarketDataset yields (Time, Assets, Features), if we flatten it,
        # it becomes (Time * Assets, Features).
        # TorchSklearnWrapper needs to reconstruct (Time, Assets, Features).
        # It needs to know Sequence_Length.
        
        self.visionary.fit(X_scaled, y_array)
        
        # Train Meta (Supports weights)
        p1 = self.analyst.predict_proba(X_scaled)[:, 1]
        
        # Visionary predict_proba returns (Samples, 2)
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
    physics_features: Sequence[str] = field(
        default_factory=lambda: (
            "hurst_200",
            "entropy_200",
            "fdi_200",
            "stability_theta",
            "stability_acf",
        )
    )
    random_state: int = 42
    trend_estimators: int = 300
    gating_epochs: int = 500
    use_cnn: bool = CNN_USE
    cnn_window: int = CNN_WINDOW_L
    cnn_mid_channels: int = CNN_C_MID
    cnn_hidden_dim: int = CNN_HIDDEN
    cnn_dropout: float = CNN_DROPOUT
    cnn_lr: float = CNN_LR
    cnn_epochs: int = CNN_EPOCHS
    cnn_batch_size: int = CNN_BATCH_SIZE
    cnn_random_state: int = CNN_RANDOM_STATE
    cnn_fill_strategy: str = CNN_FILL_EARLY
    cnn_artifacts_dir: Optional[Path | str] = CNN_ARTIFACTS_DIR
    cnn_latent_prefix: str = CNN_LATENT_PREFIX
    cnn_params: Optional[Dict[str, Any]] = None
    n_assets: int = 1

    def __post_init__(self) -> None:
        self.trend_expert = HybridTrendExpert(
            n_estimators=self.trend_estimators, 
            random_state=self.random_state,
            n_assets=self.n_assets
        )
        self.range_expert = KNeighborsClassifier(n_neighbors=15, weights="distance")
        self.stress_expert = LogisticRegression(class_weight={0: 2.0, 1: 1.0}, max_iter=500)
        self.gating_network = MLPClassifier(
            hidden_layer_sizes=(8,),
            activation="tanh",
            max_iter=self.gating_epochs,
            random_state=self.random_state,
        )
        self.feature_scaler = StandardScaler()
        self.physics_scaler = StandardScaler()
        self._fitted = False
        self.cnn_expert: Optional[CNNExpert] = None
        self.cnn_feature_columns_: List[str] = []
        self._cnn_enabled = False
        self._last_cnn_stats: Dict[str, float] = {}
        if isinstance(self.cnn_artifacts_dir, (str, Path)):
            self.cnn_artifacts_dir = Path(self.cnn_artifacts_dir)
        else:
            self.cnn_artifacts_dir = None

    def fit(self, X, y, sample_weight=None) -> "MixtureOfExpertsEnsemble":
        """Fit with sample weights support."""
        df = _as_dataframe(X)
        df = df.copy()
        self.cnn_feature_columns_ = [col for col in df.columns if col.startswith(self.cnn_latent_prefix)]
        base_df = df.drop(columns=self.cnn_feature_columns_, errors="ignore")
        physics_cols = _ensure_columns(base_df, self.physics_features)
        
        # Prepare Features
        numeric_df = base_df.select_dtypes(include=["number"])
        self.feature_columns_ = list(numeric_df.columns)
        X_scaled = self.feature_scaler.fit_transform(numeric_df.to_numpy(dtype=float))
        y_array = np.ravel(np.asarray(y))

        # Physics-Aware Sample Weighting
        trend_sample_weight = sample_weight
        if "stability_warning" in base_df.columns:
            print("    [MoE] Applying Physics-Guided Sample Weighting...")
            warnings = base_df["stability_warning"].values
            # Create weights: 1.0 for stable, 0.1 for unstable
            physics_weights = np.ones(len(base_df))
            physics_weights[warnings == 1] = 0.1
            
            if sample_weight is not None:
                trend_sample_weight = sample_weight * physics_weights
            else:
                trend_sample_weight = physics_weights

        # Train Experts (Pass weights where supported)
        print("    [MoE] Training Experts with Smart Weights...")
        self.trend_expert.fit(base_df, y_array, sample_weight=trend_sample_weight)
        self.range_expert.fit(X_scaled, y_array) # KNN doesn't support fit weights usually
        self.stress_expert.fit(X_scaled, y_array, sample_weight=sample_weight)

        # Temporal CNN Expert
        self._train_cnn_expert(df, y_array)

        # Train Gating
        print("    [MoE] Training Gating Network...")
        physics_matrix = base_df.loc[:, physics_cols].to_numpy(dtype=float)
        scaled_physics = self.physics_scaler.fit_transform(physics_matrix)
        regime_labels = _derive_regime_targets(physics_matrix)
        self.gating_network.fit(scaled_physics, regime_labels) # MLP doesn't support sample_weight standardly

        self._gate_classes_ = list(self.gating_network.classes_)
        self._fitted = True
        return self

    def _train_cnn_expert(self, df: pd.DataFrame, y_array: np.ndarray) -> None:
        """Fit the temporal CNN expert if enabled."""
        self._cnn_enabled = False
        if not self.use_cnn:
            logger.info("CNNExpert disabled via configuration.")
            self.cnn_expert = None
            return
        if not self.cnn_feature_columns_:
            logger.info("CNNExpert skipped because no temporal latent columns were provided.")
            self.cnn_expert = None
            return
            
        # Use tuned params if available, otherwise defaults
        params = self.cnn_params or {}
        
        y_series = pd.Series(y_array, index=df.index)
        cnn_df = df[self.cnn_feature_columns_].copy()
        
        self.cnn_expert = CNNExpert(
            window_length=params.get("window_length", self.cnn_window),
            mid_channels=params.get("mid_channels", self.cnn_mid_channels),
            hidden_dim=params.get("hidden_dim", self.cnn_hidden_dim),
            dropout=params.get("dropout", self.cnn_dropout),
            lr=params.get("lr", self.cnn_lr),
            weight_decay=params.get("weight_decay", 1e-2),
            epochs=params.get("epochs", self.cnn_epochs),
            batch_size=params.get("batch_size", self.cnn_batch_size),
            random_state=self.cnn_random_state,
            fill_strategy=self.cnn_fill_strategy,
            artifacts_path=self.cnn_artifacts_dir,
            dilations=params.get("dilations", (1, 2, 4, 8, 16, 32)),
            kernel_size=params.get("kernel_size", 3),
        )
        try:
            self.cnn_expert.fit(cnn_df, y_series)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("CNNExpert training failed (%s). Falling back to 3-way MoE.", exc)
            self.cnn_expert = None
            return
        self._cnn_enabled = True
        logger.info(
            "CNNExpert trained with %s features (window=%s).",
            cnn_df.shape[1],
            self.cnn_expert.window_length,
        )

    def _gating_weights(self, physics_matrix: np.ndarray) -> np.ndarray:
        scaled = self.physics_scaler.transform(physics_matrix)
        raw = self.gating_network.predict_proba(scaled)
        n_outputs = 4 if self._cnn_enabled else 3
        weights = np.full((scaled.shape[0], n_outputs), 1.0 / float(n_outputs), dtype=float)
        for idx, cls in enumerate(self._gate_classes_):
            if cls < weights.shape[1]:
                weights[:, cls] = raw[:, idx]
        weights = np.clip(weights, 1e-6, None)
        weights /= weights.sum(axis=1, keepdims=True)

        # Hard Gating: Physics Override
        # If stability_theta is critically low (< 0.005), boost Stress Expert (index 2)
        try:
            theta_idx = list(self.physics_features).index("stability_theta")
            theta_vals = physics_matrix[:, theta_idx]
            
            # Identify critical slowing down (theta -> 0)
            critical_mask = theta_vals < 0.005
            
            if np.any(critical_mask):
                # Boost Stress Expert weight
                # We add a fixed amount to stress weight and re-normalize
                boost_amount = 2.0  # Strong boost
                
                # Ensure we don't crash if index 2 (Stress) is not available (unlikely)
                if weights.shape[1] > 2:
                    weights[critical_mask, 2] += boost_amount
                    weights[critical_mask] /= weights[critical_mask].sum(axis=1, keepdims=True)
        except ValueError:
            # stability_theta not in physics_features
            pass

        if not self._cnn_enabled and weights.shape[1] > 3:
            base = weights[:, :3]
            base /= base.sum(axis=1, keepdims=True)
            return base
        return weights

    def predict_proba(self, X) -> np.ndarray:
        # Same as before
        if not self._fitted: raise RuntimeError("Not fitted")
        df = _as_dataframe(X)
        df = df.copy()
        if self._cnn_enabled:
            for col in self.cnn_feature_columns_:
                if col not in df.columns:
                    df[col] = 0.0
        base_df = df.drop(columns=self.cnn_feature_columns_, errors="ignore")
        for col in self.feature_columns_:
            if col not in base_df.columns:
                base_df[col] = 0.0
        numeric = base_df[self.feature_columns_]
        X_scaled = self.feature_scaler.transform(numeric.to_numpy(dtype=float))
        physics_matrix = base_df.loc[:, _ensure_columns(base_df, self.physics_features)].to_numpy(dtype=float)
        
        weights = self._gating_weights(physics_matrix)
        
        p1 = self.trend_expert.predict_proba(base_df) # Hybrid handles scaling internally for now to match fit
        p2 = self.range_expert.predict_proba(X_scaled)
        p3 = self.stress_expert.predict_proba(X_scaled)
        base_weights = weights[:, :3]
        base_weights = base_weights / base_weights.sum(axis=1, keepdims=True)
        base_only = (
            base_weights[:, [0]] * p1 + base_weights[:, [1]] * p2 + base_weights[:, [2]] * p3
        )
        blended = weights[:, [0]] * p1 + weights[:, [1]] * p2 + weights[:, [2]] * p3

        self._last_cnn_stats = {"weight_mean": 0.0, "delta_mean": 0.0, "delta_std": 0.0}
        if self._cnn_enabled and self.cnn_expert is not None and self.cnn_feature_columns_:
            cnn_df = df[self.cnn_feature_columns_]
            cnn_probs = self.cnn_expert.predict_proba(cnn_df)
            fallback = np.column_stack([1.0 - base_only[:, 1], base_only[:, 1]])
            invalid = ~np.isfinite(cnn_probs[:, 1])
            if invalid.any():
                cnn_probs[invalid] = fallback[invalid]
            blended += weights[:, [3]] * cnn_probs
            delta = blended[:, 1] - base_only[:, 1]
            self._last_cnn_stats = {
                "weight_mean": float(np.mean(weights[:, 3])),
                "delta_mean": float(np.mean(delta)),
                "delta_std": float(np.std(delta)),
            }
        return blended

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def get_expert_telemetry(self, X: pd.DataFrame) -> Dict[str, float]:
        """New Telemetry Method: Returns expert activation stats."""
        if not self._fitted:
            return {}
        df = _as_dataframe(X).copy()
        base_df = df.drop(columns=self.cnn_feature_columns_, errors="ignore")
        physics_cols = _ensure_columns(base_df, self.physics_features)
        physics_matrix = base_df.loc[:, physics_cols].to_numpy(dtype=float)
        weights = self._gating_weights(physics_matrix)
        
        activation = weights.mean(axis=0)
        confidence = np.max(weights, axis=1).mean()
        share_cnn = float(activation[3]) if activation.shape[0] > 3 else 0.0
        telemetry = {
            "share_trend": float(activation[0]),
            "share_range": float(activation[1]),
            "share_stress": float(activation[2]),
            "share_cnn": share_cnn,
            "gating_confidence": float(confidence),
            "cnn_weight_mean": float(self._last_cnn_stats.get("weight_mean", share_cnn)),
            "cnn_delta_mean": float(self._last_cnn_stats.get("delta_mean", 0.0)),
            "cnn_delta_std": float(self._last_cnn_stats.get("delta_std", 0.0)),
        }
        
        return telemetry
