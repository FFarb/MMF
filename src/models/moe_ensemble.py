"""
Specialized Mixture of Experts Ensemble.

SUBTRACTION STRATEGY: Removed complex, noisy components (GraphVisionary).
Focused on proven, reliable experts with clear specializations.

Experts:
1. TrendExpert (HistGradientBoosting) - Sustainable trends
2. PatternExpert (CNNExpert) - Short-term visual patterns
3. StressExpert (LogisticRegression) - Crash protection
4. RangeExpert (KNeighborsClassifier) - Mean reversion fallback
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Optional, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .cnn_temporal import CNNExpert
from .physics_experts import OUMeanReversionExpert
from ..config import (
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
    """Convert input to DataFrame."""
    if isinstance(X, pd.DataFrame):
        return X.copy()
    array = np.asarray(X)
    if array.ndim != 2:
        raise ValueError("Input feature matrix must be 2-dimensional.")
    columns = [f"feature_{idx}" for idx in range(array.shape[1])]
    return pd.DataFrame(array, columns=columns)


def _ensure_columns(df: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    """Ensure required columns exist in DataFrame."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required physics columns: {missing}")
    return list(columns)


def _derive_regime_targets(physics_matrix: np.ndarray) -> np.ndarray:
    """
    Derive regime labels for gating network training.
    
    Uses physics features to identify market regimes:
    - 0: Trend (default)
    - 1: Range (low Hurst)
    - 2: Stress (high entropy/FDI)
    - 3: Pattern (CNN-preferring ambiguous churn)
    """
    hurst = physics_matrix[:, 0]
    entropy = physics_matrix[:, 1]
    fractal_dim = physics_matrix[:, 2]
    
    # Handle NaNs
    fractal_dim = np.nan_to_num(fractal_dim, nan=np.nanmean(fractal_dim))
    entropy = np.nan_to_num(entropy, nan=np.nanmean(entropy))
    
    # Initialize as Trend (0)
    labels = np.zeros(hurst.shape[0], dtype=int)
    
    # Identify regimes
    high_fdi = np.nanpercentile(fractal_dim, 75)
    stress_mask = (entropy > 0.9) | (fractal_dim > high_fdi)
    range_mask = (~stress_mask) & (hurst <= 0.55)
    pattern_mask = (~stress_mask) & (~range_mask) & ((entropy >= 0.6) & (entropy <= 0.85))
    
    labels[range_mask] = 1   # Range
    labels[stress_mask] = 2  # Stress
    labels[pattern_mask] = 3 # Pattern (CNN)
    
    return labels


@dataclass
class TrendExpert(BaseEstimator, ClassifierMixin):
    """
    Pure HistGradientBoosting expert for sustainable trends.
    
    SUBTRACTION: Removed GraphVisionary complexity.
    WHY: HistGBM is faster, handles NaNs natively, outperforms standard GBM.
    """
    learning_rate: float = 0.05
    max_iter: int = 500
    max_depth: int = 5
    validation_fraction: float = 0.1
    early_stopping: bool = True
    random_state: int = 42
    
    def __post_init__(self) -> None:
        """Initialize the HistGradientBoosting classifier."""
        self.model = HistGradientBoostingClassifier(
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            validation_fraction=self.validation_fraction,
            early_stopping=self.early_stopping,
            random_state=self.random_state,
        )
        self.scaler_ = StandardScaler()
        self._fitted = False
        self.classes_ = np.array([0, 1])
    
    def fit(self, X, y, sample_weight=None) -> "TrendExpert":
        """
        Fit the trend expert.
        
        Parameters
        ----------
        X : DataFrame or array
            Feature matrix
        y : array
            Target labels
        sample_weight : array, optional
            Sample weights (physics gating)
        """
        df = _as_dataframe(X)
        numeric_df = df.select_dtypes(include=["number"])
        if 'asset_id' in df.columns:
            numeric_df = numeric_df.drop(columns=['asset_id'], errors='ignore')
        
        X_array = numeric_df.to_numpy(dtype=float)
        y_array = np.ravel(np.asarray(y))
        
        # Scale features
        X_scaled = self.scaler_.fit_transform(X_array)
        
        # Train with sample weights
        self.model.fit(X_scaled, y_array, sample_weight=sample_weight)
        
        self._fitted = True
        return self
    
    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities."""
        if not self._fitted:
            raise RuntimeError("TrendExpert not fitted")
        
        df = _as_dataframe(X)
        numeric_df = df.select_dtypes(include=["number"])
        if 'asset_id' in df.columns:
            numeric_df = numeric_df.drop(columns=['asset_id'], errors='ignore')
        
        X_scaled = self.scaler_.transform(numeric_df.to_numpy(dtype=float))
        return self.model.predict_proba(X_scaled)
    
    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        return np.argmax(self.predict_proba(X), axis=1)


@dataclass
class MixtureOfExpertsEnsemble(BaseEstimator, ClassifierMixin):
    """
    Specialized Mixture of Experts with 5 orthogonal experts.
    
    PHYSICS-ENHANCED ENSEMBLE:
    - Removed: GraphVisionary (complex, unstable)
    - Added: OUMeanReversionExpert (physics-based elasticity)
    
    Experts:
    1. Trend (HistGBM) - Sustainable trends
    2. Range (KNN) - Local patterns
    3. Stress (LogReg) - Crash protection
    4. Pattern (CNN) - Temporal sequences
    5. Elastic (OU) - Mean reversion / elasticity
    """
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
    
    # Trend Expert (HistGBM) params
    trend_learning_rate: float = 0.05
    trend_max_iter: int = 500
    trend_max_depth: int = 5
    
    # Gating Network params
    gating_epochs: int = 500
    
    # OU Expert params
    use_ou: bool = True
    ou_alpha: float = 1.0
    ou_lookback: int = 100
    
    # CNN Expert params
    use_cnn: bool = CNN_USE
    cnn_source_feature: str = "frac_diff"  # NEW: Feed raw time-series to CNN
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
    cnn_latent_prefix: str = CNN_LATENT_PREFIX  # DEPRECATED: Not used anymore
    cnn_params: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        """Initialize all experts and gating network."""
        # Expert 1: Trend (HistGBM)
        self.trend_expert = TrendExpert(
            learning_rate=self.trend_learning_rate,
            max_iter=self.trend_max_iter,
            max_depth=self.trend_max_depth,
            random_state=self.random_state,
        )
        
        # Expert 2: Range (KNN) - Mean reversion fallback
        self.range_expert = KNeighborsClassifier(
            n_neighbors=15,
            weights="distance"
        )
        
        # Expert 3: Stress (LogReg) - Crash protection with high regularization
        self.stress_expert = LogisticRegression(
            C=0.1,  # High regularization
            class_weight={0: 2.0, 1: 1.0},  # Bias toward caution
            max_iter=500,
            random_state=self.random_state,
        )
        
        # Expert 4: Elastic (OU) - Physics-based mean reversion
        self.ou_expert: Optional[OUMeanReversionExpert] = None
        self._ou_enabled = False
        if self.use_ou:
            self.ou_expert = OUMeanReversionExpert(
                alpha=self.ou_alpha,
                lookback_window=self.ou_lookback,
                random_state=self.random_state,
            )
            self._ou_enabled = True
        
        # Gating Network (decides who speaks)
        self.gating_network = MLPClassifier(
            hidden_layer_sizes=(8,),
            activation="tanh",
            max_iter=self.gating_epochs,
            random_state=self.random_state,
        )
        
        # Scalers
        self.feature_scaler = StandardScaler()
        self.physics_scaler = StandardScaler()
        
        # CNN Expert (Pattern)
        self.cnn_expert: Optional[CNNExpert] = None
        self.cnn_source_column_: Optional[str] = None  # Track which column CNN uses
        self._cnn_enabled = False
        self._last_cnn_stats: Dict[str, float] = {}
        
        if isinstance(self.cnn_artifacts_dir, (str, Path)):
            self.cnn_artifacts_dir = Path(self.cnn_artifacts_dir)
        else:
            self.cnn_artifacts_dir = None
        
        self._fitted = False
    
    def fit(self, X, y, sample_weight=None) -> "MixtureOfExpertsEnsemble":
        """
        Fit all experts and gating network.
        
        Parameters
        ----------
        X : DataFrame
            Feature matrix (must include physics features)
        y : array
            Target labels
        sample_weight : array, optional
            Sample weights for physics gating
        """
        print("\n[MoE] Fitting Specialized Mixture of Experts...")
        
        df = _as_dataframe(X)
        df = df.copy()
        
        # Check if CNN source feature exists
        if self.cnn_source_feature in df.columns:
            self.cnn_source_column_ = self.cnn_source_feature
            print(f"  [MoE] CNN will use time-series feature: {self.cnn_source_feature}")
        else:
            self.cnn_source_column_ = None
            print(f"  [MoE] Warning: CNN source feature '{self.cnn_source_feature}' not found")
        
        # Base features (all features - CNN uses separate column)
        base_df = df.copy()
        physics_cols = _ensure_columns(base_df, self.physics_features)
        
        # Prepare features
        numeric_df = base_df.select_dtypes(include=["number"])
        self.feature_columns_ = list(numeric_df.columns)
        X_scaled = self.feature_scaler.fit_transform(numeric_df.to_numpy(dtype=float))
        y_array = np.ravel(np.asarray(y))
        
        # Physics-Aware Sample Weighting
        trend_sample_weight = sample_weight
        if "stability_warning" in base_df.columns:
            print("  [MoE] Applying Physics-Guided Sample Weighting...")
            warnings = base_df["stability_warning"].values
            
            # Create weights: 1.0 for stable, 0.1 for chaos
            physics_weights = np.ones(len(base_df))
            physics_weights[warnings == 1] = 0.1
            
            if sample_weight is not None:
                trend_sample_weight = sample_weight * physics_weights
            else:
                trend_sample_weight = physics_weights
        
        # Train Experts
        print("  [MoE] Training Expert 1: Trend (HistGBM)...")
        self.trend_expert.fit(base_df, y_array, sample_weight=trend_sample_weight)
        
        print("  [MoE] Training Expert 2: Range (KNN)...")
        self.range_expert.fit(X_scaled, y_array)
        
        print("  [MoE] Training Expert 3: Stress (LogReg)...")
        self.stress_expert.fit(X_scaled, y_array, sample_weight=sample_weight)
        
        # Train Elastic Expert (OU)
        if self._ou_enabled and self.ou_expert is not None:
            print("  [MoE] Training Expert 4: Elastic (OU Mean Reversion)...")
            self.ou_expert.fit(base_df, y_array)
            ou_params = self.ou_expert.get_ou_parameters()
            print(f"    [OU] θ={ou_params['theta']:.3f}, μ={ou_params['mu']:.3f}, half-life={ou_params['half_life']:.1f}")
        
        # Train Pattern Expert (CNN)
        print("  [MoE] Training Expert 5: Pattern (CNN)...")
        self._train_cnn_expert(df, y_array)
        
        # Train Gating Network
        print("  [MoE] Training Gating Network...")
        physics_matrix = base_df.loc[:, physics_cols].to_numpy(dtype=float)
        scaled_physics = self.physics_scaler.fit_transform(physics_matrix)
        regime_labels = _derive_regime_targets(physics_matrix)
        self.gating_network.fit(scaled_physics, regime_labels)
        
        self._gate_classes_ = list(self.gating_network.classes_)
        self._fitted = True
        
        print("  [MoE] ✓ All experts trained successfully")
        return self
    
    def _train_cnn_expert(self, df: pd.DataFrame, y_array: np.ndarray) -> None:
        """
        Train the temporal CNN expert (Pattern).
        
        NEW: Feeds raw time-series (frac_diff) to CNN instead of PCA latents.
        This gives CNN proper temporal structure to learn patterns.
        """
        self._cnn_enabled = False
        
        if not self.use_cnn:
            logger.info("CNNExpert disabled via configuration.")
            self.cnn_expert = None
            return
        
        if self.cnn_source_column_ is None:
            logger.info(f"CNNExpert skipped: source feature '{self.cnn_source_feature}' not found.")
            self.cnn_expert = None
            return
        
        # Extract the time-series column for CNN
        if self.cnn_source_column_ not in df.columns:
            logger.warning(f"CNNExpert skipped: column '{self.cnn_source_column_}' not in DataFrame.")
            self.cnn_expert = None
            return
        
        print(f"    [CNN] Using time-series column: {self.cnn_source_column_}")
        
        # Use tuned params if available
        params = self.cnn_params or {}
        
        y_series = pd.Series(y_array, index=df.index)
        
        # Create CNN input: single column as DataFrame (CNN expects DataFrame input)
        cnn_df = df[[self.cnn_source_column_]].copy()
        
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
            self._cnn_enabled = True
            print(f"    [CNN] ✓ Trained on {cnn_df.shape[1]} time-series feature(s) (window={self.cnn_expert.window_length})")
        except Exception as exc:
            logger.warning("CNNExpert training failed (%s). Falling back to 3-way MoE.", exc)
            self.cnn_expert = None
    
    def _gating_weights(self, physics_matrix: np.ndarray) -> np.ndarray:
        """
        Compute expert weights using physics-based gating.
        
        Returns
        -------
        weights : ndarray of shape (n_samples, n_experts)
            Normalized weights for each expert
        """
        scaled = self.physics_scaler.transform(physics_matrix)
        raw = self.gating_network.predict_proba(scaled)
        
        # Count active experts
        n_experts = 3  # Base: Trend, Range, Stress
        if self._ou_enabled:
            n_experts += 1
        if self._cnn_enabled:
            n_experts += 1
        weights = np.full((scaled.shape[0], n_experts), 1.0 / float(n_experts), dtype=float)
        
        # Map gating network outputs to expert weights
        for idx, cls in enumerate(self._gate_classes_):
            if cls < weights.shape[1]:
                weights[:, cls] = raw[:, idx]
        
        # Clip and normalize
        weights = np.clip(weights, 1e-6, None)
        weights /= weights.sum(axis=1, keepdims=True)
        
        # Physics Override: Boost Stress Expert during critical slowing down
        try:
            theta_idx = list(self.physics_features).index("stability_theta")
            theta_vals = physics_matrix[:, theta_idx]
            
            # Critical slowing down (theta -> 0)
            critical_mask = theta_vals < 0.005
            
            if np.any(critical_mask):
                # Boost Stress Expert (index 2)
                boost_amount = 2.0
                if weights.shape[1] > 2:
                    weights[critical_mask, 2] += boost_amount
                    weights[critical_mask] /= weights[critical_mask].sum(axis=1, keepdims=True)
        except ValueError:
            # stability_theta not in physics_features
            pass
        
        # If CNN disabled, ensure only 3 experts
        if not self._cnn_enabled and weights.shape[1] > 3:
            base = weights[:, :3]
            base /= base.sum(axis=1, keepdims=True)
            return base
        
        return weights
    
    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities using weighted expert ensemble.
        
        Parameters
        ----------
        X : DataFrame
            Feature matrix
        
        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Class probabilities
        """
        if not self._fitted:
            raise RuntimeError("MixtureOfExpertsEnsemble not fitted")
        
        df = _as_dataframe(X)
        df = df.copy()
        
        # Base features (all features available)
        base_df = df.copy()
        
        # Ensure all feature columns exist
        for col in self.feature_columns_:
            if col not in base_df.columns:
                base_df[col] = 0.0
        
        numeric = base_df[self.feature_columns_]
        X_scaled = self.feature_scaler.transform(numeric.to_numpy(dtype=float))
        
        # Get physics features for gating
        physics_matrix = base_df.loc[:, _ensure_columns(base_df, self.physics_features)].to_numpy(dtype=float)
        weights = self._gating_weights(physics_matrix)
        
        # Get expert predictions
        p_trend = self.trend_expert.predict_proba(base_df)
        p_range = self.range_expert.predict_proba(X_scaled)
        p_stress = self.stress_expert.predict_proba(X_scaled)
        
        # Start with base 3 experts
        expert_idx = 3
        blended = (
            weights[:, [0]] * p_trend +
            weights[:, [1]] * p_range +
            weights[:, [2]] * p_stress
        )
        
        # Add OU expert if enabled
        if self._ou_enabled and self.ou_expert is not None:
            p_ou = self.ou_expert.predict_proba(base_df)
            blended += weights[:, [expert_idx]] * p_ou
            expert_idx += 1
        
        # Add CNN if enabled
        self._last_cnn_stats = {"weight_mean": 0.0, "delta_mean": 0.0, "delta_std": 0.0}
        
        if self._cnn_enabled and self.cnn_expert is not None and self.cnn_source_column_:
            # Extract CNN source column
            if self.cnn_source_column_ in df.columns:
                cnn_df = df[[self.cnn_source_column_]].copy()
                p_cnn = self.cnn_expert.predict_proba(cnn_df)
                
                # Fallback for invalid CNN predictions
                fallback = np.column_stack([1.0 - blended[:, 1], blended[:, 1]])
                
                invalid = ~np.isfinite(p_cnn[:, 1])
                if invalid.any():
                    p_cnn[invalid] = fallback[invalid]
                
                # Add CNN contribution
                blended += weights[:, [expert_idx]] * p_cnn
                
                # Track CNN stats
                self._last_cnn_stats = {
                    "weight_mean": float(np.mean(weights[:, expert_idx])),
                    "delta_mean": 0.0,
                    "delta_std": 0.0,
                }
        
        return blended
    
    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        return np.argmax(self.predict_proba(X), axis=1)
    
    def get_expert_telemetry(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Get expert activation statistics.
        
        Returns
        -------
        telemetry : dict
            Expert weights and activation stats
        """
        if not self._fitted:
            return {}
        
        df = _as_dataframe(X).copy()
        # Use all features for telemetry (no need to drop CNN columns)
        physics_cols = _ensure_columns(df, self.physics_features)
        physics_matrix = df.loc[:, physics_cols].to_numpy(dtype=float)
        
        weights = self._gating_weights(physics_matrix)
        activation = weights.mean(axis=0)
        confidence = np.max(weights, axis=1).mean()
        
        # Extract expert shares
        expert_idx = 3
        share_ou = 0.0
        share_cnn = 0.0
        
        if self._ou_enabled and activation.shape[0] > expert_idx:
            share_ou = float(activation[expert_idx])
            expert_idx += 1
        
        if self._cnn_enabled and activation.shape[0] > expert_idx:
            share_cnn = float(activation[expert_idx])
        
        telemetry = {
            "share_trend": float(activation[0]),
            "share_range": float(activation[1]),
            "share_stress": float(activation[2]),
            "share_ou": share_ou,
            "share_cnn": share_cnn,
            "gating_confidence": float(confidence),
            "cnn_weight_mean": float(self._last_cnn_stats.get("weight_mean", share_cnn)),
            "cnn_delta_mean": float(self._last_cnn_stats.get("delta_mean", 0.0)),
            "cnn_delta_std": float(self._last_cnn_stats.get("delta_std", 0.0)),
        }
        
        return telemetry
