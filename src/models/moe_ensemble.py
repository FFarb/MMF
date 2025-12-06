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
from .sde_expert import SDEExpert  # LaP-SDE: Physics + Uncertainty Quantification
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


def _get_oracle_targets(
    experts: dict,
    X: pd.DataFrame,
    y_true: np.ndarray,
    use_ou: bool = True,
    use_cnn: bool = True,
    high_loss_threshold: float = 0.8,
) -> np.ndarray:
    """
    Generate oracle labels for gating network training.
    
    ORACLE TRAINING:
    - Instead of heuristic physics rules, train gating network to predict
      which expert is actually correct for each sample
    - Solves uniform weight problem by learning from expert performance
    
    Logic:
    ------
    1. Get predictions from all experts on training data
    2. Calculate cross-entropy loss for each expert per sample
    3. Target = index of expert with lowest loss
    4. If all experts have high loss → Stress Expert (safety)
    
    Parameters
    ----------
    experts : dict
        Dictionary of trained experts
        Keys: 'trend', 'range', 'stress', 'ou', 'cnn'
    X : DataFrame
        Feature matrix
    y_true : ndarray
        True labels
    use_ou : bool
        Whether OU expert is enabled
    use_cnn : bool
        Whether CNN expert is enabled
    high_loss_threshold : float
        Threshold for "all experts failing" (default to Stress)
    
    Returns
    -------
    oracle_labels : ndarray
        Expert index for each sample (0=Trend, 1=Range, 2=Stress, 3=Pattern, 4=Elastic)
    """
    n_samples = len(y_true)
    
    # Prepare data for different expert types
    # Trend, OU, CNN use DataFrame (they handle column filtering internally)
    # Range, Stress use scaled numeric array
    
    df = _as_dataframe(X)
    numeric_df = df.select_dtypes(include=["number"])
    
    # Get predictions from all experts
    expert_probas = []
    expert_names = []
    
    # Expert 1: Trend (uses DataFrame, handles filtering internally)
    expert_probas.append(experts['trend'].predict_proba(X)[:, 1])
    expert_names.append('Trend')
    
    # Expert 2: Range (uses scaled numeric array)
    expert_probas.append(experts['range'].predict_proba(numeric_df.to_numpy(dtype=float))[:, 1])
    expert_names.append('Range')
    
    # Expert 3: Stress (uses scaled numeric array)
    expert_probas.append(experts['stress'].predict_proba(numeric_df.to_numpy(dtype=float))[:, 1])
    expert_names.append('Stress')
    
    # Expert 4: OU/Elastic (uses DataFrame)
    if use_ou and experts.get('ou') is not None:
        expert_probas.append(experts['ou'].predict_proba(X)[:, 1])
        expert_names.append('Elastic')
    
    # Expert 5: CNN/Pattern (uses DataFrame)
    if use_cnn and experts.get('cnn') is not None:
        expert_probas.append(experts['cnn'].predict_proba(X)[:, 1])
        expert_names.append('Pattern')
    
    # Stack probabilities (n_samples, n_experts)
    probas_matrix = np.column_stack(expert_probas)
    
    # Calculate cross-entropy loss for each expert
    # CE = -[y*log(p) + (1-y)*log(1-p)]
    y_true_expanded = y_true[:, np.newaxis]  # (n_samples, 1)
    
    # Clip probabilities to avoid log(0)
    probas_clipped = np.clip(probas_matrix, 1e-7, 1 - 1e-7)
    
    # Cross-entropy loss per expert
    ce_loss = -(
        y_true_expanded * np.log(probas_clipped) +
        (1 - y_true_expanded) * np.log(1 - probas_clipped)
    )
    
    # Find expert with lowest loss for each sample
    oracle_labels = np.argmin(ce_loss, axis=1)
    
    # Safety: If all experts have high loss, default to Stress Expert
    min_loss = np.min(ce_loss, axis=1)
    high_loss_mask = min_loss > high_loss_threshold
    
    if high_loss_mask.any():
        # Find Stress expert index
        stress_idx = expert_names.index('Stress') if 'Stress' in expert_names else 2
        oracle_labels[high_loss_mask] = stress_idx
        n_stress_fallback = high_loss_mask.sum()
        print(f"  [Oracle] {n_stress_fallback} samples ({n_stress_fallback/n_samples*100:.1f}%) "
              f"defaulted to Stress (all experts failing)")
    
    # Log oracle distribution
    unique, counts = np.unique(oracle_labels, return_counts=True)
    print("  [Oracle] Expert Selection Distribution:")
    for label, count in zip(unique, counts):
        expert_name = expert_names[label] if label < len(expert_names) else f"Expert {label}"
        pct = count / n_samples * 100
        avg_loss = np.mean(ce_loss[:, label])
        print(f"    {expert_name}: {count} ({pct:.1f}%) - Avg Loss: {avg_loss:.3f}")
    
    return oracle_labels


@dataclass
class TrendExpert(BaseEstimator, ClassifierMixin):
    """
    Pure HistGradientBoosting expert for sustainable trends.
    
    ASSET-AWARE UPGRADE:
    - Treats asset_id as categorical feature
    - Allows learning asset-specific rules (SOL != ETH)
    - Shares neural backbone while respecting asset physics
    
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
        # Note: categorical_features will be set dynamically in fit()
        self.model = None
        self.scaler_ = StandardScaler()
        self._fitted = False
        self.classes_ = np.array([0, 1])
        self.asset_id_col_idx_ = None  # Track asset_id column index
        self.feature_names_ = None
    
    def fit(self, X, y, sample_weight=None) -> "TrendExpert":
        """
        Fit the trend expert.
        
        ASSET-AWARE: Keeps asset_id as categorical feature for asset-specific rules.
        
        Parameters
        ----------
        X : DataFrame or array
            Feature matrix (should include asset_id if available)
        y : array
            Target labels
        sample_weight : array, optional
            Sample weights (physics gating)
        """
        df = _as_dataframe(X)
        
        # Check if asset_id exists
        has_asset_id = 'asset_id' in df.columns
        
        if has_asset_id:
            # Encode asset_id as integer (categorical)
            asset_id_series = df['asset_id']
            unique_assets = sorted(asset_id_series.unique())
            asset_to_idx = {asset: idx for idx, asset in enumerate(unique_assets)}
            asset_id_encoded = asset_id_series.map(asset_to_idx).values
            
            # Get numeric features (excluding asset_id)
            numeric_df = df.select_dtypes(include=["number"])
            numeric_df = numeric_df.drop(columns=['asset_id'], errors='ignore')
            
            # Scale numeric features
            X_numeric = numeric_df.to_numpy(dtype=float)
            X_scaled = self.scaler_.fit_transform(X_numeric)
            
            # Concatenate scaled numeric + encoded asset_id
            X_array = np.column_stack([X_scaled, asset_id_encoded])
            
            # Track asset_id column index (last column)
            self.asset_id_col_idx_ = X_array.shape[1] - 1
            self.feature_names_ = list(numeric_df.columns) + ['asset_id']
            
            # Initialize model with categorical features
            self.model = HistGradientBoostingClassifier(
                learning_rate=self.learning_rate,
                max_iter=self.max_iter,
                max_depth=self.max_depth,
                validation_fraction=self.validation_fraction,
                early_stopping=self.early_stopping,
                random_state=self.random_state,
                categorical_features=[self.asset_id_col_idx_],  # Asset-aware!
            )
            
            print(f"  [TrendExpert] Asset-aware mode: {len(unique_assets)} assets, "
                  f"asset_id at index {self.asset_id_col_idx_}")
        else:
            # No asset_id, standard mode
            numeric_df = df.select_dtypes(include=["number"])
            X_array = numeric_df.to_numpy(dtype=float)
            X_scaled = self.scaler_.fit_transform(X_array)
            X_array = X_scaled
            
            self.asset_id_col_idx_ = None
            self.feature_names_ = list(numeric_df.columns)
            
            # Initialize model without categorical features
            self.model = HistGradientBoostingClassifier(
                learning_rate=self.learning_rate,
                max_iter=self.max_iter,
                max_depth=self.max_depth,
                validation_fraction=self.validation_fraction,
                early_stopping=self.early_stopping,
                random_state=self.random_state,
            )
            
            print(f"  [TrendExpert] Standard mode (no asset_id)")
        
        y_array = np.ravel(np.asarray(y))
        
        # Train with sample weights
        self.model.fit(X_array, y_array, sample_weight=sample_weight)
        
        self._fitted = True
        return self
    
    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities."""
        if not self._fitted:
            raise RuntimeError("TrendExpert not fitted")
        
        df = _as_dataframe(X)
        
        if self.asset_id_col_idx_ is not None:
            # Asset-aware mode
            asset_id_series = df['asset_id']
            
            # Encode asset_id (handle unseen assets by mapping to 0)
            unique_assets_train = [name for name in self.feature_names_ if name != 'asset_id']
            asset_id_encoded = pd.Categorical(asset_id_series).codes
            
            # Get numeric features
            numeric_df = df.select_dtypes(include=["number"])
            numeric_df = numeric_df.drop(columns=['asset_id'], errors='ignore')
            
            # Scale and concatenate
            X_scaled = self.scaler_.transform(numeric_df.to_numpy(dtype=float))
            X_array = np.column_stack([X_scaled, asset_id_encoded])
        else:
            # Standard mode
            numeric_df = df.select_dtypes(include=["number"])
            X_scaled = self.scaler_.transform(numeric_df.to_numpy(dtype=float))
            X_array = X_scaled
        
        return self.model.predict_proba(X_array)
    
    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        return np.argmax(self.predict_proba(X), axis=1)


@dataclass
class MixtureOfExpertsEnsemble(BaseEstimator, ClassifierMixin):
    """
    Specialized Mixture of Experts with 5 orthogonal experts.
    
    PHYSICS-ENHANCED ENSEMBLE:
    - Removed: GraphVisionary (complex, unstable)
    - Upgraded: LaP-SDE (physics-informed SDE with uncertainty quantification)
    
    Experts:
    1. Trend (HistGBM) - Sustainable trends
    2. Range (KNN) - Local patterns
    3. Stress (LogReg) - Crash protection
    4. Pattern (CNN) - Temporal sequences
    5. Stochastic (LaP-SDE) - Physics + uncertainty quantification
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
    use_asset_embedding: bool = False  # NEW: Enable asset-specific gating policies
    
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
            class_weight='balanced',  # Let data decide risk (removed manual bias)
            max_iter=500,
            random_state=self.random_state,
        )
        
        # Expert 4: Stochastic (LaP-SDE) - Physics + Uncertainty Quantification
        self.sde_expert: Optional[SDEExpert] = None
        self._ou_enabled = False  # Keep variable name for compatibility
        if self.use_ou:  # Keep param name for compatibility
            self.sde_expert = SDEExpert(
                latent_dim=64,        # Max capacity (ARD will reduce)
                hidden_dims=[512, 256, 128],  # Deep encoder
                lr=0.001,             # Conservative learning rate
                epochs=100,           # Training epochs
                beta_kl=1.0,          # ARD penalty (dimensionality control)
                lambda_sparse=0.01,   # Physics sparsity
                time_steps=10,        # SDE integration steps
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
        
        # Train Stochastic Expert (LaP-SDE)
        if self._ou_enabled and self.sde_expert is not None:
            print("  [MoE] Training Expert 4: Stochastic (LaP-SDE - Physics + Uncertainty)...")
            self.sde_expert.fit(base_df, y_array)
            
            # Get telemetry
            telemetry = self.sde_expert.get_telemetry()
            active_dims = telemetry.get('active_dimensions', 0)
            snr = telemetry.get('signal_to_noise', 0)
            print(f"    [LaP-SDE] Active Dims: {active_dims}, SNR: {snr:.4f}")
        
        # Train Pattern Expert (CNN)
        print("  [MoE] Training Expert 5: Pattern (CNN)...")
        self._train_cnn_expert(df, y_array)
        
        # Train Gating Network with Oracle Labels
        print("  [MoE] Training Gating Network (Oracle Mode)...")
        print("    Generating oracle labels from expert performance...")
        
        # Create expert dictionary for oracle training
        experts_dict = {
            'trend': self.trend_expert,
            'range': self.range_expert,
            'stress': self.stress_expert,
            'ou': self.sde_expert if self._ou_enabled else None,  # LaP-SDE
            'cnn': self.cnn_expert if self._cnn_enabled else None,
        }
        
        # Generate oracle labels (which expert is best for each sample)
        oracle_labels = _get_oracle_targets(
            experts=experts_dict,
            X=base_df,
            y_true=y_array,
            use_ou=self._ou_enabled,
            use_cnn=self._cnn_enabled,
        )
        
        # Train gating network on physics features → oracle labels
        physics_matrix = base_df.loc[:, physics_cols].to_numpy(dtype=float)
        scaled_physics = self.physics_scaler.fit_transform(physics_matrix)
        
        # MULTI-ASSET: Add asset embeddings if enabled
        if self.use_asset_embedding and 'asset_id' in df.columns:
            print("  [MoE] Adding asset embeddings to gating network...")
            
            # One-hot encode asset_id
            asset_ids = df['asset_id'].values
            unique_assets = np.unique(asset_ids)
            n_assets = len(unique_assets)
            
            # Create asset mapping
            asset_to_idx = {asset: idx for idx, asset in enumerate(unique_assets)}
            asset_indices = np.array([asset_to_idx[a] for a in asset_ids])
            
            # One-hot encoding
            asset_onehot = np.zeros((len(asset_ids), n_assets))
            asset_onehot[np.arange(len(asset_ids)), asset_indices] = 1
            
            # Concatenate physics features with asset embeddings
            gating_input = np.concatenate([scaled_physics, asset_onehot], axis=1)
            
            print(f"    [Gating] Physics features: {scaled_physics.shape[1]}")
            print(f"    [Gating] Asset embedding dim: {n_assets}")
            print(f"    [Gating] Total input dim: {gating_input.shape[1]}")
            
            # Store asset mapping for prediction
            self.asset_to_idx_ = asset_to_idx
            self.n_assets_ = n_assets
        else:
            gating_input = scaled_physics
            self.asset_to_idx_ = None
            self.n_assets_ = 0
        
        self.gating_network.fit(gating_input, oracle_labels)
        
        self._gate_classes_ = list(self.gating_network.classes_)
        self._fitted = True
        
        print("  [MoE] [OK] All experts trained successfully")
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
    
    def _gating_weights(self, physics_matrix: np.ndarray, asset_ids: np.ndarray = None) -> np.ndarray:
        """
        Compute expert weights using physics-based gating.
        
        Parameters
        ----------
        physics_matrix : ndarray
            Physics features
        asset_ids : ndarray, optional
            Asset identifiers for multi-asset training
        
        Returns
        -------
        weights : ndarray of shape (n_samples, n_experts)
            Normalized weights for each expert
        """
        scaled = self.physics_scaler.transform(physics_matrix)
        
        # MULTI-ASSET: Add asset embeddings if enabled
        if self.use_asset_embedding and asset_ids is not None and self.asset_to_idx_ is not None:
            # One-hot encode asset_ids
            asset_indices = np.array([self.asset_to_idx_.get(a, 0) for a in asset_ids])
            asset_onehot = np.zeros((len(asset_ids), self.n_assets_))
            asset_onehot[np.arange(len(asset_ids)), asset_indices] = 1
            
            # Concatenate with physics features
            gating_input = np.concatenate([scaled, asset_onehot], axis=1)
        else:
            gating_input = scaled
        
        raw = self.gating_network.predict_proba(gating_input)
        
        # Count active experts
        n_experts = 3  # Base: Trend, Range, Stress
        if self._ou_enabled:
            n_experts += 1
        if self._cnn_enabled:
            n_experts += 1
        weights = np.full((gating_input.shape[0], n_experts), 1.0 / float(n_experts), dtype=float)
        
        # Map gating network outputs to expert weights
        for idx, cls in enumerate(self._gate_classes_):
            if cls < weights.shape[1]:
                weights[:, cls] = raw[:, idx]
        
        # Clip and normalize
        weights = np.clip(weights, 1e-6, None)
        weights /= weights.sum(axis=1, keepdims=True)
        
        # REMOVED: Physics Override (was causing Stress expert overheating)
        # Let Oracle training determine when Stress is needed via cross-entropy
        
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
        
        # Extract asset_ids if present (for multi-asset training)
        asset_ids = df['asset_id'].values if 'asset_id' in df.columns else None
        
        weights = self._gating_weights(physics_matrix, asset_ids=asset_ids)
        
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
        
        # Add LaP-SDE expert if enabled
        if self._ou_enabled and self.sde_expert is not None and weights.shape[1] > expert_idx:
            p_sde = self.sde_expert.predict_proba(base_df)
            blended += weights[:, [expert_idx]] * p_sde
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
        
        # Extract asset_ids if present
        asset_ids = df['asset_id'].values if 'asset_id' in df.columns else None
        
        weights = self._gating_weights(physics_matrix, asset_ids=asset_ids)
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
