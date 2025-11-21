"""
Bicameral Hybrid Ensemble: Neuro-Symbolic Trading System.

This module implements a two-system architecture that fuses:
1. Symbolic Analyst (Gradient Boosting) - Logic, levels, support/resistance
2. Topological Visionary (Deep Learning) - Patterns, shapes, multi-scale features

A Meta-Learner (Stacking) arbitrates between these experts based on market entropy
and volatility, learning when to trust each system.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
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


def _compute_entropy(X: np.ndarray) -> float:
    """Compute market entropy from feature matrix (simple proxy)."""
    if X.shape[0] < 2:
        return 0.5
    # Use coefficient of variation as entropy proxy
    std_vals = np.std(X, axis=0)
    mean_vals = np.abs(np.mean(X, axis=0)) + 1e-8
    cv = np.mean(std_vals / mean_vals)
    return min(max(cv, 0.0), 1.0)


def _compute_volatility(X: np.ndarray) -> float:
    """Compute market volatility from feature matrix (simple proxy)."""
    if X.shape[0] < 2:
        return 0.01
    # Use average standard deviation as volatility proxy
    return float(np.mean(np.std(X, axis=0)))


@dataclass
class HybridTrendExpert(BaseEstimator, ClassifierMixin):
    """
    Bicameral Trend Expert: Fuses Symbolic Analyst and Topological Visionary.
    
    Architecture:
    - Analyst: GradientBoostingClassifier (symbolic reasoning)
    - Visionary: TorchSklearnWrapper(AdaptiveConvExpert) (neural pattern recognition)
    - Meta-Learner: LogisticRegression (arbitrator)
    
    Training uses K-Fold stacking to generate unbiased meta-features.
    
    Parameters
    ----------
    n_estimators : int, optional
        Number of boosting estimators for Analyst (default: 300).
    learning_rate : float, optional
        Learning rate for Analyst (default: 0.05).
    max_depth : int, optional
        Max depth for Analyst trees (default: 3).
    random_state : int, optional
        Random seed (default: 42).
    """
    
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 3
    random_state: int = 42
    
    def __post_init__(self) -> None:
        # Symbolic Analyst: Gradient Boosting
        self.analyst = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        
        # Topological Visionary: Deep Learning (initialized in fit)
        self.visionary = None
        
        # Meta-Learner: Arbitrator
        self.meta_learner = LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            random_state=self.random_state,
        )
        
        self.scaler_ = StandardScaler()
        self._fitted = False
        self.classes_ = np.array([0, 1])
        
    def fit(
        self,
        X: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
        y: Iterable[int | float],
    ) -> "HybridTrendExpert":
        """
        Train the Bicameral Hybrid Expert using K-Fold stacking.
        
        Process:
        1. 2-Fold cross-validation to generate unbiased meta-features
        2. For each fold:
           - Train Analyst and Visionary on Fold A
           - Predict probabilities on Fold B
        3. Build meta-features: [P_analyst, P_visionary, entropy, volatility]
        4. Train meta-learner on meta-features
        5. Retrain base models on full dataset
        """
        df = _as_dataframe(X)
        numeric_df = df.select_dtypes(include=["number"])
        
        if numeric_df.empty:
            raise ValueError("HybridTrendExpert requires numeric features.")
        
        X_array = numeric_df.to_numpy(dtype=float)
        y_array = np.ravel(np.asarray(y))
        
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError("Mismatched samples between X and y.")
        
        # Scale features
        X_scaled = self.scaler_.fit_transform(X_array)
        
        # Initialize Visionary with correct feature count
        n_features = X_scaled.shape[1]
        self.visionary = TorchSklearnWrapper(
            n_features=n_features,
            hidden_dim=32,
            sequence_length=16,
            lstm_hidden=64,
            dropout=0.2,
            learning_rate=0.001,
            batch_size=64,
            max_epochs=100,
            patience=10,
            validation_split=0.15,
            random_state=self.random_state,
        )
        
        # K-Fold Stacking (k=2) to generate unbiased meta-features
        print("    [Bicameral] Generating meta-features via 2-fold stacking...")
        kfold = KFold(n_splits=2, shuffle=True, random_state=self.random_state)
        
        meta_features = np.zeros((X_scaled.shape[0], 4))  # [P_analyst, P_visionary, entropy, vol]
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_scaled)):
            X_train_fold = X_scaled[train_idx]
            y_train_fold = y_array[train_idx]
            X_val_fold = X_scaled[val_idx]
            
            # Train Analyst on this fold
            analyst_fold = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state + fold_idx,
            )
            analyst_fold.fit(X_train_fold, y_train_fold)
            
            # Train Visionary on this fold
            visionary_fold = TorchSklearnWrapper(
                n_features=n_features,
                hidden_dim=32,
                sequence_length=16,
                lstm_hidden=64,
                dropout=0.2,
                learning_rate=0.001,
                batch_size=64,
                max_epochs=100,
                patience=10,
                validation_split=0.15,
                random_state=self.random_state + fold_idx,
            )
            visionary_fold.fit(X_train_fold, y_train_fold)
            
            # Predict on validation fold
            p_analyst = analyst_fold.predict_proba(X_val_fold)[:, 1]
            p_visionary = visionary_fold.predict_proba(X_val_fold)[:, 1]
            
            # Compute market conditions for validation fold
            entropy_val = _compute_entropy(X_val_fold)
            volatility_val = _compute_volatility(X_val_fold)
            
            # Store meta-features
            meta_features[val_idx, 0] = p_analyst
            meta_features[val_idx, 1] = p_visionary
            meta_features[val_idx, 2] = entropy_val
            meta_features[val_idx, 3] = volatility_val
        
        # Train meta-learner on meta-features
        print("    [Bicameral] Training meta-learner arbitrator...")
        self.meta_learner.fit(meta_features, y_array)
        
        # Retrain base models on full dataset
        print("    [Bicameral] Retraining Analyst and Visionary on full dataset...")
        self.analyst.fit(X_scaled, y_array)
        self.visionary.fit(X_scaled, y_array)
        
        self._fitted = True
        return self
    
    def _check_is_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("HybridTrendExpert must be fitted before use.")
    
    def predict_proba(
        self, X: pd.DataFrame | np.ndarray | Sequence[Sequence[float]]
    ) -> np.ndarray:
        """
        Predict probabilities using meta-learner arbitration.
        
        Process:
        1. Get probabilities from Analyst and Visionary
        2. Compute/extract entropy and volatility
        3. Build meta-features
        4. Return meta-learner predictions
        """
        self._check_is_fitted()
        
        df = _as_dataframe(X)
        numeric_df = df.select_dtypes(include=["number"])
        X_array = numeric_df.to_numpy(dtype=float)
        X_scaled = self.scaler_.transform(X_array)
        
        # Get base model probabilities
        p_analyst = self.analyst.predict_proba(X_scaled)[:, 1]
        p_visionary = self.visionary.predict_proba(X_scaled)[:, 1]
        
        # Compute market conditions
        # Try to extract from input if available, otherwise compute
        entropy_vals = np.full(X_scaled.shape[0], _compute_entropy(X_scaled))
        volatility_vals = np.full(X_scaled.shape[0], _compute_volatility(X_scaled))
        
        # Build meta-features
        meta_features = np.column_stack([
            p_analyst,
            p_visionary,
            entropy_vals,
            volatility_vals,
        ])
        
        # Meta-learner arbitration
        return self.meta_learner.predict_proba(meta_features)
    
    def predict(
        self, X: pd.DataFrame | np.ndarray | Sequence[Sequence[float]]
    ) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)


@dataclass
class MixtureOfExpertsEnsemble(BaseEstimator, ClassifierMixin):
    """
    Adaptive ensemble that routes samples to experts based on chaos metrics.
    
    Now uses HybridTrendExpert (Bicameral) for trend detection, combining
    symbolic and neural approaches with meta-learning arbitration.

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
    trend_estimators: int = 300
    gating_epochs: int = 500

    def __post_init__(self) -> None:
        # Bicameral Hybrid Trend Expert
        self.trend_expert = HybridTrendExpert(
            n_estimators=self.trend_estimators,
            learning_rate=0.05,
            max_depth=3,
            random_state=self.random_state,
        )
        
        # Fallback specialists
        self.range_expert = KNeighborsClassifier(n_neighbors=15, weights="distance")
        self.stress_expert = LogisticRegression(
            class_weight={0: 2.0, 1: 1.0},
            max_iter=500,
            solver="lbfgs",
        )
        
        # Gating network
        self.gating_network = MLPClassifier(
            hidden_layer_sizes=(8,),
            activation="tanh",
            alpha=1e-3,
            max_iter=self.gating_epochs,
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
        print("    [MoE] Training Bicameral Hybrid Trend Expert...")
        self.trend_expert.fit(scaled_features, y_array)
        
        print("    [MoE] Training Range Expert (kNN)...")
        self.range_expert.fit(scaled_features, y_array)
        
        print("    [MoE] Training Stress Expert (Logistic)...")
        self.stress_expert.fit(scaled_features, y_array)

        # Train gating network on heuristic regimes
        print("    [MoE] Training Gating Network...")
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

    def get_system_complexity(self) -> Dict[str, int]:
        """
        Estimate the effective parameter count of the hybrid ensemble.
        """
        self._check_is_fitted()

        # Gating network (MLP) parameters
        mlp_params = 0
        for coef in self.gating_network.coefs_:
            mlp_params += coef.size
        for intercept in self.gating_network.intercepts_:
            mlp_params += intercept.size

        # Bicameral Trend Expert complexity
        trend_complexity = 0
        
        # Analyst (GradientBoosting) nodes
        if hasattr(self.trend_expert.analyst, "estimators_"):
            for tree_set in self.trend_expert.analyst.estimators_:
                for tree in tree_set:
                    trend_complexity += getattr(tree.tree_, "node_count", 0)
        
        # Visionary (Deep Learning) parameters
        if self.trend_expert.visionary is not None and self.trend_expert.visionary.model_ is not None:
            for param in self.trend_expert.visionary.model_.parameters():
                trend_complexity += param.numel()

        # Range expert stores training samples
        range_complexity = int(getattr(self.range_expert, "_fit_X", np.empty(0)).size)

        # Stress expert coefficients
        stress_complexity = 0
        if hasattr(self.stress_expert, "coef_"):
            stress_complexity += self.stress_expert.coef_.size
            stress_complexity += getattr(self.stress_expert, "intercept_", np.empty(0)).size

        total = mlp_params + trend_complexity + range_complexity + stress_complexity
        return {
            "Gating_Network_Params": mlp_params,
            "Bicameral_Trend_Expert": trend_complexity,
            "Range_Expert_Memory": range_complexity,
            "Stress_Expert_Coefs": stress_complexity,
            "Total_System_Complexity": total,
        }


__all__ = ["MixtureOfExpertsEnsemble", "HybridTrendExpert"]
