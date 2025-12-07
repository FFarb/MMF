"""
Meta-labeling trainer that filters primary model signals using HistGradientBoosting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

from .config import META_PROB_THRESHOLD, RANDOM_SEED, TRAIN_SPLIT


@dataclass
class MetaModelTrainer:
    """
    Implements Marcos Lopez de Prado's meta-labeling architecture.
    """

    probability_threshold: float = META_PROB_THRESHOLD
    train_split: float = TRAIN_SPLIT
    random_state: int = RANDOM_SEED
    model_params: Dict[str, object] = field(default_factory=dict)

    def build_meta_dataset(
        self, X: pd.DataFrame, primary_side: pd.Series, target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Keep only rows where the primary model wants to trade and create meta labels.
        """
        enriched, signal = self._attach_signal_features(X, primary_side)
        mask = signal != 0
        if mask.sum() == 0:
            raise ValueError("Primary model never produced a tradeable signal.")

        X_meta = enriched.loc[mask].copy()
        meta_labels = (target.reindex(X_meta.index).fillna(0) == 1).astype(int)
        return X_meta, meta_labels

    def fit(
        self,
        X: pd.DataFrame,
        primary_side: pd.Series,
        target: pd.Series,
    ) -> Tuple[HistGradientBoostingClassifier, Dict[str, float], pd.Series]:
        """
        Train the meta-model with an internal chronological split.
        """
        X_meta, y_meta = self.build_meta_dataset(X, primary_side, target)
        split_idx = max(1, int(len(X_meta) * self.train_split))
        X_train, X_test = X_meta.iloc[:split_idx], X_meta.iloc[split_idx:]
        y_train, y_test = y_meta.iloc[:split_idx], y_meta.iloc[split_idx:]

        params = {
            "loss": "log_loss",
            "max_iter": 400,
            "learning_rate": 0.05,
            "max_depth": 5,
            "random_state": self.random_state,
            "early_stopping": True,
        }
        params.update(self.model_params)

        model = HistGradientBoostingClassifier(**params)
        model.fit(X_train, y_train)

        if len(X_test) > 0:
            y_pred = model.predict(X_test)
            metrics = {
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "accuracy": accuracy_score(y_test, y_pred),
            }
        else:
            metrics = {"precision": np.nan, "recall": np.nan, "accuracy": np.nan}

        success_prob = pd.Series(model.predict_proba(X_meta)[:, 1], index=X_meta.index)
        return model, metrics, success_prob

    def predict_success_probability(
        self,
        model: HistGradientBoostingClassifier,
        X: pd.DataFrame,
        primary_side: pd.Series,
    ) -> pd.Series:
        """
        Return meta-model probabilities for every timestamp.
        """
        enriched, _ = self._attach_signal_features(X, primary_side)
        proba = model.predict_proba(enriched)[:, 1]
        return pd.Series(proba, index=enriched.index)

    def filter_signals(
        self,
        primary_side: pd.Series,
        success_probability: pd.Series,
    ) -> pd.Series:
        """
        Apply the decision rule: trade only when side != 0 AND success prob exceeds threshold.
        """
        aligned_prob = success_probability.sort_index()
        aligned_signal = primary_side.reindex(aligned_prob.index).fillna(0).astype(np.int8)

        mask = (aligned_signal != 0) & (aligned_prob > self.probability_threshold)
        filtered = np.where(mask, aligned_signal, 0)
        return pd.Series(filtered, index=success_probability.index, dtype=np.int8)

    def _attach_signal_features(
        self,
        X: pd.DataFrame,
        primary_side: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Add the raw signal and interaction columns expected by the meta-model.
        """
        signal = primary_side.reindex(X.index).fillna(0).astype(np.float32)
        enriched = X.copy()
        enriched["primary_signal"] = signal

        if "volatility" in enriched.columns and "volatility_x_signal" not in enriched.columns:
            enriched["volatility_x_signal"] = (enriched["volatility"] * signal).astype(np.float32)

        if "hurst" in enriched.columns and "trend_x_signal" not in enriched.columns:
            enriched["trend_x_signal"] = (enriched["hurst"] * signal).astype(np.float32)

        return enriched, signal
