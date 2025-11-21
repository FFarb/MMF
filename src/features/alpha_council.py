"""
Voting-based feature screening inspired by a council of mathematical experts.

Expert A (linear) seeks parsimony through sparse penalties, Expert B
(non-linear) gauges conditional importance via ensemble splits, and Expert C
captures chaotic dependencies using mutual information. Only features that
convince at least two experts survive the selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler


ArrayLike = Iterable[float]


@dataclass
class AlphaCouncil:
    """
    Trio of feature selectors that vote on predictive usefulness.

    Parameters
    ----------
    top_ratio : float, optional
        Fraction of features that each expert considers elite; defaults to 0.5.
    random_state : int, optional
        Seed used by stochastic experts to preserve reproducibility.
    forest_estimators : int, optional
        Number of estimators for the RandomForest-based expert.
    """

    top_ratio: float = 0.5
    random_state: int = 42
    forest_estimators: int = 300

    def __post_init__(self) -> None:
        if not 0 < self.top_ratio <= 1:
            raise ValueError("top_ratio must be within (0, 1].")

    def _prepare_inputs(
        self, X: Sequence[Sequence[float]] | pd.DataFrame, y: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray, List[str]]:
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
            features = X.to_numpy(dtype=float)
        else:
            features = np.asarray(X, dtype=float)
            feature_names = [f"feature_{idx}" for idx in range(features.shape[1])]

        if features.ndim != 2:
            raise ValueError("Feature matrix must be 2-dimensional.")
        if features.shape[1] == 0:
            raise ValueError("No features supplied to AlphaCouncil.")

        y_array = np.ravel(np.asarray(y))
        if y_array.size != features.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        return scaled_features, y_array, feature_names

    def _linear_expert(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        try:
            model = LassoCV(
                cv=5,
                random_state=self.random_state,
                n_alphas=50,
                max_iter=10000,
            )
            model.fit(X, y)
            coefs = np.abs(model.coef_)
            if np.allclose(coefs, 0):
                # Fall back to ANOVA F-test if Lasso collapses
                f_scores, _ = f_classif(X, y)
                coefs = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)
            return coefs
        except Exception as exc:  # pragma: no cover - safety net
            raise RuntimeError("Linear expert failed.") from exc

    def _nonlinear_expert(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        try:
            forest = RandomForestClassifier(
                n_estimators=self.forest_estimators,
                max_depth=None,
                min_samples_split=4,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight="balanced_subsample",
            )
            forest.fit(X, y)
            return forest.feature_importances_
        except Exception as exc:  # pragma: no cover - safety net
            raise RuntimeError("Non-linear expert failed.") from exc

    def _chaos_expert(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        try:
            mi = mutual_info_classif(X, y, discrete_features=False, random_state=self.random_state)
            return np.nan_to_num(mi, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as exc:  # pragma: no cover - safety net
            raise RuntimeError("Chaos expert failed.") from exc

    def _vote_mask(self, scores: np.ndarray) -> np.ndarray:
        n_features = scores.shape[0]
        elite_count = max(1, int(np.ceil(n_features * self.top_ratio)))
        # Rank from best to worst
        order = np.argsort(-scores)
        mask = np.zeros(n_features, dtype=bool)
        mask[order[:elite_count]] = True
        return mask

    def screen_features(
        self, X: Sequence[Sequence[float]] | pd.DataFrame, y: ArrayLike
    ) -> List[str]:
        """
        Screen candidate features using a multi-expert voting process.

        Parameters
        ----------
        X : array-like or DataFrame
            Feature matrix.
        y : array-like
            Target labels.

        Returns
        -------
        List[str]
            Names of features endorsed by at least two experts.
        """
        X_scaled, y_array, feature_names = self._prepare_inputs(X, y)

        expert_masks = []
        for scores in (
            self._linear_expert(X_scaled, y_array),
            self._nonlinear_expert(X_scaled, y_array),
            self._chaos_expert(X_scaled, y_array),
        ):
            expert_masks.append(self._vote_mask(scores))

        vote_totals = np.sum(np.column_stack(expert_masks), axis=1)
        survivors = [
            name for name, votes in zip(feature_names, vote_totals) if votes >= 2
        ]

        if not survivors:
            # When consensus fails, fall back to the best linear features.
            linear_scores = self._linear_expert(X_scaled, y_array)
            fallback_mask = self._vote_mask(linear_scores)
            survivors = [name for name, keep in zip(feature_names, fallback_mask) if keep]

        return survivors


__all__ = ["AlphaCouncil"]
