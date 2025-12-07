"""
Meta-controller responsible for adapting training effort to market difficulty.

The scheduler reacts to the entropy of current data, recent volatility, and the
stability of validation losses to determine how aggressive retraining should be.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

import numpy as np


@dataclass
class TrainingScheduler:
    """
    Adaptive training controller that balances exploration depth with stability.

    Parameters
    ----------
    base_estimators : int, optional
        Default estimator count (for tree ensembles) or proxy for epoch count.
    base_epochs : int, optional
        Default neural training epochs used when the environment is neutral.
    validation_loss_history : Iterable[float], optional
        Historical validation losses used to measure learning stability.
    """

    base_estimators: int = 200
    base_epochs: int = 25
    validation_loss_history: Iterable[float] | None = None
    _loss_history: List[float] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.base_estimators <= 0 or self.base_epochs <= 0:
            raise ValueError("Base estimators and epochs must be positive.")
        if self.validation_loss_history is not None:
            self._loss_history = [
                float(loss) for loss in self.validation_loss_history if np.isfinite(loss)
            ]

    def update_validation_losses(self, losses: Iterable[float]) -> None:
        """
        Append new validation loss observations.
        """
        for loss in losses:
            if np.isfinite(loss):
                self._loss_history.append(float(loss))

    def _loss_instability_penalty(self) -> float:
        if len(self._loss_history) < 3:
            return 0.0
        tail = np.asarray(self._loss_history[-5:], dtype=float)
        slope = tail[-1] - tail.mean()
        variance = np.var(tail)
        penalty = 0.0
        if slope > 0:
            penalty += 0.1
        if variance > 0.01 * (tail.mean() + 1e-9):
            penalty += 0.1
        return penalty

    def suggest_training_depth(self, entropy: float, volatility: float) -> dict[str, int]:
        """
        Suggest estimator/epoch depth given current market complexity.

        Parameters
        ----------
        entropy : float
            Market entropy proxy. High entropy indicates chaotic regimes.
        volatility : float
            Normalized volatility used as a secondary difficulty gauge.

        Returns
        -------
        dict[str, int]
            Recommended `n_estimators` and `epochs` values.
        """
        if not np.isfinite(entropy) or not np.isfinite(volatility):
            raise ValueError("Entropy and volatility must be finite numbers.")

        estimator_budget = float(self.base_estimators)
        epoch_budget = float(self.base_epochs)

        if entropy > 0.9:
            estimator_budget *= 1.5
            epoch_budget *= 1.5
        elif entropy < 0.5:
            estimator_budget *= 0.7
            epoch_budget *= 0.85

        if volatility > 1.25:
            estimator_budget *= 1.2
        elif volatility < 0.75:
            estimator_budget *= 0.9

        penalty = self._loss_instability_penalty()
        if penalty > 0:
            estimator_budget *= 1.0 + penalty
            epoch_budget *= 1.0 + penalty

        n_estimators = max(10, int(round(estimator_budget)))
        epochs = max(1, int(round(epoch_budget)))
        return {"n_estimators": n_estimators, "epochs": epochs}

    def should_retrain(self, performance_decay: float) -> bool:
        """
        Decide whether to trigger a retraining cycle.

        Parameters
        ----------
        performance_decay : float
            Relative change in rolling Sharpe ratio over the latest window. A
            drop of 20% should trigger retraining.

        Returns
        -------
        bool
            True when retraining is recommended.
        """
        if not np.isfinite(performance_decay):
            raise ValueError("Performance decay must be a finite float.")
        return performance_decay <= -0.20


__all__ = ["TrainingScheduler"]
