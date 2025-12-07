"""
Market Regime Model using Hidden Markov Models (HMM).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

class MarketRegimeModel:
    """
    Identifies market regimes (Trend, Range, Stress) using Gaussian HMM.
    """

    def __init__(self, n_components: int = 3, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.model = GaussianHMM(
            n_components=n_components,
            covariance_type="full",
            n_iter=100,
            random_state=random_state,
            verbose=False
        )
        self.scaler = StandardScaler()
        self.regime_map = {}

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Train HMM and label regimes. Returns DataFrame with 'regime' column.
        """
        data = df.copy()
        
        # Prepare features
        # Ensure we have the required columns
        if "log_ret" not in data.columns or "rolling_std_24" not in data.columns:
            raise ValueError("DataFrame must contain 'log_ret' and 'rolling_std_24'")

        X = data[["log_ret", "rolling_std_24"]].dropna()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train HMM
        print(f"[REGIME] Training HMM with {self.n_components} components...")
        self.model.fit(X_scaled)
        
        # Predict states
        states = self.model.predict(X_scaled)
        
        # Map states to human-readable labels
        self._map_regimes(X, states)
        
        # Assign labels
        data.loc[X.index, "regime_id"] = states
        data.loc[X.index, "regime"] = [self.regime_map[s] for s in states]
        
        return data

    def _map_regimes(self, X: pd.DataFrame, states: np.ndarray):
        """
        Auto-label states based on statistics.
        """
        df_stats = X.copy()
        df_stats["state"] = states
        
        stats = df_stats.groupby("state").agg({
            "log_ret": "mean",
            "rolling_std_24": "mean"
        })
        
        print("\n[REGIME] State Statistics:")
        print(stats)
        
        # 1. Identify Stress (Highest Volatility)
        stress_state = stats["rolling_std_24"].idxmax()
        
        # 2. Identify Trend (Highest Absolute Return among remaining)
        remaining = stats.drop(stress_state)
        trend_state = remaining["log_ret"].abs().idxmax()
        
        # 3. Identify Range (The last one)
        range_state = list(set(stats.index) - {stress_state, trend_state})[0]
        
        self.regime_map = {
            stress_state: "Stress",
            trend_state: "Trend",
            range_state: "Range"
        }
        
        print(f"[REGIME] Mapped States: {self.regime_map}")
