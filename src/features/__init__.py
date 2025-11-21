"""
Feature engineering utilities (formerly ``signal_factory.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
import warnings

from ..config import DAYS_BACK, FEATURE_STORE
from ..data_loader import MarketDataLoader

warnings.filterwarnings("ignore")


class SignalFactory:
    """
    Generates hundreds of alpha factors from OHLCV data using pandas_ta.
    """

    def __init__(self) -> None:
        self.windows = [3, 5, 8, 13, 14, 21, 34, 55, 89, 144, 200]
        self.lags = [1, 2, 3, 5, 8, 13]

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Expand an OHLCV DataFrame into a dense feature matrix.
        """
        print(f"[FEATURES] Starting Signal Factory on {len(df)} rows...")

        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        for col in ohlcv_cols:
            if col in df.columns:
                df[col] = df[col].astype(np.float32)

        df = df.copy()

        print("  [STEP A] Price transforms")
        df["log_ret"] = np.log(df["close"] / df["close"].shift(1)).astype(np.float32)
        df["log_range"] = np.log(df["high"] / df["low"]).astype(np.float32)

        body_size = np.abs(df["close"] - df["open"])
        shadow_size = (df["high"] - df["low"]) - body_size
        df["body_shadow_ratio"] = (body_size / (shadow_size + 1e-9)).astype(np.float32)

        upper_shadow = df["high"] - np.maximum(df["close"], df["open"])
        lower_shadow = np.minimum(df["close"], df["open"]) - df["low"]
        df["upper_shadow_ratio"] = (upper_shadow / (body_size + 1e-9)).astype(np.float32)
        df["lower_shadow_ratio"] = (lower_shadow / (body_size + 1e-9)).astype(np.float32)

        df["volatility"] = df["log_ret"].rolling(window=20).std().astype(np.float32)
        hurst_vals = ta.hurst(df["close"], length=100)
        if hurst_vals is not None:
            if isinstance(hurst_vals, pd.DataFrame):
                df["hurst"] = hurst_vals.iloc[:, 0].astype(np.float32)
            else:
                df["hurst"] = hurst_vals.astype(np.float32)

        print(f"  [STEP B] Parametric indicators for windows: {self.windows}")
        for window in self.windows:
            df[f"RSI_{window}"] = ta.rsi(df["close"], length=window).astype(np.float32)
            df[f"ROC_{window}"] = ta.roc(df["close"], length=window).astype(np.float32)
            df[f"CCI_{window}"] = ta.cci(df["high"], df["low"], df["close"], length=window).astype(np.float32)
            df[f"WILLR_{window}"] = ta.willr(df["high"], df["low"], df["close"], length=window).astype(np.float32)

            df[f"ATR_{window}"] = ta.atr(df["high"], df["low"], df["close"], length=window).astype(np.float32)
            df[f"NATR_{window}"] = ta.natr(df["high"], df["low"], df["close"], length=window).astype(np.float32)

            adx = ta.adx(df["high"], df["low"], df["close"], length=window)
            if adx is not None:
                df[f"ADX_{window}"] = adx[f"ADX_{window}"].astype(np.float32)
                df[f"DMP_{window}"] = adx[f"DMP_{window}"].astype(np.float32)
                df[f"DMN_{window}"] = adx[f"DMN_{window}"].astype(np.float32)

            sma = ta.sma(df["close"], length=window)
            ema = ta.ema(df["close"], length=window)
            df[f"dist_SMA_{window}"] = ((df["close"] - sma) / sma).astype(np.float32)
            df[f"dist_EMA_{window}"] = ((df["close"] - ema) / ema).astype(np.float32)

            bb = ta.bbands(df["close"], length=window, std=2)
            if bb is not None:
                df[f"BB_pctB_{window}"] = bb.iloc[:, 4].astype(np.float32)
                df[f"BB_width_{window}"] = bb.iloc[:, 3].astype(np.float32)

            if window < 50:
                df[f"MFI_{window}"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=window).astype(
                    np.float32
                )

        df["OBV"] = ta.obv(df["close"], df["volume"]).astype(np.float32)
        df["CMF"] = ta.cmf(df["high"], df["low"], df["close"], df["volume"], length=20).astype(np.float32)
        macds = ta.macd(df["close"])
        if macds is not None:
            df["MACD"] = macds.iloc[:, 0].astype(np.float32)
            df["MACD_hist"] = macds.iloc[:, 1].astype(np.float32)
            df["MACD_signal"] = macds.iloc[:, 2].astype(np.float32)

        print("  [STEP C] Statistical features")
        for window in [20, 24, 50, 100]:
            df[f"rolling_mean_{window}"] = df["close"].rolling(window=window).mean().astype(np.float32)
            df[f"rolling_std_{window}"] = df["close"].rolling(window=window).std().astype(np.float32)
            df[f"skew_{window}"] = df["close"].rolling(window=window).skew().astype(np.float32)
            df[f"kurt_{window}"] = df["close"].rolling(window=window).kurt().astype(np.float32)

            rolling_mean = df["close"].rolling(window=window).mean()
            rolling_std = df["close"].rolling(window=window).std()
            df[f"zscore_{window}"] = ((df["close"] - rolling_mean) / rolling_std).astype(np.float32)
            df[f"slope_{window}"] = ta.slope(df["close"], length=window).astype(np.float32)

        print("  [STEP C.2] Physics / Chaos Features")
        # Hurst and Entropy on Returns
        df["hurst_100"] = rolling_hurst(df["log_ret"], window=100).astype(np.float32)
        df["entropy_100"] = rolling_entropy(df["log_ret"], window=100).astype(np.float32)
        
        # FDI on Price
        df["fdi_100"] = rolling_fdi(df["close"], window=100).astype(np.float32)

        print("  [STEP D] Lagged features")
        exclude_cols = ["open", "high", "low", "close", "volume", "timestamp"]
        base_features = [col for col in df.columns if col not in exclude_cols]
        print(f"      Lagging {len(base_features)} base features across lags {self.lags}")

        for feature in base_features:
            for lag in self.lags:
                df[f"{feature}_lag_{lag}"] = df[feature].shift(lag).astype(np.float32)

        print("  [STEP E] Cleanup")
        max_window = max(self.windows)
        df = df.iloc[max_window:].copy()
        before_drop = len(df)
        df = df.dropna()
        if len(df) < before_drop:
            print(f"      Dropped {before_drop - len(df)} rows with NaNs")

        float_cols = df.select_dtypes(include=["float64"]).columns
        if len(float_cols) > 0:
            df[float_cols] = df[float_cols].astype(np.float32)

        print(f"[FEATURES] Complete. Shape: {df.shape}")
        return df


def rolling_hurst(series: pd.Series, window: int = 100) -> pd.Series:
    """
    Calculate Rolling Hurst Exponent using the R/S method.
    H < 0.5: Mean Reversion
    H = 0.5: Random Walk
    H > 0.5: Trending
    """
    def get_hurst(x):
        if len(x) < 8:
            return 0.5
        x = np.array(x)
        # Calculate log returns to stationarize? 
        # Standard R/S usually works on returns or detrended prices. 
        # Let's use log returns of the window.
        # Actually, R/S is often defined on the series itself if checking for persistence in returns?
        # Usually: Input is Returns.
        # Let's assume input is log_ret.
        
        # We need to handle the case where x is constant
        if np.std(x) == 0:
            return 0.5

        # Mean centered
        m = np.mean(x)
        y = x - m
        
        # Cumulative deviate
        z = np.cumsum(y)
        
        # Range
        r = np.max(z) - np.min(z)
        
        # Standard deviation
        s = np.std(x, ddof=1)
        
        # R/S
        if s == 0:
            return 0.5
        rs = r / s
        
        # Hurst estimate (simplified point estimate for fixed N)
        # H = log(R/S) / log(N)
        # Note: This is a rough proxy. Rigorous Hurst requires regression over multiple scales.
        # However, for a rolling feature, this proxy captures the "rescaled range" dynamic well enough.
        h = np.log(rs) / np.log(len(x))
        return h

    # Apply to rolling window
    # Note: Input should be returns, not prices, for R/S analysis of *returns* persistence.
    # If we want persistence of *price* trend, we use price? 
    # Standard interpretation: H of returns series.
    return series.rolling(window=window).apply(get_hurst, raw=True)


def rolling_fdi(series: pd.Series, window: int = 100) -> pd.Series:
    """
    Calculate Rolling Fractal Dimension Index (FDI).
    Uses the normalized path length method.
    """
    def get_fdi(x):
        if len(x) < 2:
            return 1.5
        x = np.array(x)
        
        # Normalize price to [0, 1]
        max_x = np.max(x)
        min_x = np.min(x)
        range_x = max_x - min_x
        if range_x == 0:
            return 1.5
            
        norm_x = (x - min_x) / range_x
        
        # Normalized time steps (0 to 1)
        n = len(x)
        norm_t = np.linspace(0, 1, n)
        
        # Calculate path length
        dt = norm_t[1] - norm_t[0]
        dx = np.diff(norm_x)
        length = np.sum(np.sqrt(dx**2 + dt**2))
        
        # FDI formula
        # FDI = 1 + (log(L) + log(2)) / log(2*n)
        # Note: There are variations. This is a common one for time series.
        fdi = 1 + (np.log(length) + np.log(2)) / np.log(2 * (n - 1))
        return fdi

    return series.rolling(window=window).apply(get_fdi, raw=True)


def rolling_entropy(series: pd.Series, window: int = 100, bins: int = 20) -> pd.Series:
    """
    Calculate Rolling Shannon Entropy of returns.
    Higher entropy = more unpredictability/noise.
    """
    def get_entropy(x):
        if len(x) < 2:
            return 0.0
        # Histogram
        counts, _ = np.histogram(x, bins=bins, density=True)
        # Probabilities (normalize density to sum to 1 roughly, or just use counts)
        # Actually np.histogram(density=True) gives PDF value, not probability mass.
        # We need probability mass.
        counts, _ = np.histogram(x, bins=bins)
        probs = counts / np.sum(counts)
        probs = probs[probs > 0] # Remove zeros for log
        
        # Shannon Entropy
        ent = -np.sum(probs * np.log2(probs))
        return ent

    return series.rolling(window=window).apply(get_entropy, raw=True)


def build_feature_dataset(
    loader: Optional[MarketDataLoader] = None,
    days_back: int = DAYS_BACK,
    force_refresh: bool = False,
    output_path: Path | str = FEATURE_STORE,
) -> pd.DataFrame:
    """
    Convenience helper that fetches raw data, builds signals, and stores them.
    """
    loader = loader or MarketDataLoader()
    ohlcv = loader.get_data(days_back=days_back, force_refresh=force_refresh)
    factory = SignalFactory()
    features = factory.generate_signals(ohlcv)

    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(out_path)
        print(f"[FEATURES] Saved {len(features)} rows to {out_path}")

    return features


def add_signal_interactions(
    df: pd.DataFrame,
    primary_signal: pd.Series,
    reference_df: Optional[pd.DataFrame] = None,
    volatility_col: str = "volatility",
    trend_col: str = "hurst",
) -> pd.DataFrame:
    """
    Append interaction terms needed by the meta-model.
    """
    signal = primary_signal.reindex(df.index).fillna(0).astype(np.float32)
    working = df.copy()

    if volatility_col not in working.columns:
        working[volatility_col] = _derive_volatility(working, reference_df).astype(np.float32)
    else:
        working[volatility_col] = working[volatility_col].astype(np.float32).fillna(0)

    if trend_col not in working.columns:
        working[trend_col] = _derive_hurst(working, reference_df).astype(np.float32)
    else:
        working[trend_col] = working[trend_col].astype(np.float32).fillna(0)

    working["volatility_x_signal"] = (working[volatility_col] * signal).astype(np.float32)
    working["trend_x_signal"] = (working[trend_col] * signal).astype(np.float32)

    return working


def _derive_volatility(features_df: pd.DataFrame, reference_df: Optional[pd.DataFrame]) -> pd.Series:
    if "log_ret" in features_df.columns:
        vol = features_df["log_ret"].rolling(window=20).std()
        return vol.fillna(0)
    if reference_df is not None and "close" in reference_df.columns:
        log_ret = np.log(reference_df["close"] / reference_df["close"].shift(1))
        vol = log_ret.rolling(window=20).std()
        return vol.reindex(features_df.index).fillna(0)
    return pd.Series(0.0, index=features_df.index)


from .advanced_stats import (
    apply_rolling_physics,
    calculate_fdi,
    calculate_hurst_rs,
    calculate_shannon_entropy,
)
from .alpha_council import AlphaCouncil

__all__ = [
    "AlphaCouncil",
    "apply_rolling_physics",
    "calculate_fdi",
    "calculate_hurst_rs",
    "calculate_shannon_entropy",
]


def _derive_hurst(features_df: pd.DataFrame, reference_df: Optional[pd.DataFrame]) -> pd.Series:
    price_series = None
    if reference_df is not None and "close" in reference_df.columns:
        price_series = reference_df["close"]
    elif "close" in features_df.columns:
        price_series = features_df["close"]

    if price_series is not None and hasattr(ta, "hurst"):
        hurst_vals = ta.hurst(price_series, length=100)
        if hurst_vals is not None:
            if isinstance(hurst_vals, pd.DataFrame):
                hurst_series = hurst_vals.iloc[:, 0]
            else:
                hurst_series = hurst_vals
            return hurst_series.reindex(features_df.index).fillna(0)

    return pd.Series(0.0, index=features_df.index)
