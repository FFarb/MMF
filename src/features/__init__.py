"""
Feature engineering utilities (formerly ``signal_factory.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import warnings

from ..indicators import ta_core as ta

from ..config import DAYS_BACK, FEATURE_STORE
from ..data_loader import MarketDataLoader
from .advanced_stats import (
    apply_rolling_physics,
    apply_stability_physics,
    calculate_fdi,
    calculate_hurst_rs,
    calculate_shannon_entropy,
)
from .alpha_council import AlphaCouncil
from .tensor_flex import TensorFlexFeatureRefiner, cluster_features

warnings.filterwarnings("ignore")


class SignalFactory:
    """
    Generates hundreds of alpha factors from OHLCV data using internal TA indicators.
    """

    def __init__(self) -> None:
        self.windows = [3, 5, 8, 13, 14, 21, 34, 55, 89, 144, 200]
        self.lags = [1, 2, 3, 5, 8, 13]

    def generate_signals(self, df: pd.DataFrame, macro_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Expand an OHLCV DataFrame into a dense feature matrix.
        
        MULTI-TIMEFRAME UPGRADE:
        If macro_df is provided (e.g., H1 data when df is M5), key macro indicators
        are calculated and broadcast to the micro timeframe to fix "context myopia".
        
        Parameters
        ----------
        df : pd.DataFrame
            Primary OHLCV data (e.g., M5 bars)
        macro_df : pd.DataFrame, optional
            Higher timeframe OHLCV data (e.g., H1 bars) for macro context
        
        Returns
        -------
        pd.DataFrame
            Feature matrix with both micro and macro (if provided) features
        """
        print(f"[FEATURES] Starting Signal Factory on {len(df)} rows...")
        
        if macro_df is not None:
            print(f"[FEATURES] MTF Mode: Injecting macro features from {len(macro_df)} H1 bars")

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

        df["volatility_20"] = df["log_ret"].rolling(window=20).std().astype(np.float32)
        df["volatility_100"] = df["log_ret"].rolling(window=100).std().astype(np.float32)
        df["volatility_200"] = df["log_ret"].rolling(window=200).std().astype(np.float32)
        df["volatility"] = df["volatility_20"]
        try:
            hurst_vals = ta.hurst(df["close"], length=100)
            if hurst_vals is not None:
                df["hurst"] = hurst_vals.astype(np.float32)
        except Exception:
            pass

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
                df[f"BB_pctB_{window}"] = bb[f"BBP_{window}_2"].astype(np.float32)
                df[f"BB_width_{window}"] = bb[f"BBB_{window}_2"].astype(np.float32)

            if window < 50:
                df[f"MFI_{window}"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=window).astype(
                    np.float32
                )

        df["OBV"] = ta.obv(df["close"], df["volume"]).astype(np.float32)
        df["CMF"] = ta.cmf(df["high"], df["low"], df["close"], df["volume"], length=20).astype(np.float32)
        macds = ta.macd(df["close"])
        if macds is not None:
            df["MACD"] = macds["MACD_12_26_9"].astype(np.float32)
            df["MACD_hist"] = macds["MACDh_12_26_9"].astype(np.float32)
            df["MACD_signal"] = macds["MACDs_12_26_9"].astype(np.float32)

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

        print("  [STEP C.2] Physics / Chaos Features (Numba Accelerated)")
        df = apply_rolling_physics(df, windows=[100, 200])
        df = apply_stability_physics(df, window=168)
        for window in (100, 200):
            for feature in ("hurst", "entropy", "fdi"):
                col = f"{feature}_{window}"
                if col in df.columns:
                    df[col] = df[col].astype(np.float32)
        
        # MULTI-TIMEFRAME FEATURE INJECTION
        if macro_df is not None:
            print("  [STEP MTF] Multi-Timeframe Feature Injection")
            df = self._inject_macro_features(df, macro_df)

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
    
    def _inject_macro_features(self, df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
        """
        Inject macro (higher timeframe) features into micro (lower timeframe) data.
        
        This fixes "context myopia" where 5-minute data loses the macro trend view.
        Key macro indicators are calculated on H1 data and broadcast to M5 timestamps.
        
        Parameters
        ----------
        df : pd.DataFrame
            Micro timeframe data (e.g., M5) with datetime index
        macro_df : pd.DataFrame
            Macro timeframe data (e.g., H1) with datetime index
        
        Returns
        -------
        pd.DataFrame
            df with macro features added (macro_rsi, macro_trend, etc.)
        """
        macro_df = macro_df.copy()
        
        # Ensure both have datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
        
        if not isinstance(macro_df.index, pd.DatetimeIndex):
            if 'timestamp' in macro_df.columns:
                macro_df = macro_df.set_index('timestamp')
        
        print(f"    Macro timeframe: {len(macro_df)} bars")
        print(f"    Micro timeframe: {len(df)} bars")
        
        # Calculate key macro indicators
        macro_features = pd.DataFrame(index=macro_df.index)
        
        # 1. Macro RSI (14-period on H1)
        macro_features['macro_rsi'] = ta.rsi(macro_df['close'], length=14).astype(np.float32)
        
        # 2. Macro Trend (SMA Cross: 50 vs 200)
        sma_50 = ta.sma(macro_df['close'], length=50)
        sma_200 = ta.sma(macro_df['close'], length=200)
        macro_features['macro_trend'] = ((sma_50 > sma_200).astype(int) * 2 - 1).astype(np.float32)  # +1 or -1
        
        # 3. Macro Volatility (20-period std on H1)
        macro_log_ret = np.log(macro_df['close'] / macro_df['close'].shift(1))
        macro_features['macro_volatility'] = macro_log_ret.rolling(window=20).std().astype(np.float32)
        
        # 4. Macro Momentum (ROC 21 on H1)
        macro_features['macro_momentum'] = ta.roc(macro_df['close'], length=21).astype(np.float32)
        
        # 5. Macro ADX (trend strength on H1)
        adx_macro = ta.adx(macro_df['high'], macro_df['low'], macro_df['close'], length=14)
        if adx_macro is not None:
            macro_features['macro_adx'] = adx_macro['ADX_14'].astype(np.float32)
        
        # 6. Macro MACD
        macd_macro = ta.macd(macro_df['close'])
        if macd_macro is not None:
            macro_features['macro_macd'] = macd_macro['MACD_12_26_9'].astype(np.float32)
            macro_features['macro_macd_hist'] = macd_macro['MACDh_12_26_9'].astype(np.float32)
        
        # 7. Macro Distance from SMA (price position)
        macro_features['macro_dist_sma50'] = ((macro_df['close'] - sma_50) / sma_50).astype(np.float32)
        
        print(f"    Calculated {len(macro_features.columns)} macro features")
        
        # Broadcast macro features to micro timeframe (forward fill)
        # This aligns H1 features to M5 timestamps
        macro_resampled = macro_features.reindex(
            df.index,
            method='ffill',  # Forward fill: each M5 bar gets the most recent H1 value
            limit=12  # Limit forward fill to 12 periods (1 hour for M5 data)
        )
        
        # Fill any remaining NaNs with 0 (neutral)
        macro_resampled = macro_resampled.fillna(0)
        
        # Add to micro dataframe
        for col in macro_resampled.columns:
            df[col] = macro_resampled[col]
        
        n_valid = (~macro_resampled.isna()).sum().sum()
        print(f"    ✓ Injected {len(macro_features.columns)} macro features")
        print(f"    ✓ {n_valid} valid macro values broadcast to micro timeframe")
        
        return df


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


__all__ = [
    "AlphaCouncil",
    "apply_rolling_physics",
    "calculate_fdi",
    "calculate_hurst_rs",
    "calculate_shannon_entropy",
    "TensorFlexFeatureRefiner",
    "cluster_features",
]


def _derive_hurst(features_df: pd.DataFrame, reference_df: Optional[pd.DataFrame]) -> pd.Series:
    price_series = None
    if reference_df is not None and "close" in reference_df.columns:
        price_series = reference_df["close"]
    elif "close" in features_df.columns:
        price_series = features_df["close"]

    if price_series is not None:
        try:
            hurst_vals = ta.hurst(price_series, length=100)
            if hurst_vals is not None:
                return hurst_vals.reindex(features_df.index).fillna(0)
        except Exception:
            pass

    return pd.Series(0.0, index=features_df.index)
