"""
Microstructure Features: Extract M5 "hints" for H1 training.

This module calculates aggregated M5 statistics within each H1 bar to capture
internal bar dynamics without the noise of direct M5 training.

Strategy: "H1 is Main, M5 is Hint"
- Train on H1 targets (stable, high precision)
- Input features include M5 microstructure (volatility, efficiency, noise)
- Get best of both worlds: H1 stability + M5 insights
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def calc_microstructure_features(
    m5_df: pd.DataFrame,
    h1_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Calculate microstructure features from M5 data aligned to H1 bars.
    
    This provides an "X-ray" view inside each H1 bar by aggregating M5 statistics.
    
    Parameters
    ----------
    m5_df : pd.DataFrame
        5-minute OHLCV data with datetime index
    h1_index : pd.DatetimeIndex
        H1 bar timestamps to align features to
    
    Returns
    -------
    pd.DataFrame
        Microstructure features indexed by H1 timestamps
        
    Features
    --------
    micro_volatility : float
        Standard deviation of M5 returns within the hour
        Measures internal volatility/choppiness
        
    micro_efficiency : float
        Kaufman Efficiency Ratio: Abs(H1_move) / Sum(Abs(M5_moves))
        1.0 = smooth trend, 0.0 = total chop
        
    micro_buying_pressure : float
        Ratio of up-volume to down-volume on M5 within hour
        >1.0 = buying dominates, <1.0 = selling dominates
        
    micro_max_drawdown : float
        Largest drop from high to low within the hour
        Measures internal weakness
        
    micro_range_ratio : float
        H1 range / Sum(M5 ranges)
        Measures gap efficiency
        
    micro_trend_consistency : float
        Fraction of M5 bars moving in H1 direction
        1.0 = all M5 bars aligned, 0.0 = all opposed
    """
    print(f"[MICROSTRUCTURE] Calculating features from {len(m5_df)} M5 bars")
    
    # Ensure datetime index
    if not isinstance(m5_df.index, pd.DatetimeIndex):
        if 'timestamp' in m5_df.columns:
            m5_df = m5_df.set_index('timestamp')
        else:
            raise ValueError("M5 data must have datetime index or 'timestamp' column")
    
    # Prepare M5 data
    m5_df = m5_df.copy()
    m5_df['m5_return'] = m5_df['close'].pct_change()
    m5_df['m5_range'] = m5_df['high'] - m5_df['low']
    m5_df['m5_body'] = np.abs(m5_df['close'] - m5_df['open'])
    
    # Classify M5 bars as up/down
    m5_df['m5_direction'] = (m5_df['close'] > m5_df['open']).astype(int)
    m5_df['up_volume'] = m5_df['volume'] * m5_df['m5_direction']
    m5_df['down_volume'] = m5_df['volume'] * (1 - m5_df['m5_direction'])
    
    # Resample to H1 (align with H1 index)
    # Use 'right' label to match H1 bar closing time
    resampler = m5_df.resample('1H', label='right', closed='right')
    
    micro_features = pd.DataFrame(index=h1_index)
    
    print("  [1/6] Calculating micro_volatility...")
    # 1. Micro Volatility (std of M5 returns within hour)
    micro_vol = resampler['m5_return'].std()
    micro_features['micro_volatility'] = micro_vol.reindex(h1_index).fillna(0).astype(np.float32)
    
    print("  [2/6] Calculating micro_efficiency...")
    # 2. Micro Efficiency (Kaufman Efficiency Ratio)
    # Efficiency = Net_Move / Sum_Abs_Moves
    def calc_efficiency(group):
        if len(group) < 2:
            return 0.0
        
        # Net move (H1 equivalent)
        net_move = abs(group['close'].iloc[-1] - group['open'].iloc[0])
        
        # Sum of absolute M5 moves
        sum_abs_moves = group['m5_body'].sum()
        
        if sum_abs_moves < 1e-9:
            return 0.0
        
        efficiency = net_move / sum_abs_moves
        return min(efficiency, 1.0)  # Cap at 1.0
    
    micro_eff = resampler.apply(calc_efficiency)
    micro_features['micro_efficiency'] = micro_eff.reindex(h1_index).fillna(0).astype(np.float32)
    
    print("  [3/6] Calculating micro_buying_pressure...")
    # 3. Micro Buying Pressure (up-volume / down-volume ratio)
    up_vol_sum = resampler['up_volume'].sum()
    down_vol_sum = resampler['down_volume'].sum()
    
    # Avoid division by zero
    buying_pressure = (up_vol_sum + 1) / (down_vol_sum + 1)
    micro_features['micro_buying_pressure'] = buying_pressure.reindex(h1_index).fillna(1.0).astype(np.float32)
    
    print("  [4/6] Calculating micro_max_drawdown...")
    # 4. Micro Max Drawdown (largest drop within hour)
    def calc_max_dd(group):
        if len(group) < 2:
            return 0.0
        
        # Rolling max
        rolling_max = group['high'].expanding().max()
        
        # Drawdown from rolling max
        drawdowns = (rolling_max - group['low']) / (rolling_max + 1e-9)
        
        return drawdowns.max()
    
    micro_dd = resampler.apply(calc_max_dd)
    micro_features['micro_max_drawdown'] = micro_dd.reindex(h1_index).fillna(0).astype(np.float32)
    
    print("  [5/6] Calculating micro_range_ratio...")
    # 5. Micro Range Ratio (H1 range / sum of M5 ranges)
    def calc_range_ratio(group):
        if len(group) < 2:
            return 0.0
        
        h1_range = group['high'].max() - group['low'].min()
        m5_range_sum = group['m5_range'].sum()
        
        if m5_range_sum < 1e-9:
            return 0.0
        
        return h1_range / m5_range_sum
    
    micro_rr = resampler.apply(calc_range_ratio)
    micro_features['micro_range_ratio'] = micro_rr.reindex(h1_index).fillna(0).astype(np.float32)
    
    print("  [6/6] Calculating micro_trend_consistency...")
    # 6. Micro Trend Consistency (fraction of M5 bars aligned with H1 direction)
    def calc_trend_consistency(group):
        if len(group) < 2:
            return 0.5
        
        # H1 direction
        h1_direction = 1 if group['close'].iloc[-1] > group['open'].iloc[0] else 0
        
        # Fraction of M5 bars aligned
        aligned = (group['m5_direction'] == h1_direction).sum()
        total = len(group)
        
        return aligned / total
    
    micro_tc = resampler.apply(calc_trend_consistency)
    micro_features['micro_trend_consistency'] = micro_tc.reindex(h1_index).fillna(0.5).astype(np.float32)
    
    # Summary
    n_valid = (~micro_features.isna()).sum().sum()
    n_total = len(micro_features) * len(micro_features.columns)
    
    print(f"[MICROSTRUCTURE] Complete:")
    print(f"  Features: {len(micro_features.columns)}")
    print(f"  H1 bars: {len(micro_features)}")
    print(f"  Valid values: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
    
    return micro_features


def summarize_microstructure(micro_df: pd.DataFrame) -> None:
    """Print summary statistics of microstructure features."""
    print("\n" + "=" * 72)
    print("MICROSTRUCTURE FEATURE SUMMARY")
    print("=" * 72)
    
    for col in micro_df.columns:
        values = micro_df[col].dropna()
        if len(values) > 0:
            print(f"\n{col}:")
            print(f"  Mean: {values.mean():.4f}")
            print(f"  Std:  {values.std():.4f}")
            print(f"  Min:  {values.min():.4f}")
            print(f"  Max:  {values.max():.4f}")
            print(f"  25%:  {values.quantile(0.25):.4f}")
            print(f"  50%:  {values.quantile(0.50):.4f}")
            print(f"  75%:  {values.quantile(0.75):.4f}")


__all__ = [
    "calc_microstructure_features",
    "summarize_microstructure",
]
