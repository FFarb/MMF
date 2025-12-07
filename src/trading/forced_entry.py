"""
Daily Forced Entry Policy: Ensures at least one trade per day.

This module implements a post-processing policy that prevents "zero-trade days"
by forcing the model to take its best guess when no signals exceed the threshold.

Strategy:
- Group predictions by date
- If no signals on a day: force entry on the highest probability candle
- Track forced vs natural trades for analysis

Author: QFC System v3.1
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Tuple, Dict


def apply_forced_daily_entry(
    predictions_df: pd.DataFrame,
    prob_threshold: float = 0.55,
    timestamp_col: str = 'timestamp',
    probability_col: str = 'probability',
    signal_col: str = 'signal',
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Apply forced daily entry policy to ensure at least one trade per day.
    
    Logic:
    ------
    1. Group predictions by date
    2. Check if any signal on that day exceeds prob_threshold
    3. If sum(signals) == 0 for the day:
       - Find candle with maximum probability
       - Force signal = 1 (even if prob < threshold)
    4. Mark forced trades for analysis
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with columns: [timestamp, probability, signal]
        signal should be binary: 1 (trade), -1 or 0 (no trade)
    prob_threshold : float
        Probability threshold for natural signals (default: 0.55)
    timestamp_col : str
        Name of timestamp column
    probability_col : str
        Name of probability column
    signal_col : str
        Name of signal column
    
    Returns
    -------
    modified_df : pd.DataFrame
        DataFrame with additional 'forced_entry' column (bool)
    stats : dict
        Statistics about forced vs natural trades
        
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'timestamp': pd.date_range('2024-01-01', periods=48, freq='H'),
    ...     'probability': np.random.uniform(0.3, 0.7, 48),
    ...     'signal': np.random.choice([-1, 1], 48)
    ... })
    >>> modified_df, stats = apply_forced_daily_entry(df)
    >>> print(stats)
    {'natural_trades': 15, 'forced_trades': 2, 'total_days': 2}
    """
    
    # Copy to avoid modifying original
    df = predictions_df.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Add date column for grouping
    df['date'] = df[timestamp_col].dt.date
    
    # Initialize forced_entry flag
    df['forced_entry'] = False
    
    # Track statistics
    natural_trades = 0
    forced_trades = 0
    total_days = 0
    
    # Process each day
    for date, day_df in df.groupby('date'):
        total_days += 1
        
        # Check if any natural signals (signal == 1) exist for this day
        natural_signals = (day_df[signal_col] == 1).sum()
        
        if natural_signals > 0:
            # Day has natural trades
            natural_trades += natural_signals
        else:
            # No natural trades - force entry on best candle
            # Find index of maximum probability
            best_idx = day_df[probability_col].idxmax()
            
            # Force signal = 1
            df.loc[best_idx, signal_col] = 1
            df.loc[best_idx, 'forced_entry'] = True
            
            forced_trades += 1
            
            print(f"  [ForcedEntry] {date}: No natural signals, "
                  f"forced entry at {df.loc[best_idx, timestamp_col]} "
                  f"(prob={df.loc[best_idx, probability_col]:.4f})")
    
    # Remove temporary date column
    df = df.drop(columns=['date'])
    
    # Compile statistics
    stats = {
        'natural_trades': int(natural_trades),
        'forced_trades': int(forced_trades),
        'total_days': int(total_days),
        'forced_pct': (forced_trades / total_days * 100) if total_days > 0 else 0.0,
        'avg_trades_per_day': ((natural_trades + forced_trades) / total_days) if total_days > 0 else 0.0,
    }
    
    return df, stats


def analyze_forced_entry_performance(
    predictions_df: pd.DataFrame,
    returns_col: str = 'forward_return',
    forced_col: str = 'forced_entry',
) -> Dict[str, float]:
    """
    Analyze performance difference between natural and forced trades.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with forced_entry flag and returns
    returns_col : str
        Name of forward returns column
    forced_col : str
        Name of forced entry flag column
    
    Returns
    -------
    analysis : dict
        Performance metrics for natural vs forced trades
    """
    
    if forced_col not in predictions_df.columns:
        raise ValueError(f"Column '{forced_col}' not found. Run apply_forced_daily_entry first.")
    
    if returns_col not in predictions_df.columns:
        raise ValueError(f"Column '{returns_col}' not found. Cannot analyze performance.")
    
    # Split into natural and forced trades
    natural_mask = ~predictions_df[forced_col]
    forced_mask = predictions_df[forced_col]
    
    natural_returns = predictions_df.loc[natural_mask, returns_col]
    forced_returns = predictions_df.loc[forced_mask, returns_col]
    
    analysis = {
        'natural_count': len(natural_returns),
        'natural_mean_return': natural_returns.mean() if len(natural_returns) > 0 else 0.0,
        'natural_win_rate': (natural_returns > 0).sum() / len(natural_returns) if len(natural_returns) > 0 else 0.0,
        'forced_count': len(forced_returns),
        'forced_mean_return': forced_returns.mean() if len(forced_returns) > 0 else 0.0,
        'forced_win_rate': (forced_returns > 0).sum() / len(forced_returns) if len(forced_returns) > 0 else 0.0,
    }
    
    # Calculate delta (forced - natural)
    if analysis['natural_count'] > 0 and analysis['forced_count'] > 0:
        analysis['return_delta'] = analysis['forced_mean_return'] - analysis['natural_mean_return']
        analysis['win_rate_delta'] = analysis['forced_win_rate'] - analysis['natural_win_rate']
    else:
        analysis['return_delta'] = 0.0
        analysis['win_rate_delta'] = 0.0
    
    return analysis


def print_forced_entry_report(stats: Dict[str, int], analysis: Dict[str, float] = None):
    """
    Print a formatted report of forced entry statistics.
    
    Parameters
    ----------
    stats : dict
        Statistics from apply_forced_daily_entry
    analysis : dict, optional
        Performance analysis from analyze_forced_entry_performance
    """
    
    print("\n" + "=" * 80)
    print("FORCED DAILY ENTRY POLICY REPORT")
    print("=" * 80)
    
    print(f"\nTrading Activity:")
    print(f"  Total Days: {stats['total_days']}")
    print(f"  Natural Trades: {stats['natural_trades']}")
    print(f"  Forced Trades: {stats['forced_trades']} ({stats['forced_pct']:.1f}% of days)")
    print(f"  Avg Trades/Day: {stats['avg_trades_per_day']:.2f}")
    
    if analysis:
        print(f"\nPerformance Comparison:")
        print(f"  Natural Trades:")
        print(f"    Count: {analysis['natural_count']}")
        print(f"    Mean Return: {analysis['natural_mean_return']:.4f}")
        print(f"    Win Rate: {analysis['natural_win_rate']:.2%}")
        
        print(f"  Forced Trades:")
        print(f"    Count: {analysis['forced_count']}")
        print(f"    Mean Return: {analysis['forced_mean_return']:.4f}")
        print(f"    Win Rate: {analysis['forced_win_rate']:.2%}")
        
        print(f"\n  Delta (Forced - Natural):")
        print(f"    Return: {analysis['return_delta']:+.4f}")
        print(f"    Win Rate: {analysis['win_rate_delta']:+.2%}")
        
        if analysis['return_delta'] > 0:
            print(f"\n  [OK] Forced trades are PROFITABLE (adding alpha)")
        elif analysis['return_delta'] > -0.001:
            print(f"\n  [OK] Forced trades are NEUTRAL (not destroying alpha)")
        else:
            print(f"\n  [WARNING] Forced trades are UNPROFITABLE (destroying alpha)")
    
    print("=" * 80)
