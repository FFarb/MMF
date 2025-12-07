"""
Shared Strategy Logic for Training and Simulation Synchronization.

This module provides the single source of truth for entry decisions,
ensuring training and simulation use identical logic.

Key Feature: "Forced Entry" mode ensures the system always trades
(finds best opportunity even if below threshold).

Author: QFC System v7.0 - Synchronization Patch
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
import numpy as np


@dataclass
class TradeSignal:
    """
    Container for a trade signal.
    
    Attributes
    ----------
    symbol : str
        Asset symbol (e.g., 'BTCUSDT')
    probability : float
        Model probability (0 to 1)
    signal : int
        Signal direction (1 = LONG, -1 = SHORT, 0 = NO TRADE)
    expectancy : float
        Asset expectancy from fleet summary
    forced : bool
        Whether this is a forced entry (below threshold but best available)
    timestamp : pd.Timestamp, optional
        Signal timestamp
    """
    symbol: str
    probability: float
    signal: int
    expectancy: float
    forced: bool = False
    timestamp: Optional[pd.Timestamp] = None


def decide_entry(
    predictions_df: pd.DataFrame,
    fleet_summary: pd.DataFrame,
    threshold: float = 0.55,
    use_forced_entry: bool = True,
    max_positions: int = 5,
) -> List[TradeSignal]:
    """
    Decide which trades to enter based on predictions and fleet summary.
    
    This is the SHARED LOGIC used by both training and simulation.
    
    Logic:
    ------
    1. Filter signals where probability > threshold
    2. If use_forced_entry=True AND no signals found:
       - Find asset with max(probability) across entire fleet
       - Return as single "Forced Entry" trade
    3. Sort by expectancy, limit to max_positions
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions with columns: ['symbol', 'probability', 'signal', 'timestamp']
    fleet_summary : pd.DataFrame
        Fleet summary with columns: ['asset', 'avg_expectancy']
    threshold : float, default=0.55
        Minimum probability to consider a signal
    use_forced_entry : bool, default=True
        If True, always trade (find best opportunity if none above threshold)
    max_positions : int, default=5
        Maximum number of concurrent positions
    
    Returns
    -------
    signals : list of TradeSignal
        Trade signals to execute
    """
    # Merge predictions with expectancy
    merged = predictions_df.merge(
        fleet_summary[['asset', 'avg_expectancy']],
        left_on='symbol',
        right_on='asset',
        how='left'
    )
    
    # Filter valid signals (signal == 1 for LONG)
    valid = merged[merged['signal'] == 1].copy()
    
    if len(valid) == 0:
        # No signals at all
        if use_forced_entry:
            # Find best opportunity (highest probability)
            best_idx = merged['probability'].idxmax()
            best = merged.loc[best_idx]
            
            return [TradeSignal(
                symbol=best['symbol'],
                probability=best['probability'],
                signal=1,  # Force LONG
                expectancy=best.get('avg_expectancy', 0.0),
                forced=True,
                timestamp=best.get('timestamp'),
            )]
        else:
            return []
    
    # Filter by probability threshold
    above_threshold = valid[valid['probability'] >= threshold].copy()
    
    if len(above_threshold) == 0:
        # No signals above threshold
        if use_forced_entry:
            # Find best valid signal (highest probability)
            best_idx = valid['probability'].idxmax()
            best = valid.loc[best_idx]
            
            return [TradeSignal(
                symbol=best['symbol'],
                probability=best['probability'],
                signal=1,
                expectancy=best.get('avg_expectancy', 0.0),
                forced=True,
                timestamp=best.get('timestamp'),
            )]
        else:
            return []
    
    # Sort by expectancy (descending)
    above_threshold = above_threshold.sort_values('avg_expectancy', ascending=False)
    
    # Limit to max positions
    top_signals = above_threshold.head(max_positions)
    
    # Convert to TradeSignal objects
    signals = []
    for _, row in top_signals.iterrows():
        signals.append(TradeSignal(
            symbol=row['symbol'],
            probability=row['probability'],
            signal=row['signal'],
            expectancy=row.get('avg_expectancy', 0.0),
            forced=False,
            timestamp=row.get('timestamp'),
        ))
    
    return signals


def get_latest_predictions(
    predictions_dir: str,
    symbols: List[str],
    timestamp: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Load latest predictions from CSV files.
    
    Parameters
    ----------
    predictions_dir : str
        Directory containing predictions_*.csv files
    symbols : list of str
        List of symbols to load
    timestamp : pd.Timestamp, optional
        Specific timestamp to filter (if None, uses latest)
    
    Returns
    -------
    predictions : pd.DataFrame
        Combined predictions with columns: ['symbol', 'probability', 'signal', 'timestamp']
    """
    from pathlib import Path
    
    predictions_dir = Path(predictions_dir)
    all_predictions = []
    
    for symbol in symbols:
        pred_file = predictions_dir / f"predictions_{symbol}.csv"
        
        if not pred_file.exists():
            continue
        
        df = pd.read_csv(pred_file, parse_dates=['timestamp'])
        
        if timestamp is not None:
            # Filter to specific timestamp
            df = df[df['timestamp'] == timestamp]
        else:
            # Get latest
            df = df.iloc[[-1]]
        
        if len(df) > 0:
            df['symbol'] = symbol
            all_predictions.append(df)
    
    if not all_predictions:
        return pd.DataFrame(columns=['symbol', 'probability', 'signal', 'timestamp'])
    
    return pd.concat(all_predictions, ignore_index=True)


def calculate_sync_score(
    executed_trades: List[str],
    predicted_trades: List[str],
) -> float:
    """
    Calculate synchronization score between executed and predicted trades.
    
    Parameters
    ----------
    executed_trades : list of str
        Symbols that were actually traded
    predicted_trades : list of str
        Symbols that model predicted
    
    Returns
    -------
    sync_score : float
        Synchronization score (0 to 1, 1 = perfect sync)
    """
    executed_set = set(executed_trades)
    predicted_set = set(predicted_trades)
    
    if len(predicted_set) == 0:
        return 1.0 if len(executed_set) == 0 else 0.0
    
    # Jaccard similarity
    intersection = len(executed_set & predicted_set)
    union = len(executed_set | predicted_set)
    
    return intersection / union if union > 0 else 0.0
