"""
Advanced evaluation metrics for the Deep Quant Pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

def profit_weighted_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    returns: np.ndarray,
    tp_pct: float,
    sl_pct: float
) -> pd.DataFrame:
    """
    Calculate a confusion matrix where values are Realized PnL instead of counts.
    
    Args:
        y_true: True labels (1=Win, 0=Loss)
        y_pred: Predicted labels (1=Trade, 0=No Trade)
        returns: Actual returns of the trade (if available) or we estimate based on TP/SL.
                 If returns is None, we assume:
                 - TP (True Positive): +TP_PCT
                 - FP (False Positive): -SL_PCT
                 - TN/FN: 0
    
    Returns:
        DataFrame with PnL for TP, FP, TN, FN.
    """
    # Identify quadrants
    tp_mask = (y_true == 1) & (y_pred == 1)
    fp_mask = (y_true == 0) & (y_pred == 1)
    tn_mask = (y_true == 0) & (y_pred == 0)
    fn_mask = (y_true == 1) & (y_pred == 0)
    
    # Calculate PnL
    # If we have exact returns per trade, use them. 
    # Otherwise use the theoretical TP/SL.
    # Note: 'returns' array here should ideally be the realized return for that bar's signal.
    # If we don't have it, we estimate.
    
    pnl_vector = np.zeros_like(y_true, dtype=float)
    
    # For TP (We predicted Win, and it was a Win) -> We gain TP_PCT
    pnl_vector[tp_mask] = tp_pct
    
    # For FP (We predicted Win, but it was a Loss) -> We lose SL_PCT
    pnl_vector[fp_mask] = -sl_pct
    
    # For TN/FN (We didn't trade) -> 0
    
    # If 'returns' is provided and not empty, we could use it for more precision, 
    # but y_true is already binarized based on TP/SL. 
    # So using fixed TP/SL is consistent with the labeling.
    
    total_pnl = np.sum(pnl_vector)
    
    matrix = pd.DataFrame({
        "Predicted_Win": [np.sum(pnl_vector[tp_mask]), np.sum(pnl_vector[fp_mask])],
        "Predicted_NoTrade": [0.0, 0.0]
    }, index=["Actual_Win", "Actual_Loss"])
    
    return matrix

def check_mae_mfe(
    entry_price: float,
    high_window: np.ndarray,
    low_window: np.ndarray,
    tp_pct: float,
    sl_pct: float
) -> Tuple[bool, float, float]:
    """
    Analyze Maximum Adverse Excursion (MAE) and Maximum Favorable Excursion (MFE).
    
    Returns:
        (is_win, mae_pct, mfe_pct)
        is_win: True if TP hit before SL.
        mae_pct: Max loss % experienced during the trade.
        mfe_pct: Max profit % experienced during the trade.
    """
    tp_price = entry_price * (1 + tp_pct)
    sl_price = entry_price * (1 - sl_pct)
    
    mae_price = entry_price
    mfe_price = entry_price
    
    outcome = False # Default to loss if time runs out? Or 0.
    
    for h, l in zip(high_window, low_window):
        # Update extremes
        if l < mae_price:
            mae_price = l
        if h > mfe_price:
            mfe_price = h
            
        # Check exit
        if l <= sl_price:
            outcome = False
            break
        if h >= tp_price:
            outcome = True
            break
            
    mae_pct = (mae_price - entry_price) / entry_price
    mfe_pct = (mfe_price - entry_price) / entry_price
    
    return outcome, mae_pct, mfe_pct
