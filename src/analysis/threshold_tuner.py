"""
Financial Threshold Optimization using Sharpe Proxy Metric.

This module implements a threshold tuning algorithm that maximizes the Sharpe Proxy,
a risk-adjusted performance metric that accounts for both expectancy and consistency
of trading outcomes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import precision_recall_curve, precision_score, recall_score


def compute_sharpe_proxy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    avg_win_pct: float = 0.02,
    avg_loss_pct: float = -0.01,
) -> float:
    """
    Compute Sharpe Proxy metric for trading performance.
    
    Sharpe Proxy = (Expectancy / StdDev(Outcomes)) * sqrt(N)
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_pred : np.ndarray
        Predicted binary labels (0 or 1).
    avg_win_pct : float, optional
        Average win percentage (default: +2.0%).
    avg_loss_pct : float, optional
        Average loss percentage (default: -1.0%).
        
    Returns
    -------
    float
        Sharpe Proxy value. Higher is better.
    """
    if len(y_pred) == 0 or y_pred.sum() == 0:
        return 0.0
    
    # Compute confusion matrix components
    true_positives = ((y_true == 1) & (y_pred == 1)).sum()
    false_positives = ((y_true == 0) & (y_pred == 1)).sum()
    
    # Total trades (only where we predicted 1)
    n_trades = y_pred.sum()
    
    if n_trades == 0:
        return 0.0
    
    # Win rate
    win_rate = true_positives / n_trades if n_trades > 0 else 0.0
    loss_rate = 1.0 - win_rate
    
    # Expectancy (average return per trade)
    expectancy = (win_rate * avg_win_pct) + (loss_rate * avg_loss_pct)
    
    # Simulate outcomes for standard deviation
    outcomes = []
    for _ in range(int(true_positives)):
        outcomes.append(avg_win_pct)
    for _ in range(int(false_positives)):
        outcomes.append(avg_loss_pct)
    
    if len(outcomes) < 2:
        return 0.0
    
    std_dev = np.std(outcomes)
    
    if std_dev == 0:
        return 0.0
    
    # Sharpe Proxy
    sharpe_proxy = (expectancy / std_dev) * np.sqrt(n_trades)
    
    return sharpe_proxy


def sweep_thresholds(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    threshold_min: float = 0.20,
    threshold_max: float = 0.70,
    threshold_step: float = 0.01,
    avg_win_pct: float = 0.02,
    avg_loss_pct: float = -0.01,
) -> pd.DataFrame:
    """
    Sweep thresholds and compute metrics for each.
    
    Parameters
    ----------
    probabilities : np.ndarray
        Predicted probabilities for class 1.
    y_true : np.ndarray
        True binary labels.
    threshold_min : float, optional
        Minimum threshold to test (default: 0.20).
    threshold_max : float, optional
        Maximum threshold to test (default: 0.70).
    threshold_step : float, optional
        Step size for threshold sweep (default: 0.01).
    avg_win_pct : float, optional
        Average win percentage (default: +2.0%).
    avg_loss_pct : float, optional
        Average loss percentage (default: -1.0%).
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: threshold, precision, recall, f1, sharpe_proxy,
        expectancy, n_trades.
    """
    thresholds = np.arange(threshold_min, threshold_max + threshold_step, threshold_step)
    
    results = []
    
    for threshold in thresholds:
        y_pred = (probabilities >= threshold).astype(int)
        
        # Skip if no predictions
        if y_pred.sum() == 0:
            results.append({
                'threshold': threshold,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'sharpe_proxy': 0.0,
                'expectancy': 0.0,
                'n_trades': 0,
            })
            continue
        
        # Compute metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        sharpe_proxy = compute_sharpe_proxy(y_true, y_pred, avg_win_pct, avg_loss_pct)
        
        # Compute expectancy
        n_trades = y_pred.sum()
        win_rate = precision  # Precision = TP / (TP + FP)
        expectancy = (win_rate * avg_win_pct) + ((1 - win_rate) * avg_loss_pct)
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'sharpe_proxy': sharpe_proxy,
            'expectancy': expectancy,
            'n_trades': int(n_trades),
        })
    
    return pd.DataFrame(results)


def plot_threshold_analysis(
    results: pd.DataFrame,
    probabilities: np.ndarray,
    y_true: np.ndarray,
    output_path: Path,
) -> None:
    """
    Create Plotly visualizations for threshold analysis.
    
    Parameters
    ----------
    results : pd.DataFrame
        Results from sweep_thresholds().
    probabilities : np.ndarray
        Predicted probabilities for class 1.
    y_true : np.ndarray
        True binary labels.
    output_path : Path
        Path to save the HTML visualization.
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Sharpe Proxy vs Threshold',
            'Precision-Recall Curve',
            'Expectancy vs Threshold',
            'Number of Trades vs Threshold',
        ),
        specs=[[{}, {}], [{}, {}]],
    )
    
    # 1. Sharpe Proxy vs Threshold
    fig.add_trace(
        go.Scatter(
            x=results['threshold'],
            y=results['sharpe_proxy'],
            mode='lines+markers',
            name='Sharpe Proxy',
            line=dict(color='cyan', width=2),
            marker=dict(size=4),
        ),
        row=1, col=1,
    )
    
    # Mark optimal threshold
    optimal_idx = results['sharpe_proxy'].idxmax()
    optimal_threshold = results.loc[optimal_idx, 'threshold']
    optimal_sharpe = results.loc[optimal_idx, 'sharpe_proxy']
    
    fig.add_trace(
        go.Scatter(
            x=[optimal_threshold],
            y=[optimal_sharpe],
            mode='markers',
            name=f'Optimal (T={optimal_threshold:.2f})',
            marker=dict(color='red', size=12, symbol='star'),
        ),
        row=1, col=1,
    )
    
    # 2. Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, probabilities)
    
    fig.add_trace(
        go.Scatter(
            x=recall_vals,
            y=precision_vals,
            mode='lines',
            name='PR Curve',
            line=dict(color='lime', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.1)',
        ),
        row=1, col=2,
    )
    
    # 3. Expectancy vs Threshold
    fig.add_trace(
        go.Scatter(
            x=results['threshold'],
            y=results['expectancy'] * 100,  # Convert to percentage
            mode='lines+markers',
            name='Expectancy (%)',
            line=dict(color='gold', width=2),
            marker=dict(size=4),
        ),
        row=2, col=1,
    )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3, row=2, col=1)
    
    # 4. Number of Trades vs Threshold
    fig.add_trace(
        go.Scatter(
            x=results['threshold'],
            y=results['n_trades'],
            mode='lines+markers',
            name='# Trades',
            line=dict(color='orange', width=2),
            marker=dict(size=4),
        ),
        row=2, col=2,
    )
    
    # Update layout
    fig.update_xaxes(title_text="Threshold", row=1, col=1)
    fig.update_yaxes(title_text="Sharpe Proxy", row=1, col=1)
    
    fig.update_xaxes(title_text="Recall", row=1, col=2)
    fig.update_yaxes(title_text="Precision", row=1, col=2)
    
    fig.update_xaxes(title_text="Threshold", row=2, col=1)
    fig.update_yaxes(title_text="Expectancy (%)", row=2, col=1)
    
    fig.update_xaxes(title_text="Threshold", row=2, col=2)
    fig.update_yaxes(title_text="Number of Trades", row=2, col=2)
    
    fig.update_layout(
        title_text="Financial Threshold Optimization Analysis",
        template='plotly_dark',
        height=800,
        showlegend=True,
    )
    
    # Save
    fig.write_html(str(output_path))


def run_tuning(
    validation_path: Path,
    output_dir: Path,
    avg_win_pct: float = 0.02,
    avg_loss_pct: float = -0.01,
) -> Dict[str, float]:
    """
    Run threshold tuning on validation predictions.
    
    Parameters
    ----------
    validation_path : Path
        Path to validation predictions parquet file with columns:
        'probability' and 'target'.
    output_dir : Path
        Directory to save output artifacts.
    avg_win_pct : float, optional
        Average win percentage (default: +2.0%).
    avg_loss_pct : float, optional
        Average loss percentage (default: -1.0%).
        
    Returns
    -------
    Dict[str, float]
        Dictionary with keys: 'optimal_threshold', 'optimal_sharpe_proxy',
        'precision', 'recall', 'expectancy', 'n_trades'.
    """
    # Load validation predictions
    df = pd.read_parquet(validation_path)
    
    if 'probability' not in df.columns or 'target' not in df.columns:
        raise ValueError(
            f"Validation file must contain 'probability' and 'target' columns. "
            f"Found: {df.columns.tolist()}"
        )
    
    probabilities = df['probability'].values
    y_true = df['target'].values
    
    print("\n" + "=" * 72)
    print("                  FINANCIAL THRESHOLD OPTIMIZATION")
    print("=" * 72)
    print(f"Validation samples: {len(y_true):,}")
    print(f"Positive class rate: {y_true.mean():.2%}")
    print(f"Assumptions: Avg Win = {avg_win_pct:+.2%}, Avg Loss = {avg_loss_pct:+.2%}")
    
    # Sweep thresholds
    print("\nSweeping thresholds from 0.20 to 0.70...")
    results = sweep_thresholds(
        probabilities, y_true,
        threshold_min=0.20,
        threshold_max=0.70,
        threshold_step=0.01,
        avg_win_pct=avg_win_pct,
        avg_loss_pct=avg_loss_pct,
    )
    
    # Find optimal threshold
    optimal_idx = results['sharpe_proxy'].idxmax()
    optimal_row = results.loc[optimal_idx]
    
    print("\n" + "-" * 72)
    print("OPTIMAL THRESHOLD FOUND")
    print("-" * 72)
    print(f"  Threshold        : {optimal_row['threshold']:.2f}")
    print(f"  Sharpe Proxy     : {optimal_row['sharpe_proxy']:.4f}")
    print(f"  Precision        : {optimal_row['precision']:.2%}")
    print(f"  Recall           : {optimal_row['recall']:.2%}")
    print(f"  F1 Score         : {optimal_row['f1']:.4f}")
    print(f"  Expectancy       : {optimal_row['expectancy']:+.2%}")
    print(f"  Trades Generated : {optimal_row['n_trades']:,}")
    
    # Save results
    output_dir.mkdir(exist_ok=True, parents=True)
    results_path = output_dir / "threshold_sweep_results.csv"
    results.to_csv(results_path, index=False)
    print(f"\nSaved sweep results to: {results_path}")
    
    # Generate visualizations
    viz_path = output_dir / "threshold_optimization.html"
    plot_threshold_analysis(results, probabilities, y_true, viz_path)
    print(f"Saved visualization to: {viz_path}")
    
    return {
        'optimal_threshold': float(optimal_row['threshold']),
        'optimal_sharpe_proxy': float(optimal_row['sharpe_proxy']),
        'precision': float(optimal_row['precision']),
        'recall': float(optimal_row['recall']),
        'expectancy': float(optimal_row['expectancy']),
        'n_trades': int(optimal_row['n_trades']),
    }


__all__ = ["run_tuning", "compute_sharpe_proxy", "sweep_thresholds"]
