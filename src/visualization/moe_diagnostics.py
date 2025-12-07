"""
Diagnostic & Visualization System for Bicameral MoE.

This module provides comprehensive visualization and diagnostics for the
Mixture of Experts ensemble, enabling deep insights into expert behavior,
regime switching, and market dynamics before transitioning to Neural ODEs.

Key Features:
- Trade execution visualization with expert activation
- Regime switching heatmap (gating network probabilities)
- Phase portrait for ODE preparation
- Expert performance analytics
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns


@dataclass
class DiagnosticReport:
    """Container for diagnostic data from a single fold."""
    fold_id: int
    asset: str
    timestamps: pd.DatetimeIndex
    prices: np.ndarray
    predictions: np.ndarray
    true_labels: np.ndarray
    expert_weights: np.ndarray  # Shape: (n_samples, n_experts)
    expert_names: List[str]
    frac_diff: Optional[np.ndarray] = None
    volatility: Optional[np.ndarray] = None
    physics_features: Optional[pd.DataFrame] = None


class VisualReporter:
    """
    Comprehensive visualization system for MoE diagnostics.
    
    Generates three critical plots:
    1. Trade Execution Plot: Candlestick + markers + expert background
    2. Regime Heatmap: Stacked area showing gating probabilities over time
    3. Phase Portrait: FracDiff vs Volatility (prep for Neural ODE)
    
    Parameters
    ----------
    output_dir : Path
        Directory to save plots
    expert_colors : dict, optional
        Mapping of expert names to colors
    """
    
    def __init__(
        self,
        output_dir: Path = Path("artifacts/diagnostics"),
        expert_colors: Optional[Dict[str, str]] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default expert colors
        self.expert_colors = expert_colors or {
            'Trend': '#2E7D32',      # Dark Green
            'Range': '#1976D2',      # Blue
            'Stress': '#C62828',     # Red
            'Elastic': '#F57C00',    # Orange
            'Pattern': '#7B1FA2',    # Purple
        }
    
    def generate_full_report(
        self,
        report: DiagnosticReport,
        show_plots: bool = False,
    ) -> Dict[str, Path]:
        """
        Generate all diagnostic plots for a single fold.
        
        Parameters
        ----------
        report : DiagnosticReport
            Diagnostic data from fold
        show_plots : bool
            Whether to display plots (default: False, just save)
        
        Returns
        -------
        plot_paths : dict
            Mapping of plot type to file path
        """
        print(f"\n[DIAGNOSTICS] Generating visual report for {report.asset} - Fold {report.fold_id}")
        
        plot_paths = {}
        
        # Plot 1: Trade Execution with Expert Background
        try:
            path = self.plot_trade_execution(report)
            plot_paths['trade_execution'] = path
            print(f"  ✓ Trade execution plot: {path}")
        except Exception as e:
            print(f"  ✗ Trade execution plot failed: {e}")
        
        # Plot 2: Regime Switching Heatmap
        try:
            path = self.plot_regime_heatmap(report)
            plot_paths['regime_heatmap'] = path
            print(f"  ✓ Regime heatmap: {path}")
        except Exception as e:
            print(f"  ✗ Regime heatmap failed: {e}")
        
        # Plot 3: Phase Portrait (if data available)
        if report.frac_diff is not None and report.volatility is not None:
            try:
                path = self.plot_phase_portrait(report)
                plot_paths['phase_portrait'] = path
                print(f"  ✓ Phase portrait: {path}")
            except Exception as e:
                print(f"  ✗ Phase portrait failed: {e}")
        
        # Plot 4: Expert Performance Summary
        try:
            path = self.plot_expert_performance(report)
            plot_paths['expert_performance'] = path
            print(f"  ✓ Expert performance: {path}")
        except Exception as e:
            print(f"  ✗ Expert performance failed: {e}")
        
        if show_plots:
            plt.show()
        
        return plot_paths
    
    def plot_trade_execution(self, report: DiagnosticReport) -> Path:
        """
        Plot 1: Trade execution with expert activation background.
        
        Shows:
        - Price candlesticks (or line if OHLC not available)
        - Buy/Sell markers (green/red)
        - Background color indicating dominant expert
        """
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Plot price
        ax.plot(report.timestamps, report.prices, 
               color='black', linewidth=1.5, alpha=0.8, label='Price')
        
        # Determine dominant expert for each timestamp
        dominant_expert_idx = np.argmax(report.expert_weights, axis=1)
        
        # Color background by dominant expert
        for i in range(len(report.timestamps) - 1):
            expert_idx = dominant_expert_idx[i]
            expert_name = report.expert_names[expert_idx]
            color = self.expert_colors.get(expert_name, '#CCCCCC')
            
            ax.axvspan(
                report.timestamps[i],
                report.timestamps[i + 1],
                alpha=0.15,
                color=color,
                linewidth=0,
            )
        
        # Plot trade markers
        buy_mask = report.predictions == 1
        sell_mask = report.predictions == 0
        
        # Correct predictions (green)
        correct_buy = buy_mask & (report.true_labels == 1)
        correct_sell = sell_mask & (report.true_labels == 0)
        
        # Incorrect predictions (red)
        incorrect_buy = buy_mask & (report.true_labels == 0)
        incorrect_sell = sell_mask & (report.true_labels == 1)
        
        # Plot markers
        if correct_buy.sum() > 0:
            ax.scatter(report.timestamps[correct_buy], report.prices[correct_buy],
                      marker='^', s=100, color='green', alpha=0.7, 
                      edgecolors='darkgreen', linewidths=1, label='Correct Buy', zorder=5)
        
        if correct_sell.sum() > 0:
            ax.scatter(report.timestamps[correct_sell], report.prices[correct_sell],
                      marker='v', s=100, color='lightgreen', alpha=0.7,
                      edgecolors='darkgreen', linewidths=1, label='Correct Sell', zorder=5)
        
        if incorrect_buy.sum() > 0:
            ax.scatter(report.timestamps[incorrect_buy], report.prices[incorrect_buy],
                      marker='^', s=100, color='red', alpha=0.7,
                      edgecolors='darkred', linewidths=1, label='Incorrect Buy', zorder=5)
        
        if incorrect_sell.sum() > 0:
            ax.scatter(report.timestamps[incorrect_sell], report.prices[incorrect_sell],
                      marker='v', s=100, color='lightcoral', alpha=0.7,
                      edgecolors='darkred', linewidths=1, label='Incorrect Sell', zorder=5)
        
        # Create expert legend
        expert_patches = [
            mpatches.Patch(color=self.expert_colors.get(name, '#CCCCCC'), 
                          label=f'{name} Expert', alpha=0.3)
            for name in report.expert_names
        ]
        
        # Add legends
        ax.legend(loc='upper left', fontsize=9)
        ax.add_artist(plt.legend(handles=expert_patches, loc='upper right', 
                                title='Expert Background', fontsize=9))
        
        ax.set_title(f'Trade Execution - {report.asset} (Fold {report.fold_id})', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Price ($)', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save
        filename = f"trade_execution_{report.asset}_fold{report.fold_id}.png"
        path = self.output_dir / filename
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return path
    
    def plot_regime_heatmap(self, report: DiagnosticReport) -> Path:
        """
        Plot 2: Regime switching heatmap (stacked area chart).
        
        Shows gating network probability distribution over time.
        This reveals WHEN the system switches from Trend to Mean Reversion, etc.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Stacked area chart of expert probabilities
        colors = [self.expert_colors.get(name, '#CCCCCC') for name in report.expert_names]
        
        ax1.stackplot(
            report.timestamps,
            report.expert_weights.T,
            labels=report.expert_names,
            colors=colors,
            alpha=0.7,
        )
        
        ax1.set_title(f'Expert Activation Over Time - {report.asset} (Fold {report.fold_id})', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Expert Probability', fontsize=11)
        ax1.set_ylim([0, 1])
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Bottom panel: Dominant expert timeline
        dominant_expert_idx = np.argmax(report.expert_weights, axis=1)
        
        # Create color array for dominant expert
        dominant_colors = [colors[idx] for idx in dominant_expert_idx]
        
        # Plot as colored bars
        for i in range(len(report.timestamps) - 1):
            ax2.axvspan(
                report.timestamps[i],
                report.timestamps[i + 1],
                color=dominant_colors[i],
                alpha=0.8,
            )
        
        ax2.set_title('Dominant Expert Timeline', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=11)
        ax2.set_yticks([])
        ax2.set_ylim([0, 1])
        
        # Format x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save
        filename = f"regime_heatmap_{report.asset}_fold{report.fold_id}.png"
        path = self.output_dir / filename
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return path
    
    def plot_phase_portrait(self, report: DiagnosticReport) -> Path:
        """
        Plot 3: Phase portrait (FracDiff vs Volatility).
        
        Preparation for Neural ODE implementation.
        Shows the "shape" of market state space and regime boundaries.
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Determine dominant expert for coloring
        dominant_expert_idx = np.argmax(report.expert_weights, axis=1)
        
        # Create scatter plot colored by dominant expert
        for expert_idx, expert_name in enumerate(report.expert_names):
            mask = dominant_expert_idx == expert_idx
            
            if mask.sum() > 0:
                color = self.expert_colors.get(expert_name, '#CCCCCC')
                ax.scatter(
                    report.frac_diff[mask],
                    report.volatility[mask],
                    c=color,
                    label=expert_name,
                    alpha=0.6,
                    s=50,
                    edgecolors='black',
                    linewidths=0.5,
                )
        
        # Add trajectory lines (time evolution)
        ax.plot(report.frac_diff, report.volatility, 
               color='gray', alpha=0.2, linewidth=0.5, zorder=0)
        
        # Add origin lines
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
        
        ax.set_title(f'Phase Portrait: Market State Space - {report.asset} (Fold {report.fold_id})', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Fractional Differentiation (FracDiff)', fontsize=11)
        ax.set_ylabel('Volatility', fontsize=11)
        ax.legend(title='Dominant Expert', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add annotations
        ax.text(0.02, 0.98, 'Neural ODE Prep:\nState = (FracDiff, Vol)\ndState/dt = f(State, θ)', 
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save
        filename = f"phase_portrait_{report.asset}_fold{report.fold_id}.png"
        path = self.output_dir / filename
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return path
    
    def plot_expert_performance(self, report: DiagnosticReport) -> Path:
        """
        Plot 4: Expert performance analytics.
        
        Shows:
        - Activation frequency per expert
        - Accuracy when each expert is dominant
        - Average confidence per expert
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        dominant_expert_idx = np.argmax(report.expert_weights, axis=1)
        
        # Panel 1: Activation frequency
        ax = axes[0, 0]
        activation_counts = np.bincount(dominant_expert_idx, minlength=len(report.expert_names))
        activation_pct = activation_counts / len(dominant_expert_idx) * 100
        
        colors = [self.expert_colors.get(name, '#CCCCCC') for name in report.expert_names]
        bars = ax.bar(report.expert_names, activation_pct, color=colors, alpha=0.7, edgecolor='black')
        ax.set_title('Expert Activation Frequency', fontsize=12, fontweight='bold')
        ax.set_ylabel('Activation %', fontsize=10)
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, activation_pct):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Panel 2: Accuracy when dominant
        ax = axes[0, 1]
        accuracies = []
        for expert_idx in range(len(report.expert_names)):
            mask = dominant_expert_idx == expert_idx
            if mask.sum() > 0:
                accuracy = (report.predictions[mask] == report.true_labels[mask]).mean() * 100
            else:
                accuracy = 0
            accuracies.append(accuracy)
        
        bars = ax.bar(report.expert_names, accuracies, color=colors, alpha=0.7, edgecolor='black')
        ax.set_title('Accuracy When Dominant', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy %', fontsize=10)
        ax.set_ylim([0, 100])
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        # Add value labels
        for bar, val in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Panel 3: Average confidence
        ax = axes[1, 0]
        avg_confidence = report.expert_weights.mean(axis=0) * 100
        
        bars = ax.bar(report.expert_names, avg_confidence, color=colors, alpha=0.7, edgecolor='black')
        ax.set_title('Average Expert Confidence', fontsize=12, fontweight='bold')
        ax.set_ylabel('Avg Weight %', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, avg_confidence):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Panel 4: Expert weight distribution (box plot)
        ax = axes[1, 1]
        box_data = [report.expert_weights[:, i] * 100 for i in range(len(report.expert_names))]
        bp = ax.boxplot(box_data, labels=report.expert_names, patch_artist=True)
        
        # Color boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title('Expert Weight Distribution', fontsize=12, fontweight='bold')
        ax.set_ylabel('Weight %', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'Expert Performance Analytics - {report.asset} (Fold {report.fold_id})', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Save
        filename = f"expert_performance_{report.asset}_fold{report.fold_id}.png"
        path = self.output_dir / filename
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return path


def create_diagnostic_report(
    fold_id: int,
    asset: str,
    timestamps: pd.DatetimeIndex,
    prices: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    y_pred: np.ndarray,
    expert_weights: np.ndarray,
    expert_names: List[str],
) -> DiagnosticReport:
    """
    Helper function to create DiagnosticReport from fold results.
    
    Parameters
    ----------
    fold_id : int
        Fold number
    asset : str
        Asset symbol
    timestamps : pd.DatetimeIndex
        Validation timestamps
    prices : np.ndarray
        Price data
    X_val : pd.DataFrame
        Validation features
    y_val : np.ndarray
        True labels
    y_pred : np.ndarray
        Predictions
    expert_weights : np.ndarray
        Expert gating weights (n_samples, n_experts)
    expert_names : list
        Names of experts
    
    Returns
    -------
    report : DiagnosticReport
        Diagnostic report ready for visualization
    """
    # Extract frac_diff and volatility if available
    frac_diff = X_val['frac_diff'].values if 'frac_diff' in X_val.columns else None
    
    # Calculate volatility from price if not in features
    if 'volatility' in X_val.columns:
        volatility = X_val['volatility'].values
    elif len(prices) > 20:
        # Rolling volatility
        returns = np.diff(np.log(prices))
        volatility = np.concatenate([[np.nan], pd.Series(returns).rolling(20).std().values])
    else:
        volatility = None
    
    return DiagnosticReport(
        fold_id=fold_id,
        asset=asset,
        timestamps=timestamps,
        prices=prices,
        predictions=y_pred,
        true_labels=y_val,
        expert_weights=expert_weights,
        expert_names=expert_names,
        frac_diff=frac_diff,
        volatility=volatility,
        physics_features=X_val[[c for c in X_val.columns if c.startswith('stability_')]] if any(c.startswith('stability_') for c in X_val.columns) else None,
    )


__all__ = [
    'VisualReporter',
    'DiagnosticReport',
    'create_diagnostic_report',
]
