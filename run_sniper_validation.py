"""
Sniper Validation with Trained Fleet Models.

Enhanced version with:
- All assets mode
- Validation period control (last N days)
- Detailed trade visualization
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent))

from src.data_loader import MarketDataLoader
from src.trading.sniper_engine import (
    SniperBacktestEngine,
    TradingConfig,
    calculate_entry_improvement,
    Trade,
)

# Fleet assets
FLEET_ASSETS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
    'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'LTCUSDT',
]


def load_fleet_predictions(asset: str, validation_days: int = 0) -> pd.DataFrame:
    """
    Load H1 predictions from trained hierarchical fleet.
    
    Parameters
    ----------
    asset : str
        Asset symbol
    validation_days : int
        If > 0, only load last N days for validation
    
    Returns
    -------
    pd.DataFrame
        H1 predictions with columns: ['timestamp', 'signal', 'close', 'probability']
    """
    print(f"[FLEET] Loading trained model predictions for {asset}...")
    
    predictions_file = Path(f"artifacts/fleet_predictions_{asset}.csv")
    
    if not predictions_file.exists():
        print(f"  ERROR: No predictions found at {predictions_file}")
        print(f"  Please run hierarchical fleet training first:")
        print(f"    python run_enriched_fleet.py --h1-days 730 --m5-days 150")
        raise FileNotFoundError(f"Fleet predictions not found: {predictions_file}")
    
    print(f"  Loading from: {predictions_file}")
    df_pred = pd.read_csv(predictions_file)
    df_pred['timestamp'] = pd.to_datetime(df_pred['timestamp'])
    
    # Filter to validation period if specified
    if validation_days > 0:
        cutoff_date = df_pred['timestamp'].max() - pd.Timedelta(days=validation_days)
        df_pred = df_pred[df_pred['timestamp'] > cutoff_date].copy()
        print(f"  Filtered to last {validation_days} days: {len(df_pred)} predictions")
        print(f"  Validation period: {df_pred['timestamp'].min().date()} to {df_pred['timestamp'].max().date()}")
    else:
        print(f"  Loaded {len(df_pred)} H1 predictions (all available)")
    
    print(f"  Signals: {(df_pred['signal'] == 1).sum()} long, {(df_pred['signal'] == -1).sum()} short")
    
    return df_pred


def run_sniper_validation(
    asset: str = 'BTCUSDT',
    leverage: float = 5.0,
    ou_threshold: float = 1.0,
    validation_days: int = 0,
) -> dict:
    """
    Validate sniper execution using trained fleet predictions.
    
    Parameters
    ----------
    asset : str
        Asset symbol
    leverage : float
        Leverage (max 10x)
    ou_threshold : float
        OU Z-score threshold for entry
    validation_days : int
        If > 0, only validate on last N days
    
    Returns
    -------
    dict
        Results from naive vs sniper comparison
    """
    print("=" * 72)
    print(f"SNIPER VALIDATION: {asset}")
    print("=" * 72)
    print(f"Leverage: {leverage}x")
    print(f"OU Threshold: ±{ou_threshold}")
    if validation_days > 0:
        print(f"Validation Period: Last {validation_days} days")
    
    # Load trained fleet predictions
    h1_predictions = load_fleet_predictions(asset, validation_days)
    
    # Load M5 data for execution
    print(f"\n[DATA] Loading M5 data for {asset}...")
    loader_m5 = MarketDataLoader(symbol=asset, interval="5")
    
    # Match M5 data period to predictions
    pred_start = h1_predictions['timestamp'].min()
    pred_end = h1_predictions['timestamp'].max()
    days_back = (pred_end - pred_start).days + 30
    
    df_m5 = loader_m5.get_data(days_back=days_back)
    
    if df_m5 is None or len(df_m5) < 1000:
        raise ValueError(f"Insufficient M5 data for {asset}")
    
    print(f"  Loaded {len(df_m5)} M5 bars")
    
    # Scenario A: Naive (H1 Entry)
    print("\n" + "=" * 72)
    print("SCENARIO A: NAIVE (Enter at H1 Open)")
    print("=" * 72)
    
    config_naive = TradingConfig(
        initial_capital=10000.0,
        leverage=leverage,
        use_sniper=False,
        strict_mode=False,
    )
    
    engine_naive = SniperBacktestEngine(config_naive)
    results_naive = engine_naive.run(h1_predictions, df_m5)
    
    print_results(results_naive, "NAIVE")
    
    # Scenario B: Sniper (M5 Entry)
    print("\n" + "=" * 72)
    print("SCENARIO B: SNIPER (Enter on M5 Dip/Spike)")
    print("=" * 72)
    
    config_sniper = TradingConfig(
        initial_capital=10000.0,
        leverage=leverage,
        use_sniper=True,
        strict_mode=False,
        ou_entry_long=-ou_threshold,
        ou_entry_short=ou_threshold,
    )
    
    engine_sniper = SniperBacktestEngine(config_sniper)
    results_sniper = engine_sniper.run(h1_predictions, df_m5)
    
    print_results(results_sniper, "SNIPER")
    
    # Comparison
    print("\n" + "=" * 72)
    print("COMPARISON: SNIPER vs NAIVE")
    print("=" * 72)
    
    compare_results(results_naive, results_sniper)
    
    # Entry improvement
    entry_improvement = calculate_entry_improvement(
        results_sniper['trades'],
        results_naive['trades']
    )
    
    if entry_improvement.get('n_compared', 0) > 0:
        print("\n" + "-" * 72)
        print("ENTRY PRICE IMPROVEMENT (Sniper vs Naive)")
        print("-" * 72)
        print(f"  Avg Improvement:    {entry_improvement['avg_improvement_bps']:.2f} bps")
        print(f"  Median Improvement: {entry_improvement['median_improvement_bps']:.2f} bps")
        print(f"  % Better Entries:   {entry_improvement['pct_better_entries']:.1%}")
        print(f"  Trades Compared:    {entry_improvement['n_compared']}")
    
    # Create visualization
    plot_sniper_trades(asset, df_m5, results_naive, results_sniper, h1_predictions)
    
    return {
        'asset': asset,
        'naive': results_naive,
        'sniper': results_sniper,
        'entry_improvement': entry_improvement,
    }


def plot_sniper_trades(
    asset: str,
    df_m5: pd.DataFrame,
    results_naive: dict,
    results_sniper: dict,
    h1_predictions: pd.DataFrame,
):
    """
    Create detailed visualization showing all trades.
    
    Parameters
    ----------
    asset : str
        Asset symbol
    df_m5 : pd.DataFrame
        M5 price data
    results_naive : dict
        Naive strategy results
    results_sniper : dict
        Sniper strategy results
    h1_predictions : pd.DataFrame
        H1 predictions
    """
    print(f"\n[PLOT] Creating trade visualization for {asset}...")
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Get price data for visualization period
    pred_start = h1_predictions['timestamp'].min()
    pred_end = h1_predictions['timestamp'].max()
    
    df_plot = df_m5[(df_m5.index >= pred_start) & (df_m5.index <= pred_end)].copy()
    
    if len(df_plot) == 0:
        print("  Warning: No price data in prediction period")
        return
    
    # Axis 1: Price + Naive Trades
    ax1 = axes[0]
    ax1.plot(df_plot.index, df_plot['close'], label='Price', linewidth=1, alpha=0.7, color='black')
    
    # Plot naive trades
    for trade in results_naive['trades']:
        color = 'green' if trade.pnl > 0 else 'red'
        marker = '^' if trade.direction == 'long' else 'v'
        
        # Entry
        ax1.scatter(trade.entry_time, trade.entry_price, color=color, marker=marker, 
                   s=100, alpha=0.6, edgecolors='black', linewidths=1)
        # Exit
        ax1.scatter(trade.exit_time, trade.exit_price, color=color, marker='x', 
                   s=100, alpha=0.6)
        # Connect entry to exit
        ax1.plot([trade.entry_time, trade.exit_time], 
                [trade.entry_price, trade.exit_price], 
                color=color, alpha=0.3, linewidth=1)
    
    ax1.set_title(f'{asset} - Naive Strategy (Enter at H1 Open)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=10)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Axis 2: Price + Sniper Trades
    ax2 = axes[1]
    ax2.plot(df_plot.index, df_plot['close'], label='Price', linewidth=1, alpha=0.7, color='black')
    
    # Plot sniper trades
    for trade in results_sniper['trades']:
        color = 'green' if trade.pnl > 0 else 'red'
        marker = '^' if trade.direction == 'long' else 'v'
        
        # Entry (with OU Z-score annotation)
        ax2.scatter(trade.entry_time, trade.entry_price, color=color, marker=marker, 
                   s=100, alpha=0.6, edgecolors='black', linewidths=1)
        # Exit
        ax2.scatter(trade.exit_time, trade.exit_price, color=color, marker='x', 
                   s=100, alpha=0.6)
        # Connect entry to exit
        ax2.plot([trade.entry_time, trade.exit_time], 
                [trade.entry_price, trade.exit_price], 
                color=color, alpha=0.3, linewidth=1)
    
    ax2.set_title(f'{asset} - Sniper Strategy (Enter on M5 Dip/Spike)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Price ($)', fontsize=10)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Axis 3: Equity Curves
    ax3 = axes[2]
    
    if len(results_naive['equity_curve']) > 0:
        equity_naive = results_naive['equity_curve']
        ax3.plot(equity_naive['timestamp'], equity_naive['equity'], 
                label='Naive', linewidth=2, alpha=0.8, color='blue')
    
    if len(results_sniper['equity_curve']) > 0:
        equity_sniper = results_sniper['equity_curve']
        ax3.plot(equity_sniper['timestamp'], equity_sniper['equity'], 
                label='Sniper', linewidth=2, alpha=0.8, color='orange')
    
    ax3.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    ax3.set_title('Equity Curve Comparison', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=10)
    ax3.set_ylabel('Equity ($)', fontsize=10)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save plot
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    plot_path = artifacts_dir / f"sniper_trades_{asset}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved to: {plot_path}")
    
    plt.close()


def print_results(results: dict, label: str):
    """Print backtest results."""
    stats = results['statistics']
    
    print(f"\n[{label}] Performance Metrics:")
    print(f"  Total Return:       {stats['total_return_pct']:.2%} (${stats['total_return_dollar']:,.2f})")
    print(f"  Trades:             {stats['n_trades']}")
    print(f"  Win Rate:           {stats['win_rate']:.1%}")
    print(f"  Avg Profit/Trade:   ${stats['avg_profit_per_trade']:.2f}")
    print(f"  Max Drawdown:       {stats['max_drawdown_pct']:.2%}")
    print(f"  Sharpe Ratio:       {stats['sharpe_ratio']:.2f}")
    print(f"  Final Equity:       ${results['final_equity']:,.2f}")


def compare_results(results_naive: dict, results_sniper: dict):
    """Compare two backtest results."""
    stats_naive = results_naive['statistics']
    stats_sniper = results_sniper['statistics']
    
    return_improvement = stats_sniper['total_return_pct'] - stats_naive['total_return_pct']
    sharpe_improvement = stats_sniper['sharpe_ratio'] - stats_naive['sharpe_ratio']
    
    print("\n" + "-" * 72)
    print("PERFORMANCE DELTA (Sniper - Naive)")
    print("-" * 72)
    print(f"  Return Improvement:    {return_improvement:+.2%}")
    print(f"  Sharpe Improvement:    {sharpe_improvement:+.2f}")
    print(f"  Trade Count Change:    {stats_sniper['n_trades'] - stats_naive['n_trades']:+d}")
    
    # Verdict
    print("\n" + "-" * 72)
    print("VERDICT")
    print("-" * 72)
    
    if stats_naive['n_trades'] == 0 and stats_sniper['n_trades'] == 0:
        print("  X NO TRADES: Both strategies executed no trades")
    elif stats_sniper['n_trades'] == 0:
        print("  X SNIPER FAILED: No trades executed")
        print("  Recommendation: Relax OU thresholds")
    elif return_improvement > 0 and sharpe_improvement > 0:
        print("  + SNIPER WINS: Better return AND better risk-adjusted performance")
        print("  Recommendation: Deploy sniper strategy")
    elif return_improvement > 0:
        print("  ~ SNIPER MIXED: Better return but worse risk metrics")
    else:
        print("  - NAIVE WINS: Sniper did not improve performance")


def run_all_assets_validation(
    assets: List[str],
    leverage: float = 5.0,
    ou_threshold: float = 1.0,
    validation_days: int = 0,
) -> Dict:
    """
    Run sniper validation on all assets.
    
    Parameters
    ----------
    assets : list
        List of asset symbols
    leverage : float
        Leverage (max 10x)
    ou_threshold : float
        OU Z-score threshold
    validation_days : int
        If > 0, only validate on last N days
    
    Returns
    -------
    dict
        Results for all assets
    """
    print("=" * 72)
    print("SNIPER VALIDATION: ALL ASSETS MODE")
    print("=" * 72)
    print(f"Assets: {len(assets)}")
    print(f"Validation Days: {validation_days if validation_days > 0 else 'All available'}")
    
    all_results = {}
    summary_data = []
    
    for asset in assets:
        print(f"\n{'=' * 72}")
        print(f"Processing {asset}...")
        print(f"{'=' * 72}")
        
        try:
            results = run_sniper_validation(
                asset=asset,
                leverage=leverage,
                ou_threshold=ou_threshold,
                validation_days=validation_days,
            )
            
            all_results[asset] = results
            
            # Collect summary
            summary_data.append({
                'Asset': asset,
                'Naive_Return': results['naive']['statistics']['total_return_pct'],
                'Sniper_Return': results['sniper']['statistics']['total_return_pct'],
                'Improvement': results['sniper']['statistics']['total_return_pct'] - results['naive']['statistics']['total_return_pct'],
                'Naive_Trades': results['naive']['statistics']['n_trades'],
                'Sniper_Trades': results['sniper']['statistics']['n_trades'],
                'Entry_Improvement_bps': results['entry_improvement'].get('avg_improvement_bps', 0),
            })
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Print summary
    if len(summary_data) > 0:
        print("\n" + "=" * 72)
        print("SUMMARY: ALL ASSETS")
        print("=" * 72)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Improvement', ascending=False)
        
        print(summary_df.to_string(index=False))
        
        # Save summary
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        summary_file = artifacts_dir / "sniper_validation_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSummary saved to: {summary_file}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Sniper Validation with Trained Fleet Models"
    )
    parser.add_argument("--asset", type=str, default="BTCUSDT", help="Asset symbol (or 'all' for all assets)")
    parser.add_argument("--leverage", type=float, default=5.0, help="Leverage (max 10x)")
    parser.add_argument("--ou-threshold", type=float, default=1.0, help="OU Z-score threshold")
    parser.add_argument("--validation-days", type=int, default=0, help="Last N days to validate on (0 = all)")
    
    args = parser.parse_args()
    
    if args.leverage > 10.0:
        print(f"Warning: Leverage capped at 10x (requested: {args.leverage}x)")
        args.leverage = 10.0
    
    try:
        if args.asset.lower() == 'all':
            # Run on all assets
            results = run_all_assets_validation(
                assets=FLEET_ASSETS,
                leverage=args.leverage,
                ou_threshold=args.ou_threshold,
                validation_days=args.validation_days,
            )
        else:
            # Run on single asset
            results = run_sniper_validation(
                asset=args.asset,
                leverage=args.leverage,
                ou_threshold=args.ou_threshold,
                validation_days=args.validation_days,
            )
        
        print("\n" + "=" * 72)
        print("VALIDATION COMPLETE")
        print("=" * 72)
        print("\nThis validation used TRAINED FLEET PREDICTIONS")
        print("Not naive SMA cross - your actual 45-60% precision models!")
        print("\nCheck artifacts/ for trade visualizations")
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nTo generate predictions, run:")
        print("  python run_enriched_fleet.py --h1-days 730 --m5-days 150")
