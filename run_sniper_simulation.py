"""
Sniper Simulation: H1 Strategy + M5 Execution Validation.

This script validates the "Sniper" trading architecture by comparing:
- Scenario A (Naive): Enter at H1 open, exit at H1 close
- Scenario B (Sniper): Enter on M5 dip/spike (OU), exit at H1 close

Goal: Prove that precise M5 entries add reliable alpha over naive H1 execution.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_loader import MarketDataLoader
from src.trading.sniper_engine import (
    SniperBacktestEngine,
    TradingConfig,
    calculate_entry_improvement,
)


def load_h1_signals(asset: str = 'BTCUSDT') -> pd.DataFrame:
    """
    Load H1 signals from trained model.
    
    For this simulation, we'll generate synthetic signals based on
    simple trend following (SMA cross) as a proxy.
    
    In production, load from: artifacts/specialized_moe_results.csv
    """
    print(f"[DATA] Loading H1 signals for {asset}...")
    
    # Load H1 data
    loader = MarketDataLoader(symbol=asset, interval="60")
    df_h1 = loader.get_data(days_back=365)  # 1 year
    
    if df_h1 is None or len(df_h1) < 200:
        raise ValueError(f"Insufficient H1 data for {asset}")
    
    print(f"  Loaded {len(df_h1)} H1 bars")
    
    # Generate signals (SMA cross as proxy)
    df_h1['sma_50'] = df_h1['close'].rolling(window=50).mean()
    df_h1['sma_200'] = df_h1['close'].rolling(window=200).mean()
    
    # Signal: 1 (long), -1 (short), 0 (neutral)
    df_h1['signal'] = 0
    df_h1.loc[df_h1['sma_50'] > df_h1['sma_200'], 'signal'] = 1
    df_h1.loc[df_h1['sma_50'] < df_h1['sma_200'], 'signal'] = -1
    
    # Add timestamp column
    df_h1 = df_h1.reset_index()
    if 'index' in df_h1.columns and df_h1['index'].dtype == 'datetime64[ns]':
        df_h1.rename(columns={'index': 'timestamp'}, inplace=True)
    
    # Filter to only rows with signals
    df_signals = df_h1[['timestamp', 'signal', 'close']].copy()
    
    n_long = (df_signals['signal'] == 1).sum()
    n_short = (df_signals['signal'] == -1).sum()
    n_neutral = (df_signals['signal'] == 0).sum()
    
    print(f"  Signals: {n_long} long, {n_short} short, {n_neutral} neutral")
    
    return df_signals


def load_m5_data(asset: str = 'BTCUSDT', days_back: int = 365) -> pd.DataFrame:
    """Load M5 OHLCV data."""
    print(f"[DATA] Loading M5 data for {asset}...")
    
    loader = MarketDataLoader(symbol=asset, interval="5")
    df_m5 = loader.get_data(days_back=days_back)
    
    if df_m5 is None or len(df_m5) < 1000:
        raise ValueError(f"Insufficient M5 data for {asset}")
    
    print(f"  Loaded {len(df_m5)} M5 bars")
    
    return df_m5


def run_comparison(
    asset: str = 'BTCUSDT',
    days_back: int = 365,
    leverage: float = 5.0,
) -> Dict:
    """
    Run comparison between Naive and Sniper strategies.
    
    Parameters
    ----------
    asset : str
        Asset symbol
    days_back : int
        Days of history to backtest
    leverage : float
        Leverage multiplier (max 10x)
    
    Returns
    -------
    dict
        Results from both scenarios
    """
    print("=" * 72)
    print("SNIPER SIMULATION: H1 Strategy + M5 Execution")
    print("=" * 72)
    print(f"\nAsset: {asset}")
    print(f"Period: {days_back} days")
    print(f"Leverage: {leverage}x")
    
    # Load data
    h1_signals = load_h1_signals(asset)
    m5_data = load_m5_data(asset, days_back)
    
    # Scenario A: Naive (H1 Entry)
    print("\n" + "=" * 72)
    print("SCENARIO A: NAIVE (Enter at H1 Open)")
    print("=" * 72)
    
    config_naive = TradingConfig(
        initial_capital=10000.0,
        leverage=leverage,
        use_sniper=False,  # Naive mode
        strict_mode=False,
    )
    
    engine_naive = SniperBacktestEngine(config_naive)
    results_naive = engine_naive.run(h1_signals, m5_data)
    
    print_results(results_naive, "NAIVE")
    
    # Scenario B: Sniper (M5 Entry)
    print("\n" + "=" * 72)
    print("SCENARIO B: SNIPER (Enter on M5 Dip/Spike)")
    print("=" * 72)
    
    config_sniper = TradingConfig(
        initial_capital=10000.0,
        leverage=leverage,
        use_sniper=True,  # Sniper mode
        strict_mode=True,  # Cancel if no signal by min 55
        ou_entry_long=-1.5,
        ou_entry_short=1.5,
    )
    
    engine_sniper = SniperBacktestEngine(config_sniper)
    results_sniper = engine_sniper.run(h1_signals, m5_data)
    
    print_results(results_sniper, "SNIPER")
    
    # Comparison
    print("\n" + "=" * 72)
    print("COMPARISON: SNIPER vs NAIVE")
    print("=" * 72)
    
    compare_results(results_naive, results_sniper)
    
    # Entry improvement analysis
    entry_improvement = calculate_entry_improvement(
        results_sniper['trades'],
        results_naive['trades']
    )
    
    print("\n" + "─" * 72)
    print("ENTRY PRICE IMPROVEMENT (Sniper vs Naive)")
    print("─" * 72)
    print(f"  Avg Improvement:    {entry_improvement['avg_improvement_bps']:.2f} bps")
    print(f"  Median Improvement: {entry_improvement['median_improvement_bps']:.2f} bps")
    print(f"  % Better Entries:   {entry_improvement['pct_better_entries']:.1%}")
    print(f"  Trades Compared:    {entry_improvement['n_compared']}")
    
    # Plot equity curves
    plot_equity_curves(results_naive, results_sniper, asset)
    
    return {
        'naive': results_naive,
        'sniper': results_sniper,
        'entry_improvement': entry_improvement,
    }


def print_results(results: Dict, label: str):
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


def compare_results(results_naive: Dict, results_sniper: Dict):
    """Compare two backtest results."""
    stats_naive = results_naive['statistics']
    stats_sniper = results_sniper['statistics']
    
    # Calculate improvements
    return_improvement = stats_sniper['total_return_pct'] - stats_naive['total_return_pct']
    sharpe_improvement = stats_sniper['sharpe_ratio'] - stats_naive['sharpe_ratio']
    dd_improvement = stats_sniper['max_drawdown_pct'] - stats_naive['max_drawdown_pct']
    
    print("\n" + "─" * 72)
    print("PERFORMANCE DELTA (Sniper - Naive)")
    print("─" * 72)
    print(f"  Return Improvement:    {return_improvement:+.2%}")
    print(f"  Sharpe Improvement:    {sharpe_improvement:+.2f}")
    print(f"  Drawdown Change:       {dd_improvement:+.2%} (negative is better)")
    print(f"  Trade Count Change:    {stats_sniper['n_trades'] - stats_naive['n_trades']:+d}")
    
    # Calculate risk-adjusted improvement
    if stats_naive['max_drawdown_pct'] != 0 and stats_sniper['max_drawdown_pct'] != 0:
        naive_return_dd_ratio = stats_naive['total_return_pct'] / abs(stats_naive['max_drawdown_pct'])
        sniper_return_dd_ratio = stats_sniper['total_return_pct'] / abs(stats_sniper['max_drawdown_pct'])
        ratio_improvement = sniper_return_dd_ratio - naive_return_dd_ratio
        
        print(f"\n  Return/Drawdown Ratio:")
        print(f"    Naive:  {naive_return_dd_ratio:.2f}")
        print(f"    Sniper: {sniper_return_dd_ratio:.2f}")
        print(f"    Improvement: {ratio_improvement:+.2f}")
    elif stats_naive['n_trades'] == 0 or stats_sniper['n_trades'] == 0:
        print(f"\n  WARNING: One or both strategies executed no trades")
        print(f"    Naive trades: {stats_naive['n_trades']}")
        print(f"    Sniper trades: {stats_sniper['n_trades']}")
    
    # Verdict
    print("\n" + "-" * 72)
    print("VERDICT")
    print("-" * 72)
    
    if stats_naive['n_trades'] == 0 and stats_sniper['n_trades'] == 0:
        print("  X NO TRADES: Both strategies executed no trades")
        print("  Issue: Check data alignment and signal generation")
    elif stats_sniper['n_trades'] == 0:
        print("  X SNIPER FAILED: No trades executed")
        print("  Issue: OU thresholds too strict or M5 data missing")
        print("  Recommendation: Relax OU thresholds or check data")
    elif stats_naive['n_trades'] == 0:
        print("  X NAIVE FAILED: No trades executed")
        print("  Issue: Check H1 signals and M5 data alignment")
    elif return_improvement > 0 and sharpe_improvement > 0:
        print("  + SNIPER WINS: Better return AND better risk-adjusted performance")
        print("  Recommendation: Deploy Sniper strategy to demo account")
    elif return_improvement > 0:
        print("  ~ SNIPER MIXED: Better return but worse risk metrics")
        print("  Recommendation: Review risk management parameters")
    else:
        print("  - NAIVE WINS: Sniper did not improve performance")
        print("  Recommendation: Revisit entry logic or H1 signals")


def plot_equity_curves(results_naive: Dict, results_sniper: Dict, asset: str):
    """Plot equity curves for comparison."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Equity curves
    equity_naive = results_naive['equity_curve']
    equity_sniper = results_sniper['equity_curve']
    
    if len(equity_naive) > 0:
        ax1.plot(equity_naive['timestamp'], equity_naive['equity'], 
                label='Naive (H1 Entry)', linewidth=2, alpha=0.8)
    
    if len(equity_sniper) > 0:
        ax1.plot(equity_sniper['timestamp'], equity_sniper['equity'], 
                label='Sniper (M5 Entry)', linewidth=2, alpha=0.8)
    
    ax1.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Equity ($)')
    ax1.set_title(f'Equity Curve Comparison - {asset}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Drawdown comparison
    if len(equity_naive) > 0:
        cummax_naive = equity_naive['equity'].cummax()
        drawdown_naive = (equity_naive['equity'] - cummax_naive) / cummax_naive * 100
        ax2.fill_between(equity_naive['timestamp'], drawdown_naive, 0, 
                         alpha=0.3, label='Naive Drawdown')
    
    if len(equity_sniper) > 0:
        cummax_sniper = equity_sniper['equity'].cummax()
        drawdown_sniper = (equity_sniper['equity'] - cummax_sniper) / cummax_sniper * 100
        ax2.fill_between(equity_sniper['timestamp'], drawdown_sniper, 0, 
                         alpha=0.3, label='Sniper Drawdown')
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_title('Drawdown Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    plot_path = artifacts_dir / f"sniper_simulation_{asset}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n[PLOT] Saved to: {plot_path}")
    
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Sniper Simulation: H1 Strategy + M5 Execution"
    )
    parser.add_argument("--asset", type=str, default="BTCUSDT", help="Asset symbol")
    parser.add_argument("--days", type=int, default=365, help="Days of history")
    parser.add_argument("--leverage", type=float, default=5.0, help="Leverage (max 10x)")
    
    args = parser.parse_args()
    
    # Validate leverage
    if args.leverage > 10.0:
        print(f"Warning: Leverage capped at 10x (requested: {args.leverage}x)")
        args.leverage = 10.0
    
    # Run comparison
    results = run_comparison(
        asset=args.asset,
        days_back=args.days,
        leverage=args.leverage,
    )
    
    print("\n" + "=" * 72)
    print("SIMULATION COMPLETE")
    print("=" * 72)
    print("\nNext Steps:")
    print("  1. Review equity curves in artifacts/")
    print("  2. If Sniper wins, deploy to demo account")
    print("  3. Monitor live performance vs backtest")
