"""
Sniper Simulation with Trained Fleet Models.

This script uses ACTUAL predictions from the trained hierarchical fleet,
not naive SMA cross signals. It validates M5 execution timing improvements
on top of your high-quality H1 model predictions.

Flow:
1. Load H1 predictions from trained fleet (45-60% precision)
2. Use those predictions as directional bias
3. Apply M5 OU execution timing
4. Compare naive H1 entry vs sniper M5 entry
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent))

from src.data_loader import MarketDataLoader
from src.trading.sniper_engine import (
    SniperBacktestEngine,
    TradingConfig,
    calculate_entry_improvement,
)


def load_fleet_predictions(asset: str = 'BTCUSDT') -> pd.DataFrame:
    """
    Load H1 predictions from trained hierarchical fleet.
    
    This uses the ACTUAL trained models with 45-60% precision,
    not naive SMA cross.
    
    Returns
    -------
    pd.DataFrame
        H1 predictions with columns: ['timestamp', 'signal', 'close', 'probability']
    """
    print(f"[FLEET] Loading trained model predictions for {asset}...")
    
    # Check for saved predictions
    predictions_file = Path(f"artifacts/fleet_predictions_{asset}.csv")
    
    if predictions_file.exists():
        print(f"  Loading from: {predictions_file}")
        df_pred = pd.read_csv(predictions_file)
        df_pred['timestamp'] = pd.to_datetime(df_pred['timestamp'])
        
        print(f"  Loaded {len(df_pred)} H1 predictions")
        print(f"  Signals: {(df_pred['signal'] == 1).sum()} long, {(df_pred['signal'] == -1).sum()} short")
        
        return df_pred
    else:
        print(f"  ERROR: No predictions found at {predictions_file}")
        print(f"  Please run hierarchical fleet training first:")
        print(f"    python run_enriched_fleet.py --h1-days 730 --m5-days 150")
        print(f"  Then save predictions to artifacts/fleet_predictions_{{asset}}.csv")
        raise FileNotFoundError(f"Fleet predictions not found: {predictions_file}")


def run_sniper_validation(
    asset: str = 'BTCUSDT',
    leverage: float = 5.0,
    ou_threshold: float = 1.0,  # Relaxed from 1.5
) -> dict:
    """
    Validate sniper execution using trained fleet predictions.
    
    This is the CORRECT way to test the sniper architecture:
    - Use high-quality H1 predictions from trained models
    - Apply M5 execution timing
    - Measure improvement
    
    Parameters
    ----------
    asset : str
        Asset symbol
    leverage : float
        Leverage (max 10x)
    ou_threshold : float
        OU Z-score threshold for entry
    
    Returns
    -------
    dict
        Results from naive vs sniper comparison
    """
    print("=" * 72)
    print("SNIPER VALIDATION: Trained Fleet + M5 Execution")
    print("=" * 72)
    print(f"\nAsset: {asset}")
    print(f"Leverage: {leverage}x")
    print(f"OU Threshold: ±{ou_threshold}")
    
    # Load trained fleet predictions
    h1_predictions = load_fleet_predictions(asset)
    
    # Load M5 data for execution
    print(f"\n[DATA] Loading M5 data for {asset}...")
    loader_m5 = MarketDataLoader(symbol=asset, interval="5")
    
    # Match M5 data period to predictions
    pred_start = h1_predictions['timestamp'].min()
    pred_end = h1_predictions['timestamp'].max()
    days_back = (pred_end - pred_start).days + 30  # Add buffer
    
    df_m5 = loader_m5.get_data(days_back=days_back)
    
    if df_m5 is None or len(df_m5) < 1000:
        raise ValueError(f"Insufficient M5 data for {asset}")
    
    print(f"  Loaded {len(df_m5)} M5 bars")
    print(f"  M5 range: {df_m5.index.min()} to {df_m5.index.max()}")
    
    # Scenario A: Naive (H1 Entry)
    print("\n" + "=" * 72)
    print("SCENARIO A: NAIVE (Enter at H1 Open)")
    print("Using trained fleet predictions")
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
    print("Using trained fleet predictions + M5 timing")
    print("=" * 72)
    
    config_sniper = TradingConfig(
        initial_capital=10000.0,
        leverage=leverage,
        use_sniper=True,
        strict_mode=False,  # Relaxed - enter at minute 55 if no signal
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
    
    return {
        'naive': results_naive,
        'sniper': results_sniper,
        'entry_improvement': entry_improvement,
    }


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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Sniper Validation with Trained Fleet Models"
    )
    parser.add_argument("--asset", type=str, default="BTCUSDT", help="Asset symbol")
    parser.add_argument("--leverage", type=float, default=5.0, help="Leverage (max 10x)")
    parser.add_argument("--ou-threshold", type=float, default=1.0, help="OU Z-score threshold")
    
    args = parser.parse_args()
    
    if args.leverage > 10.0:
        print(f"Warning: Leverage capped at 10x (requested: {args.leverage}x)")
        args.leverage = 10.0
    
    try:
        results = run_sniper_validation(
            asset=args.asset,
            leverage=args.leverage,
            ou_threshold=args.ou_threshold,
        )
        
        print("\n" + "=" * 72)
        print("VALIDATION COMPLETE")
        print("=" * 72)
        print("\nThis validation used TRAINED FLEET PREDICTIONS")
        print("Not naive SMA cross - your actual 45-60% precision models!")
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nTo generate predictions, run:")
        print("  python run_enriched_fleet.py --h1-days 730 --m5-days 150")
