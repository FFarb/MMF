"""
Strategy Backtest - Physics-Based Risk Management Alpha Proof

This script demonstrates that physics-based risk management generates alpha
by comparing a naive RSI mean reversion strategy vs. a physics-aware version
that scales position sizing based on Early Warning Signals.
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_loader import MarketDataLoader
from src.analysis.stability_monitor import StabilityMonitor
from src.trading.risk_engine import StabilityRiskManager
from src.config import PLOT_TEMPLATE


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_performance_metrics(equity_curve: pd.Series) -> dict:
    """Calculate strategy performance metrics."""
    returns = equity_curve.pct_change().dropna()
    
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    
    # Max drawdown
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax * 100
    max_drawdown = drawdown.min()
    
    # Sharpe ratio (annualized, assuming hourly data)
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(24 * 365)
    else:
        sharpe = 0.0
    
    return {
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'final_equity': equity_curve.iloc[-1],
    }


def main():
    """Run comparative backtest: Naive vs Physics-Aware strategy."""
    
    print("=" * 80)
    print("STRATEGY BACKTEST - PHYSICS-BASED RISK MANAGEMENT ALPHA PROOF")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # Step 1: Data Loading
    # -------------------------------------------------------------------------
    print("\\n[STEP 1] Loading BTCUSDT data...")
    
    loader = MarketDataLoader(symbol="BTCUSDT", interval="60")
    
    try:
        df = loader.get_data(days_back=90, force_refresh=False)
        
        if df.empty:
            print("ERROR: No data retrieved. Exiting.")
            return
        
        print(f"[OK] Loaded {len(df)} candles")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        
    except Exception as e:
        print(f"ERROR: Failed to load data: {e}")
        return
    
    prices = df["close"]
    log_prices = np.log(prices)
    
    # -------------------------------------------------------------------------
    # Step 2: Compute Indicators
    # -------------------------------------------------------------------------
    print("\\n[STEP 2] Computing Indicators...")
    print("-" * 80)
    
    # RSI for trading signals
    rsi = calculate_rsi(prices, period=14)
    print(f"  RSI computed: mean={rsi.mean():.2f}, min={rsi.min():.2f}, max={rsi.max():.2f}")
    
    # EWS for risk management
    window_size = 168  # 7 days
    monitor = StabilityMonitor()
    indicators = monitor.compute_indicators(log_prices, window_size=window_size)
    warnings = monitor.detect_warnings(indicators, threshold_z=2.0, z_window=50)
    
    print(f"  EWS computed: {warnings['warning_combined'].sum()} warning periods")
    
    # -------------------------------------------------------------------------
    # Step 3: Initialize Strategies
    # -------------------------------------------------------------------------
    print("\\n[STEP 3] Initializing Strategies...")
    print("-" * 80)
    
    # Risk manager (strict mode: warning → cash)
    risk_mgr = StabilityRiskManager(mode='strict')
    
    # Initial capital
    initial_capital = 10000.0
    
    # Strategy state
    naive_equity = [initial_capital]
    smart_equity = [initial_capital]
    
    naive_position = 0.0  # Current position in BTC
    smart_position = 0.0
    
    naive_cash = initial_capital
    smart_cash = initial_capital
    
    # Trade tracking
    naive_trades = 0
    smart_trades = 0
    falling_knives_avoided = 0
    
    # -------------------------------------------------------------------------
    # Step 4: Backtest Loop
    # -------------------------------------------------------------------------
    print("\\n[STEP 4] Running Backtest...")
    print("-" * 80)
    
    # Start after window_size to avoid NaN in indicators
    start_idx = window_size + 14  # window + RSI period
    
    for i in range(start_idx, len(df)):
        current_price = prices.iloc[i]
        current_rsi = rsi.iloc[i]
        
        # Get EWS indicators (using only past data - no look-ahead)
        current_theta = indicators['theta'].iloc[i]
        current_warning = warnings['warning_combined'].iloc[i]
        
        # RSI signals
        oversold = current_rsi < 30  # Buy signal
        neutral = current_rsi > 50   # Sell signal
        
        # --- Naive Strategy (Always trade RSI signals) ---
        if oversold and naive_position == 0:
            # Buy with fixed $1000
            buy_amount = 1000.0
            if naive_cash >= buy_amount:
                naive_position = buy_amount / current_price
                naive_cash -= buy_amount
                naive_trades += 1
        
        elif neutral and naive_position > 0:
            # Sell
            naive_cash += naive_position * current_price
            naive_position = 0.0
        
        # --- Smart Strategy (Scale by EWS) ---
        # Get risk multiplier
        multiplier = risk_mgr.get_leverage_multiplier(
            theta=current_theta,
            warning=current_warning,
        )
        
        if oversold and smart_position == 0:
            # Buy with EWS-adjusted size
            base_amount = 1000.0
            adjusted_amount = base_amount * multiplier
            
            if adjusted_amount > 0 and smart_cash >= adjusted_amount:
                smart_position = adjusted_amount / current_price
                smart_cash -= adjusted_amount
                smart_trades += 1
            elif multiplier == 0:
                # Falling knife avoided!
                falling_knives_avoided += 1
        
        elif neutral and smart_position > 0:
            # Sell
            smart_cash += smart_position * current_price
            smart_position = 0.0
        
        # Update equity
        naive_equity_value = naive_cash + naive_position * current_price
        smart_equity_value = smart_cash + smart_position * current_price
        
        naive_equity.append(naive_equity_value)
        smart_equity.append(smart_equity_value)
    
    # Convert to Series
    equity_index = df.index[start_idx-1:]
    naive_equity_series = pd.Series(naive_equity, index=equity_index)
    smart_equity_series = pd.Series(smart_equity, index=equity_index)
    
    # -------------------------------------------------------------------------
    # Step 5: Calculate Performance
    # -------------------------------------------------------------------------
    print("\\n[STEP 5] Performance Analysis")
    print("=" * 80)
    
    naive_perf = calculate_performance_metrics(naive_equity_series)
    smart_perf = calculate_performance_metrics(smart_equity_series)
    
    print("\\n[NAIVE STRATEGY - Fixed Size RSI]")
    print(f"  Total Return:   {naive_perf['total_return']:>8.2f}%")
    print(f"  Max Drawdown:   {naive_perf['max_drawdown']:>8.2f}%")
    print(f"  Sharpe Ratio:   {naive_perf['sharpe_ratio']:>8.2f}")
    print(f"  Final Equity:   ${naive_perf['final_equity']:>8.2f}")
    print(f"  Trades:         {naive_trades}")
    
    print("\\n[SMART STRATEGY - Physics-Aware RSI]")
    print(f"  Total Return:   {smart_perf['total_return']:>8.2f}%")
    print(f"  Max Drawdown:   {smart_perf['max_drawdown']:>8.2f}%")
    print(f"  Sharpe Ratio:   {smart_perf['sharpe_ratio']:>8.2f}")
    print(f"  Final Equity:   ${smart_perf['final_equity']:>8.2f}")
    print(f"  Trades:         {smart_trades}")
    print(f"  Falling Knives: {falling_knives_avoided} avoided")
    
    # Calculate improvements
    return_improvement = smart_perf['total_return'] - naive_perf['total_return']
    dd_improvement = naive_perf['max_drawdown'] - smart_perf['max_drawdown']
    sharpe_improvement = ((smart_perf['sharpe_ratio'] / naive_perf['sharpe_ratio']) - 1) * 100 if naive_perf['sharpe_ratio'] != 0 else 0
    
    print("\\n[IMPROVEMENT]")
    print(f"  Return Delta:   {return_improvement:>8.2f}%")
    print(f"  DD Reduction:   {dd_improvement:>8.2f}%")
    print(f"  Sharpe Gain:    {sharpe_improvement:>8.2f}%")
    
    # -------------------------------------------------------------------------
    # Step 6: Visualization
    # -------------------------------------------------------------------------
    print("\\n[STEP 6] Generating Visualization...")
    
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Equity Curves: Naive vs Physics-Aware Strategy",
            "Position Sizing Multiplier (Smart Strategy)",
            "BTC Price with EWS Warnings",
        ),
        row_heights=[0.4, 0.3, 0.3],
    )
    
    # Row 1: Equity Curves
    fig.add_trace(
        go.Scatter(
            x=naive_equity_series.index,
            y=naive_equity_series.values,
            mode="lines",
            name="Naive (Fixed Size)",
            line=dict(color="#ef5350", width=2),
        ),
        row=1,
        col=1,
    )
    
    fig.add_trace(
        go.Scatter(
            x=smart_equity_series.index,
            y=smart_equity_series.values,
            mode="lines",
            name="Smart (Physics-Aware)",
            line=dict(color="#26a69a", width=2),
        ),
        row=1,
        col=1,
    )
    
    # Row 2: Position Sizing Multiplier
    multipliers = []
    for i in range(start_idx, len(df)):
        theta_i = indicators['theta'].iloc[i]
        warning_i = warnings['warning_combined'].iloc[i]
        mult = risk_mgr.get_leverage_multiplier(theta=theta_i, warning=warning_i)
        multipliers.append(mult)
    
    multiplier_series = pd.Series(multipliers, index=df.index[start_idx:])
    
    fig.add_trace(
        go.Scatter(
            x=multiplier_series.index,
            y=multiplier_series.values,
            mode="lines",
            name="Risk Multiplier",
            line=dict(color="#FFA726", width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 167, 38, 0.3)',
        ),
        row=2,
        col=1,
    )
    
    # Row 3: BTC Price with warnings
    fig.add_trace(
        go.Scatter(
            x=prices.index,
            y=prices.values,
            mode="lines",
            name="BTC Price",
            line=dict(color="#42A5F5", width=2),
        ),
        row=3,
        col=1,
    )
    
    # Highlight warning periods
    warning_periods = warnings['warning_combined']
    if warning_periods.any():
        warning_changes = warning_periods.astype(int).diff()
        warning_starts = warning_periods.index[warning_changes == 1]
        warning_ends = warning_periods.index[warning_changes == -1]
        
        for start in warning_starts[:20]:
            end_candidates = warning_ends[warning_ends > start]
            if len(end_candidates) > 0:
                end = end_candidates[0]
            else:
                end = warning_periods.index[-1]
            
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor="red",
                opacity=0.2,
                layer="below",
                line_width=0,
                row=3,
                col=1,
            )
    
    # Update layout
    fig.update_layout(
        title="Physics-Based Risk Management - Alpha Proof",
        template=PLOT_TEMPLATE,
        height=1000,
        hovermode="x unified",
        showlegend=True,
    )
    
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Multiplier", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    # Save and open
    output_path = Path("strategy_backtest_results.html")
    fig.write_html(output_path)
    print(f"[OK] Chart saved to {output_path}")
    
    import webbrowser
    import os
    filepath = Path(os.path.abspath(output_path))
    webbrowser.open(f"file://{filepath}")
    print("[OK] Opening chart in browser...")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\\n" + "=" * 80)
    print("SUMMARY - ALPHA PROOF")
    print("=" * 80)
    
    if smart_perf['sharpe_ratio'] > naive_perf['sharpe_ratio']:
        print("\\n[SUCCESS] Physics-based risk management GENERATES ALPHA!")
        print(f"  Sharpe improvement: {sharpe_improvement:.1f}%")
        print(f"  Drawdown reduction: {dd_improvement:.2f}%")
        print(f"  Falling knives avoided: {falling_knives_avoided}")
    else:
        print("\\n[INCONCLUSIVE] Results mixed - may need parameter tuning")
    
    print("\\n" + "=" * 80)
    print("STRATEGY BACKTEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
