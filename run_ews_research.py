"""
Early Warning Signals Research - Critical Slowing Down Detection

This script backtests Early Warning Signals (EWS) on BTCUSDT price data
to detect Critical Slowing Down before crashes.
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_loader import MarketDataLoader
from src.analysis.stability_monitor import StabilityMonitor
from src.config import PLOT_TEMPLATE


def main():
    """Backtest Early Warning Signals on BTCUSDT price data."""
    
    print("=" * 80)
    print("EARLY WARNING SIGNALS RESEARCH - CRITICAL SLOWING DOWN DETECTION")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # Step 1: Data Loading
    # -------------------------------------------------------------------------
    print("\\n[STEP 1] Loading BTCUSDT price data...")
    
    loader = MarketDataLoader(symbol="BTCUSDT", interval="60")  # 1H candles
    
    try:
        # Fetch 90 days of data
        df = loader.get_data(days_back=90, force_refresh=False)
        
        if df.empty:
            print("ERROR: No data retrieved. Exiting.")
            return
        
        print(f"[OK] Loaded {len(df)} candles")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        
    except Exception as e:
        print(f"ERROR: Failed to load data: {e}")
        return
    
    # Use log-prices for stability analysis
    log_prices = np.log(df["close"])
    prices = df["close"]
    
    print(f"  Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    print(f"  Log-price range: {log_prices.min():.4f} - {log_prices.max():.4f}")
    
    # -------------------------------------------------------------------------
    # Step 2: Compute Stability Indicators
    # -------------------------------------------------------------------------
    print("\\n[STEP 2] Computing Stability Indicators")
    print("-" * 80)
    
    # Window size: 7 days * 24 hours = 168 hours
    window_size = 168
    
    print(f"  Rolling window: {window_size} hours (7 days)")
    
    monitor = StabilityMonitor()
    indicators = monitor.compute_indicators(log_prices, window_size=window_size, dt=1.0)
    
    # Drop NaN values for statistics
    valid_indicators = indicators.dropna()
    
    print(f"\\n  Indicator Statistics:")
    print(f"    ACF-1:    mean={valid_indicators['acf1'].mean():.4f}, "
          f"std={valid_indicators['acf1'].std():.4f}, "
          f"max={valid_indicators['acf1'].max():.4f}")
    print(f"    Variance: mean={valid_indicators['variance'].mean():.6f}, "
          f"std={valid_indicators['variance'].std():.6f}")
    print(f"    Theta:    mean={valid_indicators['theta'].mean():.4f}, "
          f"std={valid_indicators['theta'].std():.4f}, "
          f"min={valid_indicators['theta'].min():.4f}")
    
    # -------------------------------------------------------------------------
    # Step 3: Detect Early Warning Signals
    # -------------------------------------------------------------------------
    print("\\n[STEP 3] Detecting Early Warning Signals")
    print("-" * 80)
    
    # Detect warnings with Z-score threshold = 2.0
    warnings = monitor.detect_warnings(indicators, threshold_z=2.0, z_window=50)
    
    # Count warnings
    n_warnings_acf = warnings['warning_acf'].sum()
    n_warnings_theta = warnings['warning_theta'].sum()
    n_warnings_combined = warnings['warning_combined'].sum()
    
    print(f"  Warning Counts:")
    print(f"    ACF warnings:      {n_warnings_acf} ({n_warnings_acf/len(warnings)*100:.1f}%)")
    print(f"    Theta warnings:    {n_warnings_theta} ({n_warnings_theta/len(warnings)*100:.1f}%)")
    print(f"    Combined warnings: {n_warnings_combined} ({n_warnings_combined/len(warnings)*100:.1f}%)")
    
    # -------------------------------------------------------------------------
    # Step 4: Identify Actual Crashes
    # -------------------------------------------------------------------------
    print("\\n[STEP 4] Identifying Actual Crashes")
    print("-" * 80)
    
    # Identify crashes (>5% drop in 24H)
    crashes = monitor.identify_crashes(prices, threshold_pct=5.0, window_hours=24)
    
    n_crashes = crashes.sum()
    crash_dates = prices[crashes].index.tolist()
    
    print(f"  Crashes detected: {n_crashes}")
    if n_crashes > 0:
        print(f"  Crash dates:")
        for date in crash_dates[:5]:  # Show first 5
            price_at_crash = prices.loc[date]
            print(f"    - {date}: ${price_at_crash:.2f}")
        if n_crashes > 5:
            print(f"    ... and {n_crashes - 5} more")
    
    # -------------------------------------------------------------------------
    # Step 5: Analyze Warning Performance
    # -------------------------------------------------------------------------
    print("\\n[STEP 5] Analyzing Warning Performance")
    print("-" * 80)
    
    # Check if warnings preceded crashes
    # Look for warnings in 24-48 hours before crashes
    lead_time_hours = 48
    
    true_positives = 0
    for crash_idx in crashes[crashes].index:
        # Get window before crash
        crash_loc = warnings.index.get_loc(crash_idx)
        if crash_loc >= lead_time_hours:
            window_start = crash_loc - lead_time_hours
            window = warnings.iloc[window_start:crash_loc]
            
            # Check if any warnings in this window
            if window['warning_combined'].any():
                true_positives += 1
    
    if n_crashes > 0:
        hit_rate = (true_positives / n_crashes) * 100
        print(f"  True Positives: {true_positives}/{n_crashes} crashes ({hit_rate:.1f}%)")
        print(f"  Lead time window: {lead_time_hours} hours before crash")
    else:
        print(f"  No crashes detected in this period")
    
    # -------------------------------------------------------------------------
    # Step 6: Visualization
    # -------------------------------------------------------------------------
    print("\\n[STEP 6] Generating Visualization...")
    
    # Create 3-row subplot
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "BTC Price with Early Warning Signals",
            "Rolling Theta (Mean Reversion Speed)",
            "Rolling Autocorrelation (ACF-1)",
        ),
        row_heights=[0.4, 0.3, 0.3],
    )
    
    # Row 1: Price with warnings
    fig.add_trace(
        go.Scatter(
            x=prices.index,
            y=prices.values,
            mode="lines",
            name="BTC Price",
            line=dict(color="#26a69a", width=2),
        ),
        row=1,
        col=1,
    )
    
    # Highlight warning periods
    warning_periods = warnings['warning_combined']
    if warning_periods.any():
        # Find contiguous warning regions
        warning_changes = warning_periods.astype(int).diff()
        warning_starts = warning_periods.index[warning_changes == 1]
        warning_ends = warning_periods.index[warning_changes == -1]
        
        # Add shaded regions
        for start in warning_starts[:20]:  # Limit to first 20 for performance
            # Find corresponding end
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
                row=1,
                col=1,
            )
    
    # Mark crashes
    if n_crashes > 0:
        crash_prices = prices[crashes]
        fig.add_trace(
            go.Scatter(
                x=crash_prices.index,
                y=crash_prices.values,
                mode="markers",
                name="Crashes",
                marker=dict(size=10, color="#ef5350", symbol="x", line=dict(width=2)),
            ),
            row=1,
            col=1,
        )
    
    # Row 2: Rolling Theta
    fig.add_trace(
        go.Scatter(
            x=warnings.index,
            y=warnings['theta'],
            mode="lines",
            name="Theta",
            line=dict(color="#42A5F5", width=2),
        ),
        row=2,
        col=1,
    )
    
    # Add theta = 0 line (instability threshold)
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="red",
        opacity=0.7,
        annotation_text="Instability (theta=0)",
        row=2,
        col=1,
    )
    
    # Add Z-score bands
    fig.add_trace(
        go.Scatter(
            x=warnings.index,
            y=warnings['z_theta'],
            mode="lines",
            name="Z-score (Theta)",
            line=dict(color="#FFA726", width=1, dash="dot"),
            opacity=0.5,
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    
    # Row 3: Rolling Autocorrelation
    fig.add_trace(
        go.Scatter(
            x=warnings.index,
            y=warnings['acf1'],
            mode="lines",
            name="ACF-1",
            line=dict(color="#9C27B0", width=2),
        ),
        row=3,
        col=1,
    )
    
    # Add ACF = 1 line (critical threshold)
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="red",
        opacity=0.7,
        annotation_text="Critical (ACF=1)",
        row=3,
        col=1,
    )
    
    # Add Z-score
    fig.add_trace(
        go.Scatter(
            x=warnings.index,
            y=warnings['z_acf1'],
            mode="lines",
            name="Z-score (ACF)",
            line=dict(color="#FFA726", width=1, dash="dot"),
            opacity=0.5,
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    
    # Update layout
    fig.update_layout(
        title="Early Warning Signals - Critical Slowing Down Analysis",
        template=PLOT_TEMPLATE,
        height=1000,
        hovermode="x unified",
        showlegend=True,
    )
    
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Theta", row=2, col=1)
    fig.update_yaxes(title_text="ACF-1", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    # Save and open
    output_path = Path("ews_research_results.html")
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
    print("SUMMARY")
    print("=" * 80)
    
    print("\\n[STABILITY INDICATORS]")
    print(f"  - ACF-1 mean: {valid_indicators['acf1'].mean():.4f} (approaches 1.0 = critical)")
    print(f"  - Theta mean: {valid_indicators['theta'].mean():.4f} (approaches 0.0 = unstable)")
    print(f"  - Variance mean: {valid_indicators['variance'].mean():.6f}")
    
    print("\\n[EARLY WARNING SIGNALS]")
    print(f"  - Total warnings: {n_warnings_combined} periods")
    print(f"  - ACF warnings: {n_warnings_acf} (rising autocorrelation)")
    print(f"  - Theta warnings: {n_warnings_theta} (falling mean reversion)")
    
    print("\\n[CRASH DETECTION]")
    print(f"  - Crashes identified: {n_crashes} (>5% drop in 24H)")
    if n_crashes > 0:
        print(f"  - Warnings before crashes: {true_positives}/{n_crashes} ({hit_rate:.1f}%)")
        print(f"  - Lead time: {lead_time_hours} hours")
    
    print("\\n" + "=" * 80)
    print("EARLY WARNING SIGNALS RESEARCH COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
