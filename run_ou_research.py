"""
OU Research Validation Script

This script validates the OU calibration engine by running two experiments:
1. Experiment A: Fit OU to log-prices (expect theta ~= 0, random walk)
2. Experiment B: Fit OU to realized volatility (expect theta > 0, mean reversion)
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_loader import MarketDataLoader
from src.analysis.ou_engine import OUEstimator
from src.config import PLOT_TEMPLATE


def main():
    """Run OU research experiments on BTCUSDT data."""
    
    print("=" * 80)
    print("OU CALIBRATION ENGINE - RESEARCH VALIDATION")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # Step 1: Data Loading
    # -------------------------------------------------------------------------
    print("\\n[STEP 1] Loading BTCUSDT market data...")
    
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
    
    # -------------------------------------------------------------------------
    # Step 2: Experiment A - Log-Prices (Random Walk Test)
    # -------------------------------------------------------------------------
    print("\\n[STEP 2] Experiment A: Log-Prices (Random Walk Test)")
    print("-" * 80)
    
    # Calculate log-prices
    log_prices = np.log(df["close"])
    
    # Fit OU model
    ou_prices = OUEstimator()
    ou_prices.fit(log_prices, dt=1.0)
    
    diag_prices = ou_prices.diagnostics()
    
    print(f"  Log-Price Series:")
    print(f"    Mean: {log_prices.mean():.4f}")
    print(f"    Std:  {log_prices.std():.4f}")
    print(f"\\n  OU Parameters:")
    print(f"    Theta:         {diag_prices['theta']:.6f}  {'[OK] Near zero (Random Walk)' if diag_prices['theta'] < 0.01 else '[WARN] Not random walk'}")
    print(f"    Mu:            {diag_prices['mu']:.4f}")
    print(f"    Sigma:         {diag_prices['sigma']:.6f}")
    print(f"    Half-life:     {diag_prices['half_life']:.2f} periods")
    print(f"\\n  Regression Diagnostics:")
    print(f"    Alpha:         {diag_prices['alpha']:.6f}")
    print(f"    Beta:          {diag_prices['beta']:.6f}")
    print(f"    R^2:           {diag_prices['r_squared']:.4f}")
    print(f"    Residuals Std: {diag_prices['residuals_std']:.6f}")
    
    # -------------------------------------------------------------------------
    # Step 3: Experiment B - Realized Volatility (Mean Reversion Test)
    # -------------------------------------------------------------------------
    print("\\n[STEP 3] Experiment B: Realized Volatility (Mean Reversion Test)")
    print("-" * 80)
    
    # Calculate returns
    returns = df["close"].pct_change()
    
    # Calculate rolling realized volatility (24-period rolling std of returns)
    realized_vol = returns.rolling(window=24).std()
    
    # Drop NaN values
    realized_vol = realized_vol.dropna()
    
    # Fit OU model
    ou_vol = OUEstimator()
    ou_vol.fit(realized_vol, dt=1.0)
    
    diag_vol = ou_vol.diagnostics()
    
    print(f"  Realized Volatility Series:")
    print(f"    Mean: {realized_vol.mean():.6f}")
    print(f"    Std:  {realized_vol.std():.6f}")
    print(f"\\n  OU Parameters:")
    print(f"    Theta:         {diag_vol['theta']:.6f}  {'[OK] Mean reverting' if diag_vol['theta'] > 0.05 else '[WARN] Weak mean reversion'}")
    print(f"    Mu:            {diag_vol['mu']:.6f}")
    print(f"    Sigma:         {diag_vol['sigma']:.6f}")
    print(f"    Half-life:     {diag_vol['half_life']:.2f} periods")
    print(f"\\n  Regression Diagnostics:")
    print(f"    Alpha:         {diag_vol['alpha']:.6f}")
    print(f"    Beta:          {diag_vol['beta']:.6f}")
    print(f"    R^2:           {diag_vol['r_squared']:.4f}")
    print(f"    Residuals Std: {diag_vol['residuals_std']:.6f}")
    
    # -------------------------------------------------------------------------
    # Step 4: Comparison Table
    # -------------------------------------------------------------------------
    print("\\n[STEP 4] Comparison Table")
    print("=" * 80)
    
    comparison_df = pd.DataFrame({
        "Metric": [
            "Theta",
            "Mu",
            "Sigma",
            "Half-life (periods)",
            "R^2",
            "Mean Reverting?",
        ],
        "Log-Prices": [
            f"{diag_prices['theta']:.6f}",
            f"{diag_prices['mu']:.4f}",
            f"{diag_prices['sigma']:.6f}",
            f"{diag_prices['half_life']:.2f}",
            f"{diag_prices['r_squared']:.4f}",
            "Yes" if diag_prices['is_mean_reverting'] else "No",
        ],
        "Realized Volatility": [
            f"{diag_vol['theta']:.6f}",
            f"{diag_vol['mu']:.6f}",
            f"{diag_vol['sigma']:.6f}",
            f"{diag_vol['half_life']:.2f}",
            f"{diag_vol['r_squared']:.4f}",
            "Yes" if diag_vol['is_mean_reverting'] else "No",
        ],
    })
    
    print(comparison_df.to_string(index=False))
    
    # -------------------------------------------------------------------------
    # Step 5: Optional Visualization
    # -------------------------------------------------------------------------
    print("\\n[STEP 5] Generating Visualization...")
    
    # Simulate one OU path for volatility
    n_steps = len(realized_vol)
    simulated_vol = ou_vol.simulate(
        n_steps=n_steps,
        n_paths=1,
        initial_value=realized_vol.iloc[0],
        dt=1.0,
        random_state=42,
    )
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Realized Volatility: Real vs Simulated OU Path",
            "Log-Prices: Real Data",
        ),
    )
    
    # Plot 1: Realized Volatility
    fig.add_trace(
        go.Scatter(
            x=realized_vol.index,
            y=realized_vol.values,
            mode="lines",
            name="Real Volatility",
            line=dict(color="#26a69a", width=1.5),
        ),
        row=1,
        col=1,
    )
    
    fig.add_trace(
        go.Scatter(
            x=realized_vol.index,
            y=simulated_vol[:, 0],
            mode="lines",
            name="Simulated OU Path",
            line=dict(color="#FFA726", width=1.5, dash="dash"),
        ),
        row=1,
        col=1,
    )
    
    # Add mean line
    fig.add_hline(
        y=diag_vol['mu'],
        line_dash="dot",
        line_color="white",
        opacity=0.5,
        annotation_text=f"mu = {diag_vol['mu']:.6f}",
        row=1,
        col=1,
    )
    
    # Plot 2: Log-Prices
    fig.add_trace(
        go.Scatter(
            x=log_prices.index,
            y=log_prices.values,
            mode="lines",
            name="Log-Prices",
            line=dict(color="#ef5350", width=1.5),
        ),
        row=2,
        col=1,
    )
    
    # Update layout
    fig.update_layout(
        title=f"OU Calibration Engine - BTCUSDT Analysis (90 days, 1H candles)",
        template=PLOT_TEMPLATE,
        height=800,
        hovermode="x unified",
        showlegend=True,
    )
    
    fig.update_yaxes(title_text="Volatility", row=1, col=1)
    fig.update_yaxes(title_text="Log(Price)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    
    # Save and open
    output_path = Path("ou_research_results.html")
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
    print("\\n[OK] Experiment A (Log-Prices):")
    print(f"  - Theta = {diag_prices['theta']:.6f} (expected ~= 0 for random walk)")
    print(f"  - Result: {'PASS - Random Walk behavior confirmed' if diag_prices['theta'] < 0.01 else 'FAIL - Not random walk'}")
    
    print("\\n[OK] Experiment B (Realized Volatility):")
    print(f"  - Theta = {diag_vol['theta']:.6f} (expected > 0 for mean reversion)")
    print(f"  - Half-life = {diag_vol['half_life']:.2f} periods")
    print(f"  - Result: {'PASS - Mean reversion confirmed' if diag_vol['theta'] > 0.05 else 'FAIL - Weak/no mean reversion'}")
    
    print("\\n" + "=" * 80)
    print("OU CALIBRATION ENGINE VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
