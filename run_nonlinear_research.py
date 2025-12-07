"""
Nonlinear Dynamics Research - Linear OU vs Nonlinear Double-Well Comparison

This script compares the linear OU baseline with the nonlinear Double-Well model
to investigate whether market volatility exhibits regime bistability.
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_loader import MarketDataLoader
from src.analysis.ou_engine import OUEstimator
from src.analysis.nonlinear_engine import DoubleWellEstimator
from src.config import PLOT_TEMPLATE


def main():
    """Compare linear OU vs nonlinear Double-Well models on volatility data."""
    
    print("=" * 80)
    print("NONLINEAR DYNAMICS RESEARCH - OU vs DOUBLE-WELL COMPARISON")
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
    
    # Calculate realized volatility (same as OU research)
    returns = df["close"].pct_change()
    realized_vol = returns.rolling(window=24).std().dropna()
    
    print(f"  Realized volatility: {len(realized_vol)} points")
    print(f"  Mean: {realized_vol.mean():.6f}, Std: {realized_vol.std():.6f}")
    
    # -------------------------------------------------------------------------
    # Step 2: Fit Linear OU Model (Baseline)
    # -------------------------------------------------------------------------
    print("\\n[STEP 2] Fitting Linear OU Model (Baseline)")
    print("-" * 80)
    
    ou_model = OUEstimator()
    ou_model.fit(realized_vol, dt=1.0)
    
    ou_diag = ou_model.diagnostics()
    
    print(f"  OU Parameters:")
    print(f"    Theta:         {ou_diag['theta']:.6f}")
    print(f"    Mu:            {ou_diag['mu']:.6f}")
    print(f"    Sigma:         {ou_diag['sigma']:.6f}")
    print(f"    Half-life:     {ou_diag['half_life']:.2f} periods")
    print(f"    R^2:           {ou_diag['r_squared']:.4f}")
    
    # -------------------------------------------------------------------------
    # Step 3: Fit Nonlinear Double-Well Model (Challenger)
    # -------------------------------------------------------------------------
    print("\\n[STEP 3] Fitting Nonlinear Double-Well Model (Challenger)")
    print("-" * 80)
    
    dw_model = DoubleWellEstimator()
    dw_model.fit(realized_vol, dt=1.0)
    
    dw_diag = dw_model.diagnostics()
    
    print(f"  Double-Well Parameters:")
    print(f"    a (linear):    {dw_diag['a']:.6f}")
    print(f"    b (cubic):     {dw_diag['b']:.6f}")
    print(f"    Sigma:         {dw_diag['sigma']:.6f}")
    print(f"    R^2:           {dw_diag['r_squared']:.4f}")
    print(f"\\n  Regime Analysis:")
    print(f"    Bistable:      {dw_diag['is_bistable']}")
    print(f"    Potential:     {dw_diag['potential_type']}")
    print(f"    Tipping pts:   {dw_diag['tipping_points']}")
    
    # -------------------------------------------------------------------------
    # Step 4: Physical Interpretation
    # -------------------------------------------------------------------------
    print("\\n[STEP 4] Physical Interpretation")
    print("=" * 80)
    
    if dw_diag['is_bistable']:
        print("  [BISTABLE SYSTEM DETECTED]")
        print(f"  The market exhibits TWO stable volatility regimes:")
        
        # Unstandardize tipping points
        tipping_pts_real = np.array(dw_diag['tipping_points']) * dw_diag['std'] + dw_diag['mean']
        
        print(f"    - Low volatility regime:  ~{tipping_pts_real[0]:.6f}")
        print(f"    - Unstable equilibrium:   ~{tipping_pts_real[1]:.6f}")
        print(f"    - High volatility regime: ~{tipping_pts_real[2]:.6f}")
        print(f"\\n  Current market state: {realized_vol.iloc[-1]:.6f}")
        
        # Determine which regime we're in
        current_std = (realized_vol.iloc[-1] - dw_diag['mean']) / dw_diag['std']
        if abs(current_std - dw_diag['tipping_points'][0]) < abs(current_std - dw_diag['tipping_points'][2]):
            print(f"  -> Currently in LOW volatility regime")
        else:
            print(f"  -> Currently in HIGH volatility regime")
    else:
        print("  [MONOSTABLE SYSTEM]")
        print(f"  The market exhibits a SINGLE stable regime (like linear OU).")
        print(f"  No evidence of bistability in the current data.")
    
    # -------------------------------------------------------------------------
    # Step 5: Comparison Table
    # -------------------------------------------------------------------------
    print("\\n[STEP 5] Model Comparison")
    print("=" * 80)
    
    comparison_df = pd.DataFrame({
        "Metric": [
            "Model Type",
            "Theta / a",
            "Mu (OU) / b (DW)",
            "Sigma",
            "R^2",
            "Regime Structure",
        ],
        "Linear OU": [
            "Linear Mean Reversion",
            f"{ou_diag['theta']:.6f}",
            f"{ou_diag['mu']:.6f}",
            f"{ou_diag['sigma']:.6f}",
            f"{ou_diag['r_squared']:.4f}",
            "Single equilibrium",
        ],
        "Nonlinear DW": [
            "Cubic Drift",
            f"{dw_diag['a']:.6f}",
            f"{dw_diag['b']:.6f}",
            f"{dw_diag['sigma']:.6f}",
            f"{dw_diag['r_squared']:.4f}",
            dw_diag['potential_type'],
        ],
    })
    
    print(comparison_df.to_string(index=False))
    
    # -------------------------------------------------------------------------
    # Step 6: Visualization
    # -------------------------------------------------------------------------
    print("\\n[STEP 6] Generating Visualization...")
    
    # Simulate paths
    n_steps = len(realized_vol)
    
    # OU simulation (unstandardized)
    ou_sim_std = ou_model.simulate(
        n_steps=n_steps,
        n_paths=1,
        initial_value=realized_vol.iloc[0],
        dt=1.0,
        random_state=42,
    )
    
    # DW simulation (standardized, then unstandardize)
    initial_std = (realized_vol.iloc[0] - dw_diag['mean']) / dw_diag['std']
    dw_sim_std = dw_model.simulate(
        n_steps=n_steps,
        n_paths=1,
        initial_value=initial_std,
        dt=1.0,
        random_state=42,
    )
    dw_sim = dw_sim_std[:, 0] * dw_diag['std'] + dw_diag['mean']
    
    # Get potential shape
    x_pot, V_pot = dw_model.get_potential_shape(x_range=(-3, 3), n_points=300)
    
    # Current market state in standardized space
    current_state_std = (realized_vol.iloc[-1] - dw_diag['mean']) / dw_diag['std']
    current_V = -0.5 * dw_diag['a'] * current_state_std ** 2 + 0.25 * dw_diag['b'] * current_state_std ** 4
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.12,
        subplot_titles=(
            "Time Series: Real Volatility vs Model Simulations",
            "Potential Landscape V(x) - Regime Structure",
        ),
        row_heights=[0.5, 0.5],
    )
    
    # Plot 1: Time Series
    fig.add_trace(
        go.Scatter(
            x=realized_vol.index,
            y=realized_vol.values,
            mode="lines",
            name="Real Volatility",
            line=dict(color="#26a69a", width=2),
        ),
        row=1,
        col=1,
    )
    
    fig.add_trace(
        go.Scatter(
            x=realized_vol.index,
            y=ou_sim_std[:, 0],
            mode="lines",
            name="OU Simulation",
            line=dict(color="#FFA726", width=1.5, dash="dash"),
            opacity=0.7,
        ),
        row=1,
        col=1,
    )
    
    fig.add_trace(
        go.Scatter(
            x=realized_vol.index,
            y=dw_sim,
            mode="lines",
            name="Double-Well Simulation",
            line=dict(color="#ef5350", width=1.5, dash="dot"),
            opacity=0.7,
        ),
        row=1,
        col=1,
    )
    
    # Plot 2: Potential Landscape
    fig.add_trace(
        go.Scatter(
            x=x_pot,
            y=V_pot,
            mode="lines",
            name="Potential V(x)",
            line=dict(color="#42A5F5", width=3),
            showlegend=True,
        ),
        row=2,
        col=1,
    )
    
    # Mark equilibrium points
    for i, tp in enumerate(dw_diag['tipping_points']):
        V_tp = -0.5 * dw_diag['a'] * tp ** 2 + 0.25 * dw_diag['b'] * tp ** 4
        marker_color = "#4CAF50" if i != 1 else "#FF5722"  # Green for stable, red for unstable
        marker_name = "Stable" if i != 1 else "Unstable"
        
        fig.add_trace(
            go.Scatter(
                x=[tp],
                y=[V_tp],
                mode="markers",
                name=f"{marker_name} Equilibrium",
                marker=dict(size=12, color=marker_color, symbol="circle"),
                showlegend=(i == 0 or i == 1),
            ),
            row=2,
            col=1,
        )
    
    # Mark current market state
    fig.add_trace(
        go.Scatter(
            x=[current_state_std],
            y=[current_V],
            mode="markers",
            name="Current State",
            marker=dict(size=15, color="#FFC107", symbol="star", line=dict(width=2, color="white")),
        ),
        row=2,
        col=1,
    )
    
    # Update layout
    fig.update_layout(
        title="Nonlinear Dynamics: OU vs Double-Well Potential Analysis",
        template=PLOT_TEMPLATE,
        height=900,
        hovermode="x unified",
        showlegend=True,
    )
    
    fig.update_yaxes(title_text="Volatility", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=1)
    
    fig.update_yaxes(title_text="Potential V(x)", row=2, col=1)
    fig.update_xaxes(title_text="State x (standardized)", row=2, col=1)
    
    # Save and open
    output_path = Path("nonlinear_research_results.html")
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
    
    print("\\n[LINEAR OU MODEL]")
    print(f"  - Single equilibrium at mu = {ou_diag['mu']:.6f}")
    print(f"  - Mean reversion with theta = {ou_diag['theta']:.6f}")
    print(f"  - R^2 = {ou_diag['r_squared']:.4f}")
    
    print("\\n[NONLINEAR DOUBLE-WELL MODEL]")
    print(f"  - Potential type: {dw_diag['potential_type']}")
    print(f"  - Coefficients: a = {dw_diag['a']:.6f}, b = {dw_diag['b']:.6f}")
    print(f"  - R^2 = {dw_diag['r_squared']:.4f}")
    
    if dw_diag['is_bistable']:
        print(f"  - BISTABILITY CONFIRMED: Two stable regimes detected")
        print(f"  - Regime switching possible at tipping points")
    else:
        print(f"  - No bistability detected (monostable like OU)")
    
    print("\\n" + "=" * 80)
    print("NONLINEAR DYNAMICS RESEARCH COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
