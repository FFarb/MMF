"""
Asset Physics Comparison Tool

This script compares the market physics between two assets (e.g., BTC vs ADA)
to diagnose why trading strategies perform differently.

Objectives:
1. Load historical data for both assets
2. Compute physics metrics using PhysicsProfiler
3. Generate comprehensive visualizations
4. Produce automated diagnostic report

Use Case:
Understand why Trend Expert works on BTC but fails on ADA by identifying
fundamental differences in market regimes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config import PLOT_TEMPLATE
from src.data_loader import MarketDataLoader
from src.analysis.physics_profiler import PhysicsProfiler, compare_assets


def create_comparison_visualizations(
    profiler_a: PhysicsProfiler,
    profiler_b: PhysicsProfiler,
    asset_a_name: str,
    asset_b_name: str,
    output_path: str = "physics_comparison.html",
) -> None:
    """
    Create comprehensive visualization comparing two assets.
    
    Parameters
    ----------
    profiler_a : PhysicsProfiler
        Fitted profiler for asset A
    profiler_b : PhysicsProfiler
        Fitted profiler for asset B
    asset_a_name : str
        Name of asset A
    asset_b_name : str
        Name of asset B
    output_path : str
        Path to save HTML visualization
    """
    print(f"\n[Visualization] Creating comparison charts...")
    
    # Create subplot figure
    fig = make_subplots(
        rows=4,
        cols=2,
        subplot_titles=(
            f"Hurst Exponent Distribution: {asset_a_name} vs {asset_b_name}",
            f"Rolling Hurst Over Time",
            f"Stability Theta (OU) Distribution",
            f"Rolling Theta Over Time",
            f"Sample Entropy Distribution",
            f"Entropy vs Volatility-of-Volatility",
            f"FracDiff Correlation Distribution",
            f"Volatility-of-Volatility Distribution",
        ),
        specs=[
            [{"type": "histogram"}, {"type": "scatter"}],
            [{"type": "histogram"}, {"type": "scatter"}],
            [{"type": "histogram"}, {"type": "scatter"}],
            [{"type": "histogram"}, {"type": "histogram"}],
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.12,
    )
    
    # Color scheme
    color_a = "#1f77b4"  # Blue
    color_b = "#ff7f0e"  # Orange
    
    # Get data
    data_a = profiler_a.rolling_results_
    data_b = profiler_b.rolling_results_
    
    # Row 1: Hurst Exponent
    # Histogram
    fig.add_trace(
        go.Histogram(
            x=data_a['hurst'].dropna(),
            name=asset_a_name,
            marker_color=color_a,
            opacity=0.7,
            nbinsx=50,
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(
            x=data_b['hurst'].dropna(),
            name=asset_b_name,
            marker_color=color_b,
            opacity=0.7,
            nbinsx=50,
        ),
        row=1, col=1
    )
    
    # Rolling Hurst
    fig.add_trace(
        go.Scatter(
            x=data_a.index,
            y=data_a['hurst'],
            name=f"{asset_a_name} Hurst",
            line=dict(color=color_a, width=1),
            mode='lines',
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=data_b.index,
            y=data_b['hurst'],
            name=f"{asset_b_name} Hurst",
            line=dict(color=color_b, width=1),
            mode='lines',
        ),
        row=1, col=2
    )
    
    # Add reference line at H=0.5 (random walk)
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=2)
    
    # Row 2: Stability Theta
    # Histogram
    fig.add_trace(
        go.Histogram(
            x=data_a['theta'].dropna(),
            name=f"{asset_a_name} Theta",
            marker_color=color_a,
            opacity=0.7,
            nbinsx=50,
            showlegend=False,
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Histogram(
            x=data_b['theta'].dropna(),
            name=f"{asset_b_name} Theta",
            marker_color=color_b,
            opacity=0.7,
            nbinsx=50,
            showlegend=False,
        ),
        row=2, col=1
    )
    
    # Rolling Theta
    fig.add_trace(
        go.Scatter(
            x=data_a.index,
            y=data_a['theta'],
            name=f"{asset_a_name} Theta",
            line=dict(color=color_a, width=1),
            mode='lines',
            showlegend=False,
        ),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=data_b.index,
            y=data_b['theta'],
            name=f"{asset_b_name} Theta",
            line=dict(color=color_b, width=1),
            mode='lines',
            showlegend=False,
        ),
        row=2, col=2
    )
    
    # Row 3: Sample Entropy
    # Histogram
    fig.add_trace(
        go.Histogram(
            x=data_a['entropy'].dropna(),
            name=f"{asset_a_name} Entropy",
            marker_color=color_a,
            opacity=0.7,
            nbinsx=50,
            showlegend=False,
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Histogram(
            x=data_b['entropy'].dropna(),
            name=f"{asset_b_name} Entropy",
            marker_color=color_b,
            opacity=0.7,
            nbinsx=50,
            showlegend=False,
        ),
        row=3, col=1
    )
    
    # Scatter: Entropy vs Vol-of-Vol
    fig.add_trace(
        go.Scatter(
            x=data_a['entropy'].dropna(),
            y=data_a['vol_of_vol'].dropna(),
            name=f"{asset_a_name}",
            mode='markers',
            marker=dict(color=color_a, size=3, opacity=0.5),
            showlegend=False,
        ),
        row=3, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=data_b['entropy'].dropna(),
            y=data_b['vol_of_vol'].dropna(),
            name=f"{asset_b_name}",
            mode='markers',
            marker=dict(color=color_b, size=3, opacity=0.5),
            showlegend=False,
        ),
        row=3, col=2
    )
    
    # Row 4: FracDiff Correlation and Vol-of-Vol
    # FracDiff Correlation Histogram
    fig.add_trace(
        go.Histogram(
            x=data_a['fracdiff_corr'].dropna(),
            name=f"{asset_a_name} FracDiff",
            marker_color=color_a,
            opacity=0.7,
            nbinsx=50,
            showlegend=False,
        ),
        row=4, col=1
    )
    fig.add_trace(
        go.Histogram(
            x=data_b['fracdiff_corr'].dropna(),
            name=f"{asset_b_name} FracDiff",
            marker_color=color_b,
            opacity=0.7,
            nbinsx=50,
            showlegend=False,
        ),
        row=4, col=1
    )
    
    # Vol-of-Vol Histogram
    fig.add_trace(
        go.Histogram(
            x=data_a['vol_of_vol'].dropna(),
            name=f"{asset_a_name} Vol-of-Vol",
            marker_color=color_a,
            opacity=0.7,
            nbinsx=50,
            showlegend=False,
        ),
        row=4, col=2
    )
    fig.add_trace(
        go.Histogram(
            x=data_b['vol_of_vol'].dropna(),
            name=f"{asset_b_name} Vol-of-Vol",
            marker_color=color_b,
            opacity=0.7,
            nbinsx=50,
            showlegend=False,
        ),
        row=4, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=f"Market Physics Comparison: {asset_a_name} vs {asset_b_name}",
        template=PLOT_TEMPLATE,
        height=1600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        barmode='overlay',
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Hurst Exponent", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Hurst", row=1, col=2)
    
    fig.update_xaxes(title_text="Theta (Mean Reversion Speed)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=2)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_yaxes(title_text="Theta", row=2, col=2)
    
    fig.update_xaxes(title_text="Sample Entropy", row=3, col=1)
    fig.update_xaxes(title_text="Entropy", row=3, col=2)
    fig.update_yaxes(title_text="Frequency", row=3, col=1)
    fig.update_yaxes(title_text="Vol-of-Vol", row=3, col=2)
    
    fig.update_xaxes(title_text="FracDiff Correlation", row=4, col=1)
    fig.update_xaxes(title_text="Volatility-of-Volatility", row=4, col=2)
    fig.update_yaxes(title_text="Frequency", row=4, col=1)
    fig.update_yaxes(title_text="Frequency", row=4, col=2)
    
    # Save
    fig.write_html(output_path)
    print(f"  [OK] Saved visualization to {output_path}")


def generate_diagnostic_report(
    profiler_a: PhysicsProfiler,
    profiler_b: PhysicsProfiler,
    comparison_df: pd.DataFrame,
    asset_a_name: str,
    asset_b_name: str,
    output_path: str = "physics_diagnostic_report.txt",
) -> None:
    """
    Generate a comprehensive text report with automated conclusions.
    
    Parameters
    ----------
    profiler_a : PhysicsProfiler
        Fitted profiler for asset A
    profiler_b : PhysicsProfiler
        Fitted profiler for asset B
    comparison_df : pd.DataFrame
        Comparison dataframe
    asset_a_name : str
        Name of asset A
    asset_b_name : str
        Name of asset B
    output_path : str
        Path to save report
    """
    print(f"\n[Report] Generating diagnostic report...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MARKET PHYSICS DIAGNOSTIC REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Assets Compared: {asset_a_name} vs {asset_b_name}\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary Statistics
        f.write("-" * 80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n\n")
        
        f.write(f"{asset_a_name} Metrics:\n")
        f.write(profiler_a.get_summary().to_string())
        f.write("\n\n")
        
        f.write(f"{asset_b_name} Metrics:\n")
        f.write(profiler_b.get_summary().to_string())
        f.write("\n\n")
        
        # Comparison Table
        f.write("-" * 80 + "\n")
        f.write("COMPARATIVE ANALYSIS\n")
        f.write("-" * 80 + "\n\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        # Diagnostics
        f.write("-" * 80 + "\n")
        f.write(f"{asset_a_name} DIAGNOSTICS\n")
        f.write("-" * 80 + "\n\n")
        for metric, diagnosis in profiler_a.diagnose().items():
            f.write(f"  {metric:15s}: {diagnosis}\n")
        f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write(f"{asset_b_name} DIAGNOSTICS\n")
        f.write("-" * 80 + "\n\n")
        for metric, diagnosis in profiler_b.diagnose().items():
            f.write(f"  {metric:15s}: {diagnosis}\n")
        f.write("\n")
        
        # Automated Conclusions
        f.write("=" * 80 + "\n")
        f.write("AUTOMATED CONCLUSIONS\n")
        f.write("=" * 80 + "\n\n")
        
        # Analyze key differences
        hurst_diff = profiler_a.results_['hurst']['mean'] - profiler_b.results_['hurst']['mean']
        theta_diff = profiler_a.results_['theta']['mean'] - profiler_b.results_['theta']['mean']
        entropy_diff = profiler_a.results_['entropy']['mean'] - profiler_b.results_['entropy']['mean']
        
        f.write("Key Findings:\n\n")
        
        # Hurst Analysis
        if abs(hurst_diff) > 0.05:
            if hurst_diff > 0:
                f.write(f"1. HURST EXPONENT: {asset_a_name} has significantly higher Hurst ({profiler_a.results_['hurst']['mean']:.3f}) "
                       f"than {asset_b_name} ({profiler_b.results_['hurst']['mean']:.3f}).\n")
                f.write(f"   → {asset_a_name} exhibits stronger trend persistence.\n")
                f.write(f"   → Trend Expert should perform BETTER on {asset_a_name}.\n")
                f.write(f"   → {asset_b_name} may be too mean-reverting for trend strategies.\n\n")
            else:
                f.write(f"1. HURST EXPONENT: {asset_b_name} has significantly higher Hurst ({profiler_b.results_['hurst']['mean']:.3f}) "
                       f"than {asset_a_name} ({profiler_a.results_['hurst']['mean']:.3f}).\n")
                f.write(f"   → {asset_b_name} exhibits stronger trend persistence.\n")
                f.write(f"   → Trend Expert should perform BETTER on {asset_b_name}.\n")
                f.write(f"   → {asset_a_name} may be too mean-reverting for trend strategies.\n\n")
        else:
            f.write(f"1. HURST EXPONENT: Both assets have similar Hurst values (~{profiler_a.results_['hurst']['mean']:.3f}).\n")
            f.write(f"   → Trend persistence is comparable between assets.\n\n")
        
        # Theta Analysis
        if abs(theta_diff) > 0.05:
            if theta_diff > 0:
                f.write(f"2. MEAN REVERSION SPEED: {asset_a_name} has higher theta ({profiler_a.results_['theta']['mean']:.3f}) "
                       f"than {asset_b_name} ({profiler_b.results_['theta']['mean']:.3f}).\n")
                f.write(f"   → {asset_a_name} reverts to mean faster.\n")
                f.write(f"   → Range Expert should perform BETTER on {asset_a_name}.\n\n")
            else:
                f.write(f"2. MEAN REVERSION SPEED: {asset_b_name} has higher theta ({profiler_b.results_['theta']['mean']:.3f}) "
                       f"than {asset_a_name} ({profiler_a.results_['theta']['mean']:.3f}).\n")
                f.write(f"   → {asset_b_name} reverts to mean faster.\n")
                f.write(f"   → Range Expert should perform BETTER on {asset_b_name}.\n\n")
        else:
            f.write(f"2. MEAN REVERSION SPEED: Both assets have similar theta values (~{profiler_a.results_['theta']['mean']:.3f}).\n\n")
        
        # Entropy Analysis
        if abs(entropy_diff) > 0.3:
            if entropy_diff > 0:
                f.write(f"3. NOISE LEVEL: {asset_a_name} has higher entropy ({profiler_a.results_['entropy']['mean']:.3f}) "
                       f"than {asset_b_name} ({profiler_b.results_['entropy']['mean']:.3f}).\n")
                f.write(f"   → {asset_a_name} is noisier and less predictable.\n")
                f.write(f"   → All strategies may struggle more on {asset_a_name}.\n\n")
            else:
                f.write(f"3. NOISE LEVEL: {asset_b_name} has higher entropy ({profiler_b.results_['entropy']['mean']:.3f}) "
                       f"than {asset_a_name} ({profiler_a.results_['entropy']['mean']:.3f}).\n")
                f.write(f"   → {asset_b_name} is noisier and less predictable.\n")
                f.write(f"   → All strategies may struggle more on {asset_b_name}.\n\n")
        else:
            f.write(f"3. NOISE LEVEL: Both assets have similar entropy (~{profiler_a.results_['entropy']['mean']:.3f}).\n\n")
        
        # Strategy Recommendations
        f.write("-" * 80 + "\n")
        f.write("STRATEGY RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n\n")
        
        # For Asset A
        f.write(f"{asset_a_name}:\n")
        if profiler_a.results_['hurst']['mean'] > 0.55:
            f.write("  [+] Enable Trend Expert (High Hurst)\n")
        else:
            f.write("  [-] Disable Trend Expert (Low Hurst)\n")
        
        if profiler_a.results_['theta']['mean'] > 0.1:
            f.write("  [+] Enable Range Expert (High Theta)\n")
        else:
            f.write("  [-] Disable Range Expert (Low Theta)\n")
        
        if profiler_a.results_['vol_of_vol']['std'] / profiler_a.results_['vol_of_vol']['mean'] > 1.0:
            f.write("  [+] Enable Stress Expert (High Vol-of-Vol)\n")
        else:
            f.write("  ~ Stress Expert optional\n")
        f.write("\n")
        
        # For Asset B
        f.write(f"{asset_b_name}:\n")
        if profiler_b.results_['hurst']['mean'] > 0.55:
            f.write("  [+] Enable Trend Expert (High Hurst)\n")
        else:
            f.write("  [-] Disable Trend Expert (Low Hurst)\n")
        
        if profiler_b.results_['theta']['mean'] > 0.1:
            f.write("  [+] Enable Range Expert (High Theta)\n")
        else:
            f.write("  [-] Disable Range Expert (Low Theta)\n")
        
        if profiler_b.results_['vol_of_vol']['std'] / profiler_b.results_['vol_of_vol']['mean'] > 1.0:
            f.write("  [+] Enable Stress Expert (High Vol-of-Vol)\n")
        else:
            f.write("  ~ Stress Expert optional\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"  [OK] Saved diagnostic report to {output_path}")


def main():
    """Main execution function."""
    print("=" * 72)
    print("ASSET PHYSICS COMPARISON TOOL")
    print("=" * 72)
    
    # Configuration
    ASSET_A = "BTCUSDT"
    ASSET_B = "ADAUSDT"
    INTERVAL = "60"  # 1-hour candles
    DAYS_BACK = 730  # 2 years
    
    print(f"\nConfiguration:")
    print(f"  Asset A: {ASSET_A}")
    print(f"  Asset B: {ASSET_B}")
    print(f"  Interval: {INTERVAL}m")
    print(f"  History: {DAYS_BACK} days (~2 years)")
    
    # Step 1: Load Data
    print(f"\n{'=' * 72}")
    print("STEP 1: DATA LOADING")
    print(f"{'=' * 72}")
    
    loader_a = MarketDataLoader(symbol=ASSET_A, interval=INTERVAL)
    df_a = loader_a.get_data(days_back=DAYS_BACK)
    
    loader_b = MarketDataLoader(symbol=ASSET_B, interval=INTERVAL)
    df_b = loader_b.get_data(days_back=DAYS_BACK)
    
    print(f"\n[Data] {ASSET_A}: {len(df_a)} candles")
    print(f"[Data] {ASSET_B}: {len(df_b)} candles")
    
    # Step 2: Compute Physics Metrics
    print(f"\n{'=' * 72}")
    print("STEP 2: PHYSICS PROFILING")
    print(f"{'=' * 72}")
    
    profiler_a, profiler_b, comparison = compare_assets(
        asset_a_prices=df_a['close'],
        asset_b_prices=df_b['close'],
        asset_a_name=ASSET_A,
        asset_b_name=ASSET_B,
        verbose=True,
    )
    
    # Step 3: Visualizations
    print(f"\n{'=' * 72}")
    print("STEP 3: VISUALIZATION")
    print(f"{'=' * 72}")
    
    create_comparison_visualizations(
        profiler_a=profiler_a,
        profiler_b=profiler_b,
        asset_a_name=ASSET_A,
        asset_b_name=ASSET_B,
        output_path="physics_comparison.html",
    )
    
    # Step 4: Generate Report
    print(f"\n{'=' * 72}")
    print("STEP 4: DIAGNOSTIC REPORT")
    print(f"{'=' * 72}")
    
    generate_diagnostic_report(
        profiler_a=profiler_a,
        profiler_b=profiler_b,
        comparison_df=comparison,
        asset_a_name=ASSET_A,
        asset_b_name=ASSET_B,
        output_path="physics_diagnostic_report.txt",
    )
    
    # Step 5: Summary Table
    print(f"\n{'=' * 72}")
    print("STEP 5: SUMMARY TABLE")
    print(f"{'=' * 72}")
    
    # Create summary table
    summary_data = []
    for metric in ['hurst', 'theta', 'entropy', 'fracdiff_corr', 'vol_of_vol']:
        summary_data.append({
            'Metric': metric.upper(),
            f'{ASSET_A}_Mean': f"{profiler_a.results_[metric]['mean']:.4f}",
            f'{ASSET_A}_Std': f"{profiler_a.results_[metric]['std']:.4f}",
            f'{ASSET_B}_Mean': f"{profiler_b.results_[metric]['mean']:.4f}",
            f'{ASSET_B}_Std': f"{profiler_b.results_[metric]['std']:.4f}",
            'Difference': f"{profiler_a.results_[metric]['mean'] - profiler_b.results_[metric]['mean']:.4f}",
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
    
    # Save summary to CSV
    summary_df.to_csv("physics_comparison_summary.csv", index=False)
    print(f"\n  [OK] Saved summary table to physics_comparison_summary.csv")
    
    # Final Conclusion
    print(f"\n{'=' * 72}")
    print("AUTOMATED CONCLUSION")
    print(f"{'=' * 72}")
    
    hurst_a = profiler_a.results_['hurst']['mean']
    hurst_b = profiler_b.results_['hurst']['mean']
    
    if hurst_a > 0.55 and hurst_b < 0.45:
        print(f"\n[DIAGNOSIS] {ASSET_A} has significantly higher Hurst ({hurst_a:.3f}) than {ASSET_B} ({hurst_b:.3f}).")
        print(f"   -> {ASSET_A} is TRENDING, {ASSET_B} is MEAN-REVERTING.")
        print(f"   -> Trend Expert works on {ASSET_A} but FAILS on {ASSET_B} due to fundamental regime difference.")
        print(f"   -> SOLUTION: Disable Trend Expert for low-Hurst assets like {ASSET_B}.")
    elif hurst_b > 0.55 and hurst_a < 0.45:
        print(f"\n[DIAGNOSIS] {ASSET_B} has significantly higher Hurst ({hurst_b:.3f}) than {ASSET_A} ({hurst_a:.3f}).")
        print(f"   -> {ASSET_B} is TRENDING, {ASSET_A} is MEAN-REVERTING.")
        print(f"   -> Trend Expert works on {ASSET_B} but FAILS on {ASSET_A} due to fundamental regime difference.")
        print(f"   -> SOLUTION: Disable Trend Expert for low-Hurst assets like {ASSET_A}.")
    else:
        print(f"\n[DIAGNOSIS] {ASSET_A} (H={hurst_a:.3f}) and {ASSET_B} (H={hurst_b:.3f}) have similar Hurst values.")
        print(f"   -> Trend persistence is comparable.")
        print(f"   -> Look at other metrics (Theta, Entropy, Vol-of-Vol) for differentiation.")
    
    print(f"\n{'=' * 72}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 72}")
    print("\nGenerated Files:")
    print("  1. physics_comparison.html - Interactive visualizations")
    print("  2. physics_diagnostic_report.txt - Detailed text report")
    print("  3. physics_comparison_summary.csv - Summary table")
    print("\n")


if __name__ == "__main__":
    main()

