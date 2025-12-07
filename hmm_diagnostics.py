"""
Gaussian Diagnostics for HMM Paper Trading Bot

This script visualizes the internal Gaussian distributions of Hidden Markov Models
to evaluate whether using only log returns is sufficient, or if adding volatility
features would improve state separation.

Outputs:
    - hmm_diagnostics.html: Interactive visualization comparing single vs. multi-feature HMMs
"""

import sys
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import multivariate_normal
from hmmlearn.hmm import GaussianHMM
from pybit.unified_trading import HTTP
import warnings
warnings.filterwarnings('ignore')

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configuration
SYMBOL = "BTCUSDT"
INTERVAL = "5"  # 5-minute candles
CANDLE_LIMIT = 1000
VOLATILITY_WINDOW = 24
N_STATES = 3

# Color scheme for states
STATE_COLORS = {
    'BULL': '#00ff88',    # Green
    'BEAR': '#ff4444',    # Red
    'NEUTRAL': '#ffaa00'  # Orange
}

def fetch_candles(limit: int = CANDLE_LIMIT) -> pd.DataFrame:
    """Fetch historical candles from Bybit."""
    print(f"📊 Fetching {limit} candles of {INTERVAL}m data for {SYMBOL}...")
    
    client = HTTP(testnet=False)
    resp = client.get_kline(
        category="linear",
        symbol=SYMBOL,
        interval=INTERVAL,
        limit=limit
    )
    
    if resp.get("retCode") != 0:
        raise ValueError(f"API error: {resp.get('retMsg', 'Unknown error')}")
    
    data_list = resp.get("result", {}).get("list", [])
    if not data_list:
        raise ValueError("Empty result from API")
    
    df = pd.DataFrame(data_list, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df.sort_index()
    
    print(f"✅ Fetched {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate log returns and rolling volatility."""
    print("🔧 Engineering features...")
    
    # Log returns
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Rolling volatility (standard deviation of returns)
    df['volatility'] = df['log_returns'].rolling(window=VOLATILITY_WINDOW).std()
    
    # Drop NaN values
    df = df.dropna()
    
    print(f"✅ Features engineered. {len(df)} valid samples.")
    return df

def train_hmm_single_feature(df: pd.DataFrame) -> tuple:
    """Train HMM on log returns only (current bot logic)."""
    print("🤖 Training HMM on Log Returns only...")
    
    X = df['log_returns'].values.reshape(-1, 1)
    
    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="full",
        n_iter=1000,
        random_state=42
    )
    model.fit(X)
    
    # Predict states
    states = model.predict(X)
    
    # Label states based on mean returns
    state_means = []
    for i in range(N_STATES):
        state_data = X[states == i]
        if len(state_data) > 0:
            state_means.append(np.nanmean(state_data))
        else:
            state_means.append(np.nan)
    
    # Filter out NaN values for labeling
    valid_indices = [i for i, m in enumerate(state_means) if not np.isnan(m)]
    valid_means = [state_means[i] for i in valid_indices]
    
    if len(valid_means) < 2:
        raise ValueError("Not enough valid states to label")
    
    max_idx = valid_indices[int(np.argmax(valid_means))]
    min_idx = valid_indices[int(np.argmin(valid_means))]
    neutral_idx = next((i for i in valid_indices if i not in (max_idx, min_idx)), valid_indices[0])
    
    label_map = {max_idx: 'BULL', min_idx: 'BEAR', neutral_idx: 'NEUTRAL'}
    
    print(f"✅ Single-feature HMM trained. State means: {[f'{m:.6f}' if not np.isnan(m) else 'N/A' for m in state_means]}")
    
    return model, states, label_map

def train_hmm_multi_feature(df: pd.DataFrame) -> tuple:
    """Train HMM on log returns + volatility."""
    print("🤖 Training HMM on Log Returns + Volatility...")
    
    # Standardize features to have similar scales (prevents covariance issues)
    X_raw = df[['log_returns', 'volatility']].values
    
    # Z-score normalization
    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0)
    X = (X_raw - X_mean) / X_std
    
    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="full",
        n_iter=1000,
        random_state=42
    )
    model.fit(X)
    
    # Predict states
    states = model.predict(X)
    
    # Label states based on mean returns (first feature, in original scale)
    state_means = []
    for i in range(N_STATES):
        state_data = X_raw[states == i, 0]
        if len(state_data) > 0:
            state_means.append(np.nanmean(state_data))
        else:
            state_means.append(np.nan)
    
    # Filter out NaN values for labeling
    valid_indices = [i for i, m in enumerate(state_means) if not np.isnan(m)]
    valid_means = [state_means[i] for i in valid_indices]
    
    if len(valid_means) < 2:
        raise ValueError("Not enough valid states to label")
    
    max_idx = valid_indices[int(np.argmax(valid_means))]
    min_idx = valid_indices[int(np.argmin(valid_means))]
    neutral_idx = next((i for i in valid_indices if i not in (max_idx, min_idx)), valid_indices[0])
    
    label_map = {max_idx: 'BULL', min_idx: 'BEAR', neutral_idx: 'NEUTRAL'}
    
    print(f"✅ Multi-feature HMM trained. State means: {[f'{m:.6f}' if not np.isnan(m) else 'N/A' for m in state_means]}")
    
    # Store normalization parameters for later use
    model._X_mean = X_mean
    model._X_std = X_std
    
    return model, states, label_map

def plot_single_feature_diagnostics(df: pd.DataFrame, model: GaussianHMM, states: np.ndarray, label_map: dict) -> go.Figure:
    """Create histogram with overlaid Gaussian PDFs."""
    print("📈 Creating single-feature diagnostic plot...")
    
    fig = go.Figure()
    
    # Create histogram for each state
    for state_idx, state_label in label_map.items():
        state_data = df['log_returns'].values[states == state_idx]
        
        fig.add_trace(go.Histogram(
            x=state_data,
            name=state_label,
            opacity=0.6,
            marker_color=STATE_COLORS[state_label],
            nbinsx=50,
            histnorm='probability density'
        ))
    
    # Overlay Gaussian PDFs
    x_range = np.linspace(df['log_returns'].min(), df['log_returns'].max(), 500)
    
    for state_idx, state_label in label_map.items():
        mean = model.means_[state_idx][0]
        std = np.sqrt(model.covars_[state_idx][0, 0])
        
        # Calculate PDF
        pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean) / std) ** 2)
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=pdf,
            name=f'{state_label} PDF',
            line=dict(color=STATE_COLORS[state_label], width=3),
            mode='lines'
        ))
    
    fig.update_layout(
        title={
            'text': '📊 Current Bot Logic: Log Returns Only<br><sub>Histogram + Gaussian PDFs by State</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Log Returns',
        yaxis_title='Probability Density',
        barmode='overlay',
        template='plotly_dark',
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_multi_feature_diagnostics(df: pd.DataFrame, model: GaussianHMM, states: np.ndarray, label_map: dict) -> go.Figure:
    """Create scatter plot with covariance ellipses."""
    print("📈 Creating multi-feature diagnostic plot...")
    
    fig = go.Figure()
    
    # Scatter plot for each state
    for state_idx, state_label in label_map.items():
        state_mask = states == state_idx
        
        fig.add_trace(go.Scatter(
            x=df['log_returns'].values[state_mask],
            y=df['volatility'].values[state_mask],
            mode='markers',
            name=state_label,
            marker=dict(
                color=STATE_COLORS[state_label],
                size=4,
                opacity=0.6
            )
        ))
    
    # Draw covariance ellipses for each state
    theta = np.linspace(0, 2 * np.pi, 100)
    
    # Denormalize means and covariances back to original scale
    X_mean = model._X_mean
    X_std = model._X_std
    
    for state_idx, state_label in label_map.items():
        # Denormalize mean
        mean_normalized = model.means_[state_idx]
        mean = mean_normalized * X_std + X_mean
        
        # Denormalize covariance
        cov_normalized = model.covars_[state_idx]
        # Cov_original = diag(std) @ Cov_normalized @ diag(std)
        cov = np.diag(X_std) @ cov_normalized @ np.diag(X_std)
        
        # Eigenvalues and eigenvectors for ellipse
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Handle complex eigenvalues (shouldn't happen but just in case)
        if np.iscomplexobj(eigenvalues):
            eigenvalues = np.real(eigenvalues)
            eigenvectors = np.real(eigenvectors)
        
        # Ensure positive eigenvalues
        eigenvalues = np.abs(eigenvalues)
        
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        
        # 2-sigma ellipse (95% confidence)
        width = 2 * np.sqrt(eigenvalues[0]) * 2
        height = 2 * np.sqrt(eigenvalues[1]) * 2
        
        # Ellipse points
        ellipse_x = mean[0] + (width/2) * np.cos(theta) * np.cos(angle) - (height/2) * np.sin(theta) * np.sin(angle)
        ellipse_y = mean[1] + (width/2) * np.cos(theta) * np.sin(angle) + (height/2) * np.sin(theta) * np.cos(angle)
        
        fig.add_trace(go.Scatter(
            x=ellipse_x,
            y=ellipse_y,
            mode='lines',
            name=f'{state_label} 95% Ellipse',
            line=dict(color=STATE_COLORS[state_label], width=3, dash='dash'),
            showlegend=True
        ))
        
        # Add mean marker
        fig.add_trace(go.Scatter(
            x=[mean[0]],
            y=[mean[1]],
            mode='markers',
            name=f'{state_label} Mean',
            marker=dict(
                color=STATE_COLORS[state_label],
                size=15,
                symbol='star',
                line=dict(color='white', width=2)
            ),
            showlegend=False
        ))
    
    fig.update_layout(
        title={
            'text': '🎯 Enhanced Logic: Returns + Volatility<br><sub>Scatter Plot with 95% Confidence Ellipses</sub>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Log Returns',
        yaxis_title='Rolling Volatility (24-period)',
        template='plotly_dark',
        height=600,
        hovermode='closest',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig

def calculate_separation_metrics(model: GaussianHMM, states: np.ndarray, label_map: dict, feature_dim: int) -> dict:
    """Calculate metrics to quantify state separation."""
    print("📊 Calculating separation metrics...")
    
    metrics = {}
    
    # Bhattacharyya distance between states (for 1D Gaussians)
    if feature_dim == 1:
        distances = {}
        # Only iterate over states that exist in label_map
        valid_states = list(label_map.keys())
        for i_idx, i in enumerate(valid_states):
            for j in valid_states[i_idx + 1:]:
                mean_i = model.means_[i][0]
                mean_j = model.means_[j][0]
                var_i = model.covars_[i][0, 0]
                var_j = model.covars_[j][0, 0]
                
                # Bhattacharyya distance
                distance = 0.25 * np.log(0.25 * ((var_i/var_j) + (var_j/var_i) + 2)) + \
                          0.25 * ((mean_i - mean_j)**2 / (var_i + var_j))
                
                label_i = label_map[i]
                label_j = label_map[j]
                distances[f'{label_i}-{label_j}'] = distance
        
        metrics['bhattacharyya_distances'] = distances
    
    # State purity (percentage of samples in each state)
    state_counts = {}
    for state_idx, state_label in label_map.items():
        count = np.sum(states == state_idx)
        percentage = (count / len(states)) * 100
        state_counts[state_label] = {'count': count, 'percentage': percentage}
    
    metrics['state_distribution'] = state_counts
    
    # Mean and std for each state
    state_stats = {}
    for state_idx, state_label in label_map.items():
        state_stats[state_label] = {
            'mean': model.means_[state_idx].tolist(),
            'covariance': model.covars_[state_idx].tolist()
        }
    
    metrics['state_statistics'] = state_stats
    
    return metrics

def create_metrics_table(metrics_single: dict, metrics_multi: dict) -> go.Figure:
    """Create a comparison table of metrics."""
    print("📋 Creating metrics comparison table...")
    
    # Prepare table data
    headers = ['Metric', 'Returns Only', 'Returns + Volatility']
    
    rows = []
    
    # State distribution
    rows.append(['<b>State Distribution</b>', '', ''])
    for state in ['BULL', 'BEAR', 'NEUTRAL']:
        single_pct = metrics_single['state_distribution'].get(state, {}).get('percentage', 0.0)
        multi_pct = metrics_multi['state_distribution'].get(state, {}).get('percentage', 0.0)
        rows.append([f'  {state}', f'{single_pct:.1f}%', f'{multi_pct:.1f}%'])
    
    # Bhattacharyya distances
    if 'bhattacharyya_distances' in metrics_single:
        rows.append(['<b>Separation (Bhattacharyya)</b>', '', ''])
        for pair, distance in metrics_single['bhattacharyya_distances'].items():
            rows.append([f'  {pair}', f'{distance:.4f}', 'N/A (2D)'])
    
    # Create table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=headers,
            fill_color='#1f77b4',
            align='left',
            font=dict(color='white', size=14, family='Arial Black')
        ),
        cells=dict(
            values=list(zip(*rows)),
            fill_color=[['#2d2d2d', '#1a1a1a'] * len(rows)],
            align='left',
            font=dict(color='white', size=12),
            height=30
        )
    )])
    
    fig.update_layout(
        title={
            'text': '📊 Model Comparison Metrics',
            'x': 0.5,
            'xanchor': 'center'
        },
        template='plotly_dark',
        height=400
    )
    
    return fig

def main():
    """Main execution function."""
    print("=" * 80)
    print("🔬 HMM GAUSSIAN DIAGNOSTICS")
    print("=" * 80)
    
    # Step 1: Fetch data
    df = fetch_candles(CANDLE_LIMIT)
    
    # Step 2: Engineer features
    df = engineer_features(df)
    
    # Step 3: Train models
    model_single, states_single, label_map_single = train_hmm_single_feature(df)
    model_multi, states_multi, label_map_multi = train_hmm_multi_feature(df)
    
    # Step 4: Calculate metrics
    metrics_single = calculate_separation_metrics(model_single, states_single, label_map_single, feature_dim=1)
    metrics_multi = calculate_separation_metrics(model_multi, states_multi, label_map_multi, feature_dim=2)
    
    # Step 5: Create visualizations
    fig1 = plot_single_feature_diagnostics(df, model_single, states_single, label_map_single)
    fig2 = plot_multi_feature_diagnostics(df, model_multi, states_multi, label_map_multi)
    fig3 = create_metrics_table(metrics_single, metrics_multi)
    
    # Step 6: Combine into dashboard
    print("🎨 Creating combined dashboard...")
    
    from plotly.subplots import make_subplots
    
    # Create subplot layout
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.35, 0.45, 0.20],
        subplot_titles=(
            '📊 Current Bot Logic: Log Returns Only',
            '🎯 Enhanced Logic: Returns + Volatility',
            '📋 Model Comparison'
        ),
        vertical_spacing=0.08,
        specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "table"}]]
    )
    
    # Add traces from fig1
    for trace in fig1.data:
        fig.add_trace(trace, row=1, col=1)
    
    # Add traces from fig2
    for trace in fig2.data:
        fig.add_trace(trace, row=2, col=1)
    
    # Add table from fig3
    for trace in fig3.data:
        fig.add_trace(trace, row=3, col=1)
    
    # Update layout
    fig.update_xaxes(title_text="Log Returns", row=1, col=1)
    fig.update_yaxes(title_text="Probability Density", row=1, col=1)
    
    fig.update_xaxes(title_text="Log Returns", row=2, col=1)
    fig.update_yaxes(title_text="Rolling Volatility (24-period)", row=2, col=1)
    
    fig.update_layout(
        title={
            'text': f'🔬 HMM Gaussian Diagnostics Dashboard<br><sub>{SYMBOL} | {INTERVAL}m | {len(df)} samples</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24}
        },
        template='plotly_dark',
        height=1400,
        showlegend=True,
        hovermode='closest'
    )
    
    # Step 7: Save output
    output_file = "hmm_diagnostics.html"
    fig.write_html(output_file)
    print(f"\n✅ Dashboard saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    
    print("\n🔹 Single Feature Model (Returns Only):")
    for state, stats in metrics_single['state_distribution'].items():
        print(f"   {state}: {stats['percentage']:.1f}% ({stats['count']} samples)")
    
    if 'bhattacharyya_distances' in metrics_single:
        print("\n   State Separation (Bhattacharyya Distance):")
        for pair, distance in metrics_single['bhattacharyya_distances'].items():
            print(f"   {pair}: {distance:.4f} (higher = better separation)")
    
    print("\n🔹 Multi Feature Model (Returns + Volatility):")
    for state, stats in metrics_multi['state_distribution'].items():
        print(f"   {state}: {stats['percentage']:.1f}% ({stats['count']} samples)")
    
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATION:")
    print("=" * 80)
    
    # Simple heuristic: check if states are well-separated
    if 'bhattacharyya_distances' in metrics_single:
        avg_distance = np.mean(list(metrics_single['bhattacharyya_distances'].values()))
        if avg_distance > 1.0:
            print("✅ Returns-only model shows GOOD state separation (distance > 1.0)")
            print("   → Current bot logic is sufficient!")
        elif avg_distance > 0.5:
            print("⚠️  Returns-only model shows MODERATE state separation (0.5 < distance < 1.0)")
            print("   → Consider adding volatility for better discrimination")
        else:
            print("❌ Returns-only model shows POOR state separation (distance < 0.5)")
            print("   → STRONGLY recommend adding volatility feature!")
        
        print(f"\n   Average Bhattacharyya Distance: {avg_distance:.4f}")
    
    print("\n" + "=" * 80)
    print(f"🎉 Analysis complete! Open {output_file} to explore the results.")
    print("=" * 80)

if __name__ == "__main__":
    main()
