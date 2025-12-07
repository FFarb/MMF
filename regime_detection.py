"""
Market Regime Detection using Gaussian Hidden Markov Model
Analyzes BTC futures data to identify distinct market states
"""

import pandas as pd
import numpy as np
from hmmlearn import hmm
import plotly.graph_objects as go
from pathlib import Path

# ============================================================
# 1. Load Data
# ============================================================
print("Loading data from btc_futures.parquet...")
df = pd.read_parquet("btc_futures.parquet")
print(f"Loaded {len(df)} rows")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
print(f"Columns: {df.columns.tolist()}\n")

# ============================================================
# 2. Feature Engineering
# ============================================================
print("Calculating features...")

# Calculate Log Returns: ln(Close / Close_shifted)
df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

# Calculate Rolling Volatility: Standard deviation of Log Returns over a 24-hour window
df['rolling_volatility'] = df['log_returns'].rolling(window=24).std()

# Drop NaN values created by rolling window and log returns
df_clean = df.dropna().copy()
print(f"After dropping NaN: {len(df_clean)} rows\n")

# ============================================================
# 3. HMM Training
# ============================================================
print("Training Gaussian HMM...")

# Prepare feature matrix [Log Returns, Rolling Volatility]
X = df_clean[['log_returns', 'rolling_volatility']].values

# Create and train the model
model = hmm.GaussianHMM(
    n_components=3,
    covariance_type='full',
    n_iter=1000,
    random_state=42
)

model.fit(X)
print("Model training completed!")

# Predict hidden states
hidden_states = model.predict(X)
df_clean['state'] = hidden_states

print(f"Model converged: {model.monitor_.converged}")
print(f"Number of iterations: {model.monitor_.iter}\n")

# ============================================================
# 4. Analysis
# ============================================================
print("=" * 60)
print("MARKET REGIME ANALYSIS")
print("=" * 60)

# Calculate statistics for each state
state_stats = df_clean.groupby('state').agg({
    'log_returns': ['mean', 'std', 'count'],
    'rolling_volatility': 'mean'
}).round(6)

state_stats.columns = ['Mean_Return', 'Std_Return', 'Count', 'Mean_Volatility']

print("\nState Statistics:")
print(state_stats)
print()

# Identify regime characteristics
for state in range(3):
    state_data = df_clean[df_clean['state'] == state]
    mean_ret = state_data['log_returns'].mean()
    mean_vol = state_data['rolling_volatility'].mean()
    pct_time = (len(state_data) / len(df_clean)) * 100
    
    # Classify regime
    if mean_ret > 0 and mean_vol < state_stats['Mean_Volatility'].median():
        regime_type = "🟢 BULLISH (Low Vol Uptrend)"
    elif mean_ret < 0 and mean_vol < state_stats['Mean_Volatility'].median():
        regime_type = "🔵 BEARISH (Low Vol Downtrend)"
    elif mean_vol > state_stats['Mean_Volatility'].median():
        regime_type = "🔴 HIGH VOLATILITY (Unstable)"
    else:
        regime_type = "⚪ NEUTRAL (Noise/Sideways)"
    
    print(f"State {state}: {regime_type}")
    print(f"  - Mean Return: {mean_ret:.6f}")
    print(f"  - Mean Volatility: {mean_vol:.6f}")
    print(f"  - Time in state: {pct_time:.1f}%")
    print()

print("=" * 60)
print()

# ============================================================
# 5. Visualization
# ============================================================
print("Creating visualization...")

# Define colors for each state
state_colors = {
    0: 'blue',
    1: 'green',
    2: 'red'
}

# Create figure
fig = go.Figure()

# Plot close price colored by state
for state in range(3):
    state_mask = df_clean['state'] == state
    state_df = df_clean[state_mask]
    
    fig.add_trace(go.Scatter(
        x=state_df.index,
        y=state_df['close'],
        mode='lines',
        name=f'State {state}',
        line=dict(color=state_colors[state], width=1.5),
        hovertemplate='<b>State %{text}</b><br>' +
                      'Time: %{x}<br>' +
                      'Price: $%{y:,.2f}<br>' +
                      '<extra></extra>',
        text=[state] * len(state_df)
    ))

# Update layout
fig.update_layout(
    title={
        'text': 'BTC Futures: Market Regime Detection (HMM)',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20, 'family': 'Arial Black'}
    },
    xaxis_title='Date',
    yaxis_title='Close Price (USD)',
    hovermode='x unified',
    template='plotly_dark',
    height=700,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    xaxis=dict(
        rangeslider=dict(visible=True),
        type='date'
    )
)

# Save the plot
output_file = "btc_regimes.html"
fig.write_html(output_file)
print(f"✓ Visualization saved to: {output_file}")

# Open the file in browser
import webbrowser
webbrowser.open(f"file://{Path(output_file).absolute()}")
print(f"✓ Opening visualization in browser...")
