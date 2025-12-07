"""
Complete HMM Backtesting Engine for Bybit BTCUSDT Futures

Features:
- Fresh data fetching from Bybit API (90 days)
- Smart HMM state identification (automatic LONG/SHORT/NEUTRAL labeling)
- Realistic backtesting with transaction fees (0.06%)
- Professional visualization with trade markers and equity curves
"""

import pandas as pd
import numpy as np
from hmmlearn import hmm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import webbrowser
from data_manager import BybitDataManager

# Configuration
SYMBOL = "BTCUSDT"
INTERVAL = "60"  # 1 hour
DAYS_BACK = 90
INITIAL_CAPITAL = 100  # Starting capital in USD
TRANSACTION_FEE = 0.0006  # 0.06% per trade
N_COMPONENTS = 3  # Bull, Bear, Neutral states
ROLLING_WINDOW = 24  # 24 hours for volatility

print("="*70)
print("HMM BACKTESTING ENGINE - BYBIT BTCUSDT FUTURES")
print("="*70)

# ============================================================
# 1. DATA FETCHING - OPTIMIZED WITH CACHING
# ============================================================
print("\n[1/5] Loading data (using smart cache)...")

# Use optimized data manager
data_manager = BybitDataManager(symbol=SYMBOL, interval=INTERVAL)
df = data_manager.get_data(days_back=DAYS_BACK)

print(f"  ✓ Loaded {len(df)} candles")
print(f"  Date range: {df.index[0]} to {df.index[-1]}")
print(f"  Close price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
print("\n[2/5] Calculating features...")

# Calculate Log Returns
df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

# Calculate Rolling Volatility (24-hour window)
df['rolling_volatility'] = df['log_returns'].rolling(window=ROLLING_WINDOW).std()

# Drop NaN values
df_clean = df.dropna().copy()
print(f"  ✓ Calculated log returns and {ROLLING_WINDOW}h rolling volatility")
print(f"  Clean data: {len(df_clean)} rows (after removing NaN)")

# ============================================================
# 3. HMM TRAINING & SMART STATE IDENTIFICATION
# ============================================================
print("\n[3/5] Training HMM and identifying states...")

# Prepare feature matrix [Log Returns, Rolling Volatility]
X = df_clean[['log_returns', 'rolling_volatility']].values

# Create and train the model
model = hmm.GaussianHMM(
    n_components=N_COMPONENTS,
    covariance_type='full',
    n_iter=1000,
    random_state=42
)

model.fit(X)
print(f"  ✓ HMM training completed (converged: {model.monitor_.converged})")

# Predict hidden states
hidden_states = model.predict(X)
df_clean['state'] = hidden_states

# SMART STATE LABELING: Assign signals based on mean returns
state_means = df_clean.groupby('state')['log_returns'].mean().sort_values()

# Verify we have all 3 states
unique_states = df_clean['state'].unique()
print(f"  Unique states found: {sorted(unique_states)}")

if len(unique_states) != N_COMPONENTS:
    print(f"  [WARNING] Expected {N_COMPONENTS} states, found {len(unique_states)}")

# State with highest mean return = LONG
# State with lowest mean return = SHORT
# Middle state = NEUTRAL
long_state = state_means.idxmax()
short_state = state_means.idxmin()

# Find neutral state (the remaining one)
all_possible_states = set(range(N_COMPONENTS))
assigned_states = {long_state, short_state}
remaining_states = list(all_possible_states - assigned_states)
neutral_state = remaining_states[0] if remaining_states else long_state  # Fallback

# Create signal mapping
signal_map = {
    long_state: 1,      # LONG
    short_state: -1,    # SHORT
    neutral_state: 0    # NEUTRAL
}

df_clean['signal'] = df_clean['state'].map(signal_map)

# Handle any unmapped states (shouldn't happen, but safety first)
if df_clean['signal'].isna().any():
    print(f"  [ERROR] Some states were not mapped to signals!")
    df_clean['signal'].fillna(0, inplace=True)

# Create readable labels for visualization
label_map = {
    long_state: 'LONG',
    short_state: 'SHORT',
    neutral_state: 'NEUTRAL'
}
df_clean['regime'] = df_clean['state'].map(label_map)

print(f"\n  State Identification:")
for state in [0, 1, 2]:
    mean_ret = df_clean[df_clean['state'] == state]['log_returns'].mean()
    mean_vol = df_clean[df_clean['state'] == state]['rolling_volatility'].mean()
    regime = label_map[state]
    signal = signal_map[state]
    pct_time = (df_clean['state'] == state).sum() / len(df_clean) * 100
    
    print(f"  State {state} = {regime:7s} (Signal: {signal:2d}) | "
          f"Mean Return: {mean_ret:+.6f} | Mean Vol: {mean_vol:.6f} | "
          f"Time: {pct_time:.1f}%")

# ============================================================
# 4. VECTORIZED BACKTESTING WITH FEES
# ============================================================
print("\n[4/5] Running backtest with transaction fees...")

# Shift signal to avoid lookahead bias (trade at next bar based on current state)
df_clean['signal_shifted'] = df_clean['signal'].shift(1)

# Calculate strategy returns (signal * asset returns)
df_clean['strategy_log_ret'] = df_clean['signal_shifted'] * df_clean['log_returns']

# Identify trades (signal changes)
df_clean['trade'] = (df_clean['signal_shifted'].diff() != 0) & (df_clean['signal_shifted'].notna())
total_trades = df_clean['trade'].sum()

# Apply transaction fees on trades
df_clean['strategy_log_ret_with_fees'] = df_clean['strategy_log_ret'].copy()
df_clean.loc[df_clean['trade'], 'strategy_log_ret_with_fees'] -= TRANSACTION_FEE

# Calculate cumulative returns (equity curves)
df_clean['buy_hold_equity'] = INITIAL_CAPITAL * np.exp(df_clean['log_returns'].cumsum())
df_clean['strategy_equity'] = INITIAL_CAPITAL * np.exp(df_clean['strategy_log_ret_with_fees'].cumsum())

print(f"  ✓ Backtest completed")
print(f"  Total trades executed: {total_trades}")

# ============================================================
# 5. PERFORMANCE METRICS
# ============================================================
print("\n" + "="*70)
print("PERFORMANCE METRICS")
print("="*70)

# Buy & Hold metrics
bh_total_return = (df_clean['buy_hold_equity'].iloc[-1] / INITIAL_CAPITAL - 1) * 100
bh_sharpe = df_clean['log_returns'].mean() / df_clean['log_returns'].std() * np.sqrt(252 * 24)
bh_cummax = df_clean['buy_hold_equity'].cummax()
bh_drawdown = ((bh_cummax - df_clean['buy_hold_equity']) / bh_cummax).max() * 100

# Strategy metrics
strategy_total_return = (df_clean['strategy_equity'].iloc[-1] / INITIAL_CAPITAL - 1) * 100
strategy_sharpe = df_clean['strategy_log_ret_with_fees'].mean() / df_clean['strategy_log_ret_with_fees'].std() * np.sqrt(252 * 24)
strategy_cummax = df_clean['strategy_equity'].cummax()
strategy_drawdown = ((strategy_cummax - df_clean['strategy_equity']) / strategy_cummax).max() * 100

print(f"\nBuy & Hold Strategy:")
print(f"  Total Return:     {bh_total_return:+.2f}%")
print(f"  Sharpe Ratio:     {bh_sharpe:.3f}")
print(f"  Max Drawdown:     {bh_drawdown:.2f}%")
print(f"  Final Value:      ${df_clean['buy_hold_equity'].iloc[-1]:.2f}")

print(f"\nHMM Strategy (Net of {TRANSACTION_FEE*100}% fees):")
print(f"  Total Return:     {strategy_total_return:+.2f}%")
print(f"  Sharpe Ratio:     {strategy_sharpe:.3f}")
print(f"  Max Drawdown:     {strategy_drawdown:.2f}%")
print(f"  Total Trades:     {total_trades}")
print(f"  Final Value:      ${df_clean['strategy_equity'].iloc[-1]:.2f}")

print(f"\nOutperformance:     {strategy_total_return - bh_total_return:+.2f}%")
print("="*70)

# ============================================================
# 6. VISUALIZATION
# ============================================================
print("\n[5/5] Creating visualization...")

# Create subplot figure (2 rows)
fig = make_subplots(
    rows=2, cols=1,
    row_heights=[0.6, 0.4],
    subplot_titles=(
        'BTC Price with HMM States & Trade Entries',
        'Portfolio Growth: Buy & Hold vs HMM Strategy'
    ),
    vertical_spacing=0.1,
    shared_xaxes=True
)

# ============================================================
# ROW 1: PRICE CHART WITH STATES & TRADE MARKERS
# ============================================================

# Background: BTC close price (grey line)
fig.add_trace(go.Scatter(
    x=df_clean.index,
    y=df_clean['close'],
    mode='lines',
    name='BTC Price',
    line=dict(color='rgba(150, 150, 150, 0.5)', width=1),
    hovertemplate='Price: $%{y:,.2f}<br>%{x}<extra></extra>'
), row=1, col=1)

# State overlay: Colored markers (not lines!)
state_colors = {
    'LONG': 'rgb(0, 255, 0)',      # Green
    'SHORT': 'rgb(255, 0, 0)',      # Red
    'NEUTRAL': 'rgb(100, 149, 237)' # Blue
}

for regime, color in state_colors.items():
    regime_data = df_clean[df_clean['regime'] == regime]
    fig.add_trace(go.Scatter(
        x=regime_data.index,
        y=regime_data['close'],
        mode='markers',
        name=f'{regime} State',
        marker=dict(color=color, size=3),
        hovertemplate=f'<b>{regime}</b><br>Price: $%{{y:,.2f}}<br>%{{x}}<extra></extra>'
    ), row=1, col=1)

# Trade entry markers
# Long entries: signal changed to +1
long_entries = df_clean[(df_clean['signal_shifted'].shift(1) != 1) & (df_clean['signal_shifted'] == 1)]
fig.add_trace(go.Scatter(
    x=long_entries.index,
    y=long_entries['close'],
    mode='markers',
    name='LONG Entry',
    marker=dict(symbol='triangle-up', size=10, color='lime', line=dict(width=1, color='darkgreen')),
    hovertemplate='<b>LONG ENTRY</b><br>Price: $%{y:,.2f}<br>%{x}<extra></extra>'
), row=1, col=1)

# Short entries: signal changed to -1
short_entries = df_clean[(df_clean['signal_shifted'].shift(1) != -1) & (df_clean['signal_shifted'] == -1)]
fig.add_trace(go.Scatter(
    x=short_entries.index,
    y=short_entries['close'],
    mode='markers',
    name='SHORT Entry',
    marker=dict(symbol='triangle-down', size=10, color='red', line=dict(width=1, color='darkred')),
    hovertemplate='<b>SHORT ENTRY</b><br>Price: $%{y:,.2f}<br>%{x}<extra></extra>'
), row=1, col=1)

# ============================================================
# ROW 2: EQUITY CURVES
# ============================================================

# Buy & Hold equity
fig.add_trace(go.Scatter(
    x=df_clean.index,
    y=df_clean['buy_hold_equity'],
    mode='lines',
    name='Buy & Hold',
    line=dict(color='cyan', width=2),
    hovertemplate='Buy & Hold: $%{y:.2f}<br>%{x}<extra></extra>'
), row=2, col=1)

# Strategy equity
fig.add_trace(go.Scatter(
    x=df_clean.index,
    y=df_clean['strategy_equity'],
    mode='lines',
    name='HMM Strategy (Net)',
    line=dict(color='lime', width=2),
    hovertemplate='HMM Strategy: $%{y:.2f}<br>%{x}<extra></extra>'
), row=2, col=1)

# ============================================================
# LAYOUT CONFIGURATION
# ============================================================

fig.update_layout(
    title={
        'text': f'HMM Backtesting Engine - {SYMBOL} ({DAYS_BACK} Days)',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 22, 'family': 'Arial Black', 'color': 'white'}
    },
    template='plotly_dark',
    height=900,
    hovermode='x unified',
    legend=dict(
        orientation="v",
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor='rgba(0,0,0,0.5)'
    ),
    xaxis2=dict(
        rangeslider=dict(visible=True),
        type='date'
    )
)

# Update axes labels
fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
fig.update_yaxes(title_text="Portfolio Value ($)", row=2, col=1)
fig.update_xaxes(title_text="Date", row=2, col=1)

# Save the plot
output_file = "hmm_backtest_complete.html"
fig.write_html(output_file)
print(f"  ✓ Visualization saved to: {output_file}")

# Open in browser
webbrowser.open(f"file://{Path(output_file).absolute()}")
print(f"  ✓ Opening in browser...")

print("\n" + "="*70)
print("BACKTEST COMPLETE ✓")
print("="*70)
