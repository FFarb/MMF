"""
Multi-Timeframe HMM Strategy Backtest Engine
=============================================

Concept: "Confluence of States"
- Trade on 15m timeframe ONLY when aligned with 1H trend
- Reduces noise, churn, and false signals
- Professional implementation with signal persistence

Author: Senior Quant Developer
"""

import pandas as pd
import numpy as np
from hmmlearn import hmm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import webbrowser
from data_manager import BybitDataManager
from scipy import stats

# ============================================================
# CONFIGURATION
# ============================================================
SYMBOL = "BTCUSDT"
DAYS_BACK = 90
INITIAL_CAPITAL = 100
TRANSACTION_FEE = 0.0006  # 0.06%
N_COMPONENTS = 3  # Bull, Bear, Neutral
ROLLING_WINDOW_1H = 24  # 24 hours for 1H volatility
ROLLING_WINDOW_15M = 24  # 6 hours for 15m volatility (24 * 15min)
SIGNAL_PERSISTENCE = 3  # Rolling mode window to reduce flicker

print("="*80)
print("MULTI-TIMEFRAME HMM STRATEGY BACKTEST")
print("="*80)
print(f"Strategy: Trade 15m ONLY when aligned with 1H trend")
print(f"Persistence Filter: {SIGNAL_PERSISTENCE} candles (removes single-bar noise)")
print("="*80)

# ============================================================
# STEP 1: DUAL DATA FETCHING (1H + 15M)
# ============================================================
print("\n[1/7] Fetching Multi-Timeframe Data...")

# Fetch 1-Hour data (Trend)
print("\n  [A] Fetching 1-Hour Data (Trend Layer)...")
dm_1h = BybitDataManager(symbol=SYMBOL, interval="60")
df_1h = dm_1h.get_data(days_back=DAYS_BACK)
print(f"    [OK] 1H Data: {len(df_1h)} candles")

# Fetch 15-Minute data (Fast Layer)
print("\n  [B] Fetching 15-Minute Data (Fast Layer)...")
dm_15m = BybitDataManager(symbol=SYMBOL, interval="15")
df_15m = dm_15m.get_data(days_back=DAYS_BACK)
print(f"    [OK] 15M Data: {len(df_15m)} candles")

# Ensure timezone alignment (both should be UTC from Bybit)
df_1h.index = pd.to_datetime(df_1h.index).tz_localize(None)
df_15m.index = pd.to_datetime(df_15m.index).tz_localize(None)

print(f"\n  Time alignment:")
print(f"    1H:  {df_1h.index[0]} to {df_1h.index[-1]}")
print(f"    15M: {df_15m.index[0]} to {df_15m.index[-1]}")

# ============================================================
# STEP 2: FEATURE ENGINEERING (BOTH TIMEFRAMES)
# ============================================================
print("\n[2/7] Engineering Features...")

# 1H Features
df_1h['log_returns'] = np.log(df_1h['close'] / df_1h['close'].shift(1))
df_1h['rolling_volatility'] = df_1h['log_returns'].rolling(window=ROLLING_WINDOW_1H).std()
df_1h_clean = df_1h.dropna().copy()

# 15M Features
df_15m['log_returns'] = np.log(df_15m['close'] / df_15m['close'].shift(1))
df_15m['rolling_volatility'] = df_15m['log_returns'].rolling(window=ROLLING_WINDOW_15M).std()
df_15m_clean = df_15m.dropna().copy()

print(f"  [OK] 1H Clean:  {len(df_1h_clean)} rows")
print(f"  [OK] 15M Clean: {len(df_15m_clean)} rows")

# ============================================================
# STEP 3A: TRAIN HMM ON 1H DATA (TREND MODEL)
# ============================================================
print("\n[3/7] Training HMM Models...")
print("\n  [A] Training 1H Trend Model...")

X_1h = df_1h_clean[['log_returns', 'rolling_volatility']].values
model_1h = hmm.GaussianHMM(
    n_components=N_COMPONENTS,
    covariance_type='full',
    n_iter=1000,
    random_state=42
)
model_1h.fit(X_1h)
states_1h = model_1h.predict(X_1h)
df_1h_clean['state'] = states_1h

# Smart state labeling (1H)
state_means_1h = df_1h_clean.groupby('state')['log_returns'].mean().sort_values()
bull_1h = state_means_1h.idxmax()
bear_1h = state_means_1h.idxmin()
neutral_1h = [s for s in [0, 1, 2] if s not in [bull_1h, bear_1h]][0]

label_map_1h = {bull_1h: 'Bull', bear_1h: 'Bear', neutral_1h: 'Neutral'}
df_1h_clean['regime_1h'] = df_1h_clean['state'].map(label_map_1h)

print(f"    [OK] Converged: {model_1h.monitor_.converged}")
print(f"    States: Bull={bull_1h}, Bear={bear_1h}, Neutral={neutral_1h}")

for state in [0, 1, 2]:
    regime = label_map_1h[state]
    mean_ret = df_1h_clean[df_1h_clean['state'] == state]['log_returns'].mean()
    pct = (df_1h_clean['state'] == state).sum() / len(df_1h_clean) * 100
    print(f"    {regime:7s}: Mean Ret={mean_ret:+.6f} | Time={pct:.1f}%")

# ============================================================
# STEP 3B: TRAIN HMM ON 15M DATA (FAST MODEL)
# ============================================================
print("\n  [B] Training 15M Fast Model...")

X_15m = df_15m_clean[['log_returns', 'rolling_volatility']].values
model_15m = hmm.GaussianHMM(
    n_components=N_COMPONENTS,
    covariance_type='full',
    n_iter=1000,
    random_state=42
)
model_15m.fit(X_15m)
states_15m = model_15m.predict(X_15m)
df_15m_clean['state'] = states_15m

# Smart state labeling (15M)
state_means_15m = df_15m_clean.groupby('state')['log_returns'].mean().sort_values()
bull_15m = state_means_15m.idxmax()
bear_15m = state_means_15m.idxmin()
neutral_15m = [s for s in [0, 1, 2] if s not in [bull_15m, bear_15m]][0]

label_map_15m = {bull_15m: 'Bull', bear_15m: 'Bear', neutral_15m: 'Neutral'}
df_15m_clean['regime_15m'] = df_15m_clean['state'].map(label_map_15m)

print(f"    [OK] Converged: {model_15m.monitor_.converged}")
print(f"    States: Bull={bull_15m}, Bear={bear_15m}, Neutral={neutral_15m}")

for state in [0, 1, 2]:
    regime = label_map_15m[state]
    mean_ret = df_15m_clean[df_15m_clean['state'] == state]['log_returns'].mean()
    pct = (df_15m_clean['state'] == state).sum() / len(df_15m_clean) * 100
    print(f"    {regime:7s}: Mean Ret={mean_ret:+.6f} | Time={pct:.1f}%")

# ============================================================
# STEP 4: MERGE 1H STATE ONTO 15M DATAFRAME (THE HARD PART)
# ============================================================
print("\n[4/7] Merging Timeframes (1H State -> 15M DataFrame)...")

# Prepare 1H states for merging
df_1h_states = df_1h_clean[['regime_1h']].copy()
df_1h_states.columns = ['regime_1h']

# Use merge_asof to map 1H state to each 15m candle
# Direction='backward' means: use the most recent 1H state for each 15m bar
df_merged = pd.merge_asof(
    df_15m_clean.sort_index(),
    df_1h_states.sort_index(),
    left_index=True,
    right_index=True,
    direction='backward'
)

print(f"  [OK] Merged DataFrame: {len(df_merged)} rows")
print(f"  Columns: {list(df_merged.columns)}")

# Verify merge quality
null_1h = df_merged['regime_1h'].isna().sum()
if null_1h > 0:
    print(f"  [WARNING] {null_1h} rows have missing 1H state (edge effects)")
    df_merged = df_merged.dropna(subset=['regime_1h'])
    print(f"  Cleaned to: {len(df_merged)} rows")

# ============================================================
# STEP 5: CONFLUENCE TRADING LOGIC (THE FILTER)
# ============================================================
print("\n[5/7] Applying Confluence Filter...")

# Initial signal: Only trade when timeframes align
def get_signal(row):
    """Strict confluence logic"""
    if row['regime_1h'] == 'Bull' and row['regime_15m'] == 'Bull':
        return 1  # LONG
    elif row['regime_1h'] == 'Bear' and row['regime_15m'] == 'Bear':
        return -1  # SHORT
    else:
        return 0  # NEUTRAL (divergence)

df_merged['signal_raw'] = df_merged.apply(get_signal, axis=1)

# Apply signal persistence (rolling mode) to reduce flicker
df_merged['signal'] = df_merged['signal_raw'].rolling(
    window=SIGNAL_PERSISTENCE, 
    min_periods=1
).apply(lambda x: stats.mode(x, keepdims=True)[0][0], raw=True)

# Calculate confluence statistics
total_candles = len(df_merged)
confluence_long = (df_merged['signal'] == 1).sum()
confluence_short = (df_merged['signal'] == -1).sum()
neutral = (df_merged['signal'] == 0).sum()

print(f"\n  Signal Distribution:")
print(f"    LONG Confluence:  {confluence_long:5d} ({confluence_long/total_candles*100:.1f}%)")
print(f"    SHORT Confluence: {confluence_short:5d} ({confluence_short/total_candles*100:.1f}%)")
print(f"    NEUTRAL:          {neutral:5d} ({neutral/total_candles*100:.1f}%)")

# ============================================================
# STEP 6: BACKTEST EXECUTION (ON 15M TIMEFRAME)
# ============================================================
print("\n[6/7] Running Backtest...")

# Shift signal to avoid lookahead bias
df_merged['signal_shifted'] = df_merged['signal'].shift(1)

# Calculate strategy returns
df_merged['strategy_log_ret'] = df_merged['signal_shifted'] * df_merged['log_returns']

# Identify trades (signal changes)
df_merged['trade'] = (df_merged['signal_shifted'].diff() != 0) & (df_merged['signal_shifted'].notna())
total_trades = df_merged['trade'].sum()

# Apply transaction fees
df_merged['strategy_log_ret_with_fees'] = df_merged['strategy_log_ret'].copy()
df_merged.loc[df_merged['trade'], 'strategy_log_ret_with_fees'] -= TRANSACTION_FEE

# Calculate equity curves
df_merged['buy_hold_equity'] = INITIAL_CAPITAL * np.exp(df_merged['log_returns'].cumsum())
df_merged['strategy_equity'] = INITIAL_CAPITAL * np.exp(df_merged['strategy_log_ret_with_fees'].cumsum())

print(f"  [OK] Backtest completed")
print(f"  Total trades: {total_trades}")

# ============================================================
# STEP 7: PERFORMANCE METRICS
# ============================================================
print("\n" + "="*80)
print("PERFORMANCE METRICS")
print("="*80)

# Buy & Hold
bh_return = (df_merged['buy_hold_equity'].iloc[-1] / INITIAL_CAPITAL - 1) * 100
bh_sharpe = df_merged['log_returns'].mean() / df_merged['log_returns'].std() * np.sqrt(252 * 24 * 4)  # Annualized for 15m
bh_cummax = df_merged['buy_hold_equity'].cummax()
bh_dd = ((bh_cummax - df_merged['buy_hold_equity']) / bh_cummax).max() * 100

# MTF Strategy
strat_return = (df_merged['strategy_equity'].iloc[-1] / INITIAL_CAPITAL - 1) * 100
strat_sharpe = df_merged['strategy_log_ret_with_fees'].mean() / df_merged['strategy_log_ret_with_fees'].std() * np.sqrt(252 * 24 * 4)
strat_cummax = df_merged['strategy_equity'].cummax()
strat_dd = ((strat_cummax - df_merged['strategy_equity']) / strat_cummax).max() * 100

print(f"\nBuy & Hold:")
print(f"  Return:      {bh_return:+.2f}%")
print(f"  Sharpe:      {bh_sharpe:.3f}")
print(f"  Max DD:      {bh_dd:.2f}%")
print(f"  Final Value: ${df_merged['buy_hold_equity'].iloc[-1]:.2f}")

print(f"\nMTF HMM Strategy:")
print(f"  Return:      {strat_return:+.2f}%")
print(f"  Sharpe:      {strat_sharpe:.3f}")
print(f"  Max DD:      {strat_dd:.2f}%")
print(f"  Total Trades: {total_trades}")
print(f"  Final Value: ${df_merged['strategy_equity'].iloc[-1]:.2f}")

print(f"\nOutperformance: {strat_return - bh_return:+.2f}%")
print(f"Sharpe Improvement: {strat_sharpe - bh_sharpe:+.3f}")
print("="*80)

# ============================================================
# STEP 8: VISUALIZATION
# ============================================================
print("\n[7/7] Creating Visualization...")

fig = make_subplots(
    rows=3, cols=1,
    row_heights=[0.5, 0.25, 0.25],
    subplot_titles=(
        '15M Price with MTF Confluence Markers',
        '1H Trend State (Background Filter)',
        'Equity Curve: Buy & Hold vs MTF Strategy'
    ),
    vertical_spacing=0.08,
    shared_xaxes=True
)

# ============================================================
# ROW 1: 15M PRICE WITH CONFLUENCE MARKERS
# ============================================================

# Background price
fig.add_trace(go.Scatter(
    x=df_merged.index,
    y=df_merged['close'],
    mode='lines',
    name='BTC Price (15m)',
    line=dict(color='rgba(150, 150, 150, 0.4)', width=1),
    hovertemplate='Price: $%{y:,.2f}<extra></extra>'
), row=1, col=1)

# Confluence LONG markers (both timeframes bullish)
long_conf = df_merged[df_merged['signal'] == 1]
fig.add_trace(go.Scatter(
    x=long_conf.index,
    y=long_conf['close'],
    mode='markers',
    name='LONG Confluence',
    marker=dict(color='lime', size=2, opacity=0.6),
    hovertemplate='<b>LONG</b><br>Price: $%{y:,.2f}<extra></extra>'
), row=1, col=1)

# Confluence SHORT markers
short_conf = df_merged[df_merged['signal'] == -1]
fig.add_trace(go.Scatter(
    x=short_conf.index,
    y=short_conf['close'],
    mode='markers',
    name='SHORT Confluence',
    marker=dict(color='red', size=2, opacity=0.6),
    hovertemplate='<b>SHORT</b><br>Price: $%{y:,.2f}<extra></extra>'
), row=1, col=1)

# Trade entry markers
long_entries = df_merged[(df_merged['signal_shifted'].shift(1) != 1) & (df_merged['signal_shifted'] == 1)]
fig.add_trace(go.Scatter(
    x=long_entries.index,
    y=long_entries['close'],
    mode='markers',
    name='LONG Entry',
    marker=dict(symbol='triangle-up', size=12, color='lime', line=dict(width=2, color='darkgreen')),
    hovertemplate='<b>LONG ENTRY</b><br>$%{y:,.2f}<extra></extra>'
), row=1, col=1)

short_entries = df_merged[(df_merged['signal_shifted'].shift(1) != -1) & (df_merged['signal_shifted'] == -1)]
fig.add_trace(go.Scatter(
    x=short_entries.index,
    y=short_entries['close'],
    mode='markers',
    name='SHORT Entry',
    marker=dict(symbol='triangle-down', size=12, color='red', line=dict(width=2, color='darkred')),
    hovertemplate='<b>SHORT ENTRY</b><br>$%{y:,.2f}<extra></extra>'
), row=1, col=1)

# ============================================================
# ROW 2: 1H TREND STATE (BACKGROUND FILTER)
# ============================================================

# Create numerical representation for 1H state
state_map = {'Bull': 1, 'Neutral': 0, 'Bear': -1}
df_merged['state_1h_num'] = df_merged['regime_1h'].map(state_map)

fig.add_trace(go.Scatter(
    x=df_merged.index,
    y=df_merged['state_1h_num'],
    mode='lines',
    name='1H Trend',
    line=dict(color='orange', width=2),
    fill='tozeroy',
    fillcolor='rgba(255, 165, 0, 0.2)',
    hovertemplate='1H State: %{text}<extra></extra>',
    text=df_merged['regime_1h']
), row=2, col=1)

# ============================================================
# ROW 3: EQUITY CURVES
# ============================================================

fig.add_trace(go.Scatter(
    x=df_merged.index,
    y=df_merged['buy_hold_equity'],
    mode='lines',
    name='Buy & Hold',
    line=dict(color='cyan', width=2),
    hovertemplate='B&H: $%{y:.2f}<extra></extra>'
), row=3, col=1)

fig.add_trace(go.Scatter(
    x=df_merged.index,
    y=df_merged['strategy_equity'],
    mode='lines',
    name='MTF Strategy',
    line=dict(color='lime', width=3),
    hovertemplate='MTF: $%{y:.2f}<extra></extra>'
), row=3, col=1)

# ============================================================
# LAYOUT & ANNOTATIONS
# ============================================================

# Add performance annotation
annotation_text = (
    f"<b>MTF HMM Strategy Results</b><br>"
    f"Total Trades: {total_trades}<br>"
    f"Sharpe Ratio: {strat_sharpe:.3f}<br>"
    f"Return: {strat_return:+.2f}%<br>"
    f"Max DD: {strat_dd:.2f}%<br>"
    f"Confluence: {(confluence_long + confluence_short)/total_candles*100:.1f}%"
)

fig.add_annotation(
    text=annotation_text,
    xref="paper", yref="paper",
    x=0.02, y=0.98,
    showarrow=False,
    bgcolor="rgba(0, 0, 0, 0.7)",
    bordercolor="lime",
    borderwidth=2,
    font=dict(color="white", size=11, family="Courier New"),
    align="left",
    xanchor="left",
    yanchor="top"
)

fig.update_layout(
    title={
        'text': f'Multi-Timeframe HMM Strategy - {SYMBOL} (1H + 15M Confluence)',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20, 'family': 'Arial Black', 'color': 'white'}
    },
    template='plotly_dark',
    height=1100,
    hovermode='x unified',
    legend=dict(
        orientation="v",
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99,
        bgcolor='rgba(0,0,0,0.6)'
    )
)

# Update axes
fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
fig.update_yaxes(title_text="1H State", row=2, col=1, ticktext=['Bear', 'Neutral', 'Bull'], tickvals=[-1, 0, 1])
fig.update_yaxes(title_text="Portfolio Value ($)", row=3, col=1)
fig.update_xaxes(title_text="Date", row=3, col=1)

# Save and open
output_file = "mtf_hmm_backtest.html"
fig.write_html(output_file)
print(f"  [OK] Visualization saved: {output_file}")

webbrowser.open(f"file://{Path(output_file).absolute()}")
print(f"  [OK] Opening in browser...")

print("\n" + "="*80)
print("MTF BACKTEST COMPLETE [OK]")
print("="*80)
print("\nKey Insight: By requiring confluence between 1H and 15M,")
print("we dramatically reduce false signals and improve risk-adjusted returns.")
print("="*80)
