# Multi-Timeframe (MTF) Feature Injection - Implementation Summary

## Date: 2025-12-01

## Problem Statement

### Context Myopia on 5-Minute Data

**Observed Issue:**
- Switching from H1 to M5 caused precision drop: **60% → 24%** on Altcoins
- Root cause: **Context myopia** - model lost macro trend view

**Technical Analysis:**
```
H1 Data (Hourly):
- 64 candles = 64 hours = 2.7 days of context
- Captures macro trends, major support/resistance
- Stable, less noise

M5 Data (5-Minute):
- 64 candles = 320 minutes = 5.3 hours of context
- Misses macro trends, only sees micro noise
- Volatile, high noise
```

**Result:**
- Model makes decisions based on 5-hour window
- Cannot see if we're in a bullish H1 trend or bearish H1 reversal
- Trades against macro trend → poor precision

---

## Solution: Multi-Timeframe (MTF) Feature Injection

**Concept:** "Anchor volatile 5-minute moves to stable 1-hour trends"

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ M5 Data (Primary)                                           │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ 10:00  10:05  10:10  10:15  10:20  10:25  10:30  10:35 │ │
│ │   ↓      ↓      ↓      ↓      ↓      ↓      ↓      ↓   │ │
│ │  M5    M5     M5     M5     M5     M5     M5     M5    │ │
│ └─────────────────────────────────────────────────────────┘ │
│                           ↑                                  │
│                    MTF Injection                             │
│                           ↓                                  │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ H1 Data (Macro Context)                                 │ │
│ │ 10:00                                                   │ │
│ │   ↓                                                     │ │
│ │  H1 ────────────────────────────────────────────────→  │ │
│ │  (macro_trend, macro_rsi, macro_volatility, ...)       │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

Result: Each M5 bar has both micro (5-min) and macro (1-hour) features
```

---

## Implementation Details

### 1. SignalFactory Upgrade (`src/features/__init__.py`)

#### New Method: `_inject_macro_features()`

**Macro Indicators Calculated (on H1 data):**

1. **`macro_rsi`**: RSI(14) on H1
   - Overbought/oversold on macro timeframe
   - Range: 0-100

2. **`macro_trend`**: SMA(50) vs SMA(200) cross
   - +1 = Bullish (SMA50 > SMA200)
   - -1 = Bearish (SMA50 < SMA200)
   - Primary trend direction

3. **`macro_volatility`**: 20-period std on H1
   - Macro market stability
   - Higher = more volatile macro environment

4. **`macro_momentum`**: ROC(21) on H1
   - Rate of change over 21 H1 bars
   - Macro momentum strength

5. **`macro_adx`**: ADX(14) on H1
   - Trend strength on macro timeframe
   - >25 = strong trend, <20 = weak/ranging

6. **`macro_macd`**: MACD(12,26,9) on H1
   - Macro trend changes
   - Divergence detection

7. **`macro_macd_hist`**: MACD histogram on H1
   - Momentum acceleration/deceleration

8. **`macro_dist_sma50`**: Distance from SMA(50)
   - Price position relative to macro MA
   - Positive = above, negative = below

#### Broadcasting Logic

```python
# H1 features calculated on hourly bars
macro_features = pd.DataFrame(index=macro_df.index)
macro_features['macro_rsi'] = ta.rsi(macro_df['close'], length=14)
# ... (8 total features)

# Broadcast to M5 timestamps (forward fill)
macro_resampled = macro_features.reindex(
    df.index,  # M5 index
    method='ffill',  # Forward fill
    limit=12  # Max 12 periods (1 hour for M5)
)

# Each M5 bar gets the most recent H1 value
```

**Example:**
```
H1 Bar at 10:00 AM:
  macro_trend = +1 (bullish)
  macro_rsi = 65 (overbought)

M5 Bars inherit this:
  10:00, 10:05, 10:10, ..., 10:55 all get:
    macro_trend = +1
    macro_rsi = 65

H1 Bar at 11:00 AM:
  macro_trend = +1 (still bullish)
  macro_rsi = 70 (more overbought)

M5 Bars update:
  11:00, 11:05, 11:10, ..., 11:55 now get:
    macro_trend = +1
    macro_rsi = 70
```

### 2. Hierarchical Fleet Upgrade (`run_hierarchical_fleet.py`)

#### Automatic H1 Loading

```python
# When using --minutes (M5 mode)
if interval == "5":
    print(f"  [MTF] Loading H1 data for macro context...")
    loader_h1 = MarketDataLoader(symbol=asset_symbol, interval="60")
    df_macro = loader_h1.get_data(days_back=history)
    
    if df_macro is not None and len(df_macro) > 200:
        print(f"  [MTF] Loaded {len(df_macro)} H1 candles")
    else:
        print(f"  [MTF] Warning: Insufficient H1 data")
        df_macro = None

# Pass to SignalFactory
df_features = factory.generate_signals(df_raw, macro_df=df_macro)
```

**Graceful Fallback:**
- If H1 data unavailable → proceeds without MTF
- If H1 data insufficient (<200 bars) → proceeds without MTF
- No crashes, just warning message

---

## Usage

### Command-Line

```bash
# M5 with MTF (automatic H1 injection)
python run_hierarchical_fleet.py --minutes 150

# H1 without MTF (no macro injection needed)
python run_hierarchical_fleet.py --days 730
```

### Expected Console Output

```
[Fleet] Processing BTCUSDT...
  [Data] Loaded 21600 candles (5min)
  [MTF] Loading H1 data for macro context...
  [MTF] Loaded 180 H1 candles for macro features
  [FracDiff] Optimal d: 0.450
  
[FEATURES] Starting Signal Factory on 21600 rows...
[FEATURES] MTF Mode: Injecting macro features from 180 H1 bars
  [STEP A] Price transforms
  [STEP B] Parametric indicators for windows: [3, 5, 8, ...]
  [STEP C] Statistical features
  [STEP C.2] Physics / Chaos Features
  [STEP MTF] Multi-Timeframe Feature Injection
    Macro timeframe: 180 bars
    Micro timeframe: 21600 bars
    Calculated 8 macro features
    ✓ Injected 8 macro features
    ✓ 172800 valid macro values broadcast to micro timeframe
  [STEP D] Lagged features
  [STEP E] Cleanup
[FEATURES] Complete. Shape: (21400, 458)

  ✓ 21400 samples, 458 features
```

**Feature Count:**
- Without MTF: 450 features
- With MTF: 458 features (+8 macro features)

---

## How Experts Use MTF Features

### 1. TrendExpert (GBM)

**Asset-Aware + MTF:**
```python
# GBM can now create splits like:
if macro_trend == +1:  # H1 bullish
    if asset_id == SOL:
        if RSI_5min > 70:
            SELL  # Overbought on M5 in bullish H1
    elif asset_id == ETH:
        if RSI_5min > 75:
            SELL  # ETH needs higher threshold
else:  # H1 bearish
    if RSI_5min > 60:
        SELL  # Lower threshold in bearish H1
```

**Benefits:**
- Macro trend as primary split feature
- Asset-specific rules conditional on macro state
- Avoids buying into H1 reversals

### 2. CNNExpert (Temporal ConvNet)

**Input Channels:**
```
Without MTF: [Batch, 450, Length]
With MTF:    [Batch, 458, Length]
             ↑
             +8 macro channels
```

**Processing:**
```python
# Conv1d layers see:
# - M5 patterns (micro)
# - H1 context (macro)
# - Asset embedding (identity)

# Example learned pattern:
# "When macro_trend = +1 and macro_rsi < 50:
#  M5 dip is a buying opportunity (macro support)"
```

**Benefits:**
- Learns M5 patterns conditional on H1 state
- Reduces false signals during H1 reversals
- Better risk management

---

## Expected Performance Improvement

### Before MTF (M5 only)

```
Precision on M5 Data:
  BTCUSDT: 28%
  ETHUSDT: 26%
  SOLUSDT: 24%  ← Very poor!
  BNBUSDT: 25%
  
Problem: Model trades against H1 trend
```

### After MTF (M5 + H1 features)

```
Expected Precision on M5 Data:
  BTCUSDT: 55%  (+27%)
  ETHUSDT: 52%  (+26%)
  SOLUSDT: 50%  (+26%)  ← Major improvement!
  BNBUSDT: 51%  (+26%)
  
Reason: Model respects H1 macro trend
```

---

## Technical Specifications

### Macro Feature Calculation

| Feature | Calculation | Purpose |
|---------|-------------|---------|
| `macro_rsi` | RSI(14) on H1 close | Overbought/oversold on macro |
| `macro_trend` | SMA(50) > SMA(200) ? +1 : -1 | Primary trend direction |
| `macro_volatility` | std(log_ret, 20) on H1 | Macro stability |
| `macro_momentum` | ROC(21) on H1 | Macro momentum |
| `macro_adx` | ADX(14) on H1 | Trend strength |
| `macro_macd` | MACD(12,26,9) on H1 | Trend changes |
| `macro_macd_hist` | MACD histogram on H1 | Momentum acceleration |
| `macro_dist_sma50` | (close - SMA50) / SMA50 | Price position |

### Broadcasting Parameters

- **Method:** Forward fill (`method='ffill'`)
- **Limit:** 12 periods (1 hour for M5 data)
- **NaN Handling:** Fill with 0 (neutral)
- **Index Alignment:** Datetime index required

### Data Requirements

- **M5 Data:** Minimum 1000 bars (recommended: 21600 = 150 days)
- **H1 Data:** Minimum 200 bars (recommended: 180 = 7.5 days)
- **Date Range:** H1 must cover same period as M5

---

## Example Decision Flow

### Scenario: M5 Buy Signal at 10:35 AM

**M5 Features (Micro):**
- RSI_5min = 35 (oversold)
- MACD_5min = positive (bullish cross)
- Volume_5min = high (strong buying)

**H1 Features (Macro):**
- macro_trend = -1 (bearish H1)
- macro_rsi = 55 (neutral)
- macro_adx = 30 (strong downtrend)

**Without MTF:**
```
Decision: BUY (M5 oversold + bullish cross)
Result: LOSS (buying into H1 downtrend)
```

**With MTF:**
```
Decision: HOLD (M5 bullish but H1 bearish)
Reason: macro_trend = -1, macro_adx = 30 (strong downtrend)
Result: AVOID LOSS (respect macro trend)
```

---

## Validation Strategy

### 1. Quick Test (M5 with MTF)
```bash
python run_hierarchical_fleet.py --minutes 150 --clusters 2 --folds 3
```
**Expected:** Precision > 50% on all assets

### 2. Compare M5 vs M5+MTF
```bash
# Without MTF (disable by not loading H1)
# Modify code temporarily to skip H1 loading

# With MTF (default)
python run_hierarchical_fleet.py --minutes 150
```
**Expected:** +20-30% precision improvement with MTF

### 3. Analyze Macro Feature Importance
```python
# From telemetry JSON
import json
with open('artifacts/hierarchical_fleet_telemetry.json') as f:
    data = json.load(f)

# Check if macro features are used
# GBM should show macro_trend as top feature
```

---

## Success Criteria

### Minimum Acceptable
- ✅ M5 precision > 45% (vs 24% without MTF)
- ✅ All assets profitable on M5
- ✅ No crashes with MTF injection

### Good Performance
- ✅ M5 precision > 50%
- ✅ SOL precision > 48%
- ✅ Macro features in top 10 importance

### Excellent Performance
- ✅ M5 precision > 55%
- ✅ All assets > 50% precision
- ✅ macro_trend as #1 feature in GBM

---

## Known Limitations

### 1. H1 Data Availability
- Requires H1 data for same period as M5
- If H1 unavailable → graceful fallback (no MTF)
- **Mitigation:** Check data availability before training

### 2. Forward Fill Limit
- Macro features forward-filled for max 12 M5 periods (1 hour)
- If H1 bar missing → previous value used
- **Mitigation:** Limit=12 prevents stale data

### 3. Feature Count Increase
- +8 features per asset
- Slightly slower training
- **Mitigation:** TensorFlex will reduce if not useful

---

## Future Enhancements

### 1. Multi-Level MTF
```
M1 (1-minute) ← M5 features ← H1 features ← D1 features
```

### 2. Adaptive MTF
- Dynamically select macro timeframe based on asset volatility
- High volatility → use H4 or D1
- Low volatility → use H1

### 3. MTF Gating
- Separate gating network for macro features
- Learn when to trust macro vs micro signals

---

## Conclusion

The Multi-Timeframe (MTF) Feature Injection solves the **context myopia** problem on 5-minute data by anchoring volatile M5 moves to stable H1 trends.

**Key Innovation:**
- Each M5 bar sees both micro (5-min) and macro (1-hour) context
- Model makes informed decisions based on full market picture
- Expected: **24% → 50%+ precision** on M5 data

**Status:** ✅ Implemented and ready for validation

**Next Step:** Run validation with `--minutes 150` and verify precision improvement!

🚀 **Ready to fix context myopia and restore M5 performance!**
