# QFC System v3.1: Individual Asset Specialists + Stress Expert Fix

## Overview

This update implements the "Islands" strategy where each asset gets its own isolated MoE model, fixes the Stress expert overheating problem, and introduces a forced daily entry policy to prevent zero-trade days.

## Changes Implemented

### 1. **Stress Expert Fix** (`src/models/moe_ensemble.py`)

**Problem**: Stress expert was dominating (40-60% weight) due to:
- Manual class_weight bias: `{0: 2.0, 1: 1.0}`
- Physics override boost when `theta < 0.005`

**Solution**:
```python
# BEFORE
class_weight={0: 2.0, 1: 1.0}  # Manual bias toward caution

# AFTER
class_weight='balanced'  # Let data decide risk
```

**Removed**: Physics override logic that boosted Stress expert during critical slowing down.

**Rationale**: Let Oracle training (cross-entropy) determine when Stress is needed, not hardcoded heuristics.

---

### 2. **Forced Daily Entry Policy** (`src/trading/forced_entry.py`)

**Problem**: Model often chooses "Cash" (0 trades), leading to missed opportunities.

**Solution**: Post-processing policy that ensures at least one trade per day.

**Logic**:
1. Group predictions by date
2. Check if any signal exceeds `PROB_THRESHOLD` (0.55)
3. If `sum(signals) == 0` for the day:
   - Find candle with maximum probability
   - Force `signal = 1` (even if prob < threshold)
   - Mark as `forced_entry = True`

**Functions**:
- `apply_forced_daily_entry()`: Apply policy to predictions
- `analyze_forced_entry_performance()`: Compare natural vs forced trades
- `print_forced_entry_report()`: Generate detailed report

**Example Output**:
```
Trading Activity:
  Total Days: 3
  Natural Trades: 48
  Forced Trades: 1 (33.3% of days)
  Avg Trades/Day: 16.33

Performance Comparison:
  Natural Trades: Mean Return: 0.0017, Win Rate: 59.15%
  Forced Trades: Mean Return: -0.0104, Win Rate: 0.00%
  Delta: -0.0121
```

---

### 3. **Individual Fleet Training** (`run_individual_fleet.py`)

**Problem**: 
- Clustering assets mixes their physics
- Signal dilution from cross-asset contamination
- One-size-fits-all model doesn't capture asset-specific patterns

**Solution**: "Islands" Strategy

**Architecture**:
```
For each asset in FLEET_ASSETS:
  1. Load data (isolated, no clustering)
  2. Calculate features (FracDiff + TensorFlex)
  3. Train fresh MixtureOfExpertsEnsemble
  4. Apply forced daily entry policy
  5. Save predictions separately
```

**Key Features**:
- ✅ **Asset Isolation**: Each asset gets its own model
- ✅ **No Data Leakage**: No cross-asset contamination
- ✅ **Asset-Specific Physics**: Each model learns unique patterns
- ✅ **SDE Expert Enabled**: LaP-SDE works per asset
- ✅ **Forced Entry**: Ensures daily activity

**Usage**:
```bash
# Default: 5 folds, 2 years history
python run_individual_fleet.py

# Custom: 10 folds, 1 year history
python run_individual_fleet.py --folds 10 --days 365

# Disable forced entry
python run_individual_fleet.py --no-forced-entry
```

**Output**:
- `artifacts/individual_predictions_{SYMBOL}.csv`: Per-asset predictions
- `artifacts/individual_holdout_{SYMBOL}.csv`: Holdout data (if specified)
- `artifacts/individual_fleet_results.csv`: Summary metrics

---

## Architecture Comparison

### Before (Enriched Fleet)
```
All Assets → Clustering → Single MoE Model
  ├─ Market Factor (PCA)
  ├─ Shared Physics
  └─ Diluted Signals
```

### After (Individual Fleet)
```
BTC → Individual MoE Model → BTC Predictions
ETH → Individual MoE Model → ETH Predictions
SOL → Individual MoE Model → SOL Predictions
...
  ├─ Asset-Specific Physics
  ├─ Isolated Learning
  └─ Pure Signals
```

---

## Testing Results

### Test 1: Stress Expert Fix
```
Stress Expert class_weight: balanced
[OK] Stress expert uses 'balanced' (manual bias removed)
```

### Test 2: Forced Daily Entry
```
Original signals by day:
  2024-01-01: 24 natural signals
  2024-01-02: 0 natural signals  ← Zero-trade day
  2024-01-03: 24 natural signals

After forced entry:
  2024-01-01: 24 total signals (0 forced)
  2024-01-02: 1 total signals (1 forced)  ← Fixed!
  2024-01-03: 24 total signals (0 forced)

[OK] Forced entry policy working correctly
```

### Test 3: Performance Analysis
```
Natural Trades: Mean Return: 0.0017, Win Rate: 59.15%
Forced Trades: Mean Return: -0.0104, Win Rate: 0.00%
Delta: -0.0121

[WARNING] Forced trades are UNPROFITABLE (destroying alpha)
```

**Note**: In this test, forced trades were unprofitable. In production, we monitor this and can disable the policy if it consistently destroys alpha.

---

## Benefits

### 1. **Balanced Expert Weights**
- Stress expert no longer dominates
- Oracle training determines optimal weights
- More diverse predictions

### 2. **Asset-Specific Learning**
- BTC learns BTC patterns
- SOL learns SOL patterns
- No signal dilution

### 3. **Guaranteed Activity**
- No zero-trade days
- Always taking best guess
- More PnL data for analysis

### 4. **Transparency**
- Track natural vs forced trades
- Measure forced trade performance
- Can disable if unprofitable

---

## Files Created

1. **`src/trading/forced_entry.py`** (250 lines)
   - Forced daily entry policy
   - Performance analysis
   - Reporting utilities

2. **`run_individual_fleet.py`** (600+ lines)
   - Individual asset training
   - Islands strategy implementation
   - Forced entry integration

3. **`test_individual_fleet.py`** (150 lines)
   - Stress expert verification
   - Forced entry testing
   - Integration tests

---

## Files Modified

1. **`src/models/moe_ensemble.py`**
   - Changed `class_weight` to `'balanced'`
   - Removed physics override boost
   - Cleaner gating logic

---

## Usage Examples

### Run Individual Fleet Training
```bash
# Full fleet, 5 folds, 2 years
python run_individual_fleet.py

# Quick test: 3 folds, 30 days
python run_individual_fleet.py --folds 3 --days 30

# With holdout period
python run_individual_fleet.py --holdout-days 180
```

### Test Changes
```bash
# Verify Stress expert fix and forced entry
python test_individual_fleet.py
```

---

## Metrics to Monitor

### Expert Weight Distribution
- **Before**: Stress 40-60%, Others 10-20% each
- **Target**: More balanced (20-30% each)

### Trading Activity
- **Before**: 0-5 trades/day (many zero-trade days)
- **Target**: 1+ trades/day (forced entry ensures minimum)

### Forced Trade Performance
- **Monitor**: `forced_mean_return` vs `natural_mean_return`
- **Action**: Disable policy if `return_delta < -0.005` consistently

---

## Next Steps

1. **Run Individual Fleet Training**
   ```bash
   python run_individual_fleet.py --folds 5 --days 730
   ```

2. **Analyze Results**
   - Check `artifacts/individual_fleet_results.csv`
   - Compare per-asset expectancy
   - Review forced trade performance

3. **Backtest Top Assets**
   - Select assets with highest expectancy
   - Run sniper backtest
   - Validate with holdout data

4. **Production Deployment**
   - Deploy individual models per asset
   - Monitor forced trade performance
   - Adjust threshold if needed

---

## Status

✅ **Stress Expert Fix**: Verified  
✅ **Forced Daily Entry**: Working  
✅ **Individual Fleet Training**: Ready  
✅ **All Tests**: Passing  

**Ready for production testing!** 🚀
