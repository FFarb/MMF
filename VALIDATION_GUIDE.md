# Hierarchical Fleet Validation Guide

## Quick Start

### Option 1: Hourly Data (Recommended for Initial Testing)
```bash
python run_hierarchical_fleet.py --days 730 --clusters 3 --folds 5
```

### Option 2: 5-Minute Data (High Resolution)
```bash
python run_hierarchical_fleet.py --minutes 150 --clusters 3 --folds 5
```

### Option 3: Quick Test (Faster Iteration)
```bash
python run_hierarchical_fleet.py --days 365 --clusters 2 --folds 3
```

---

## All Available Arguments

```bash
python run_hierarchical_fleet.py \
    --clusters 3 \              # Number of asset clusters (default: 3)
    --folds 5 \                 # Number of CV folds (default: 5)
    --days 730 \                # Days of hourly data (default: 730 = 2 years)
    --minutes 0 \               # Days of 5-minute data (0 = disabled, recommended: 120-180)
    --max-d 0.65 \              # Max fractional differentiation order (default: 0.65)
    --factor-method pca         # Market factor method: pca, mean, weighted_mean
```

---

## Recommended Configurations

### 1. Full Production Validation (Hourly)
```bash
python run_hierarchical_fleet.py --days 730 --clusters 3 --folds 5
```
- **Time:** 30-60 minutes
- **Purpose:** Complete validation with 2 years of hourly data
- **Best for:** Final validation before deployment

### 2. High-Resolution Validation (5-Minute)
```bash
python run_hierarchical_fleet.py --minutes 150 --clusters 3 --folds 5
```
- **Time:** 45-90 minutes
- **Purpose:** High-resolution pattern detection (150 days = ~5 months)
- **Best for:** Finding intraday patterns, better signal quality

### 3. Quick Development Test
```bash
python run_hierarchical_fleet.py --days 365 --clusters 2 --folds 3
```
- **Time:** 15-20 minutes
- **Purpose:** Fast iteration during development
- **Best for:** Testing changes, debugging

### 4. Deep Analysis (Maximum Data)
```bash
python run_hierarchical_fleet.py --minutes 180 --clusters 3 --folds 5 --factor-method pca
```
- **Time:** 60-120 minutes
- **Purpose:** Maximum data for deep analysis (180 days of 5-min = ~6 months)
- **Best for:** Finding optimal configuration, research

---

## Output Files

After running, you'll get:

### 1. CSV Summary (`artifacts/hierarchical_fleet_results.csv`)
```csv
Asset,Cluster,Precision,Recall,Expectancy,Folds
BTCUSDT,0,0.635,0.452,0.01270,5
ETHUSDT,0,0.615,0.452,0.01230,5
SOLUSDT,1,0.592,0.485,0.00920,5
...
```

### 2. JSON Telemetry (`artifacts/hierarchical_fleet_telemetry.json`)
```json
{
  "config": {
    "n_clusters": 3,
    "n_folds": 5,
    "interval": "5",
    "history_days_5min": 150,
    ...
  },
  "cluster_assignments": {
    "BTCUSDT": 0,
    "ETHUSDT": 0,
    "SOLUSDT": 1,
    ...
  },
  "cluster_results": {
    "0": {
      "members": ["BTCUSDT", "ETHUSDT"],
      "is_dominant": true,
      "avg_precision": 0.625,
      "avg_f1": 0.532,
      ...
    }
  },
  "per_asset_detailed": {
    "BTCUSDT": [
      {
        "cluster_id": 0,
        "fold": 1,
        "precision": 0.635,
        "recall": 0.452,
        "f1": 0.527,
        "accuracy": 0.612,
        "expectancy": 0.01270,
        "n_samples": 2500
      },
      ...
    ]
  }
}
```

---

## What to Look For

### 1. Asset-Aware Mode Confirmation
```
[TrendExpert] Asset-aware mode: 10 assets, asset_id at index 45
[TemporalConvNet] Asset-aware mode: 10 assets, emb_dim=8, input_channels=53
```
✅ **GOOD:** Asset-aware experts are active

### 2. Cluster Assignments
```
CLUSTER SUMMARY
Cluster 0 (DOMINANT): BTCUSDT, ETHUSDT
Cluster 1: BNBUSDT, SOLUSDT, XRPUSDT
Cluster 2: ADAUSDT, AVAXUSDT, DOGEUSDT, LINKUSDT, LTCUSDT
```
✅ **GOOD:** BTC and ETH in dominant cluster

### 3. Per-Asset Performance
```
PER-ASSET PERFORMANCE SUMMARY
Asset      Cluster  Precision  Recall  Expectancy
BTCUSDT    0        0.635      0.452   0.01270
SOLUSDT    1        0.592      0.485   0.00920  ← Should improve with asset-aware!
```
✅ **TARGET:** SOL > 58%, BNB > 60%

### 4. Success Criteria
```
✓ All Assets Profitable: PASS
✓ Avg Precision > 55%: PASS (58.9%)
✓ Alt Clusters Expectancy: 0.00750
```
✅ **PASS:** All criteria met

---

## Comparing Hourly vs 5-Minute

### Hourly Data (--days 730)
**Pros:**
- ✅ More history (2 years)
- ✅ Less noise
- ✅ Faster training
- ✅ Better for long-term trends

**Cons:**
- ❌ Lower resolution
- ❌ Misses intraday patterns
- ❌ Slower signal updates

### 5-Minute Data (--minutes 150)
**Pros:**
- ✅ High resolution
- ✅ Captures intraday patterns
- ✅ Better signal quality
- ✅ Faster reaction time

**Cons:**
- ❌ Less history (150 days = 5 months)
- ❌ More noise
- ❌ Slower training
- ❌ Requires more data storage

### Recommendation
**Start with hourly** (`--days 730`) for initial validation, then **test 5-minute** (`--minutes 150`) to see if higher resolution improves performance.

---

## Analyzing Results

### 1. Load JSON Telemetry
```python
import json
import pandas as pd

# Load telemetry
with open('artifacts/hierarchical_fleet_telemetry.json') as f:
    data = json.load(f)

# View cluster assignments
print(data['cluster_assignments'])

# View per-asset results
for asset, results in data['per_asset_detailed'].items():
    df = pd.DataFrame(results)
    print(f"\n{asset}:")
    print(df[['fold', 'precision', 'recall', 'expectancy']])
```

### 2. Compare Clusters
```python
# Load telemetry
with open('artifacts/hierarchical_fleet_telemetry.json') as f:
    data = json.load(f)

# Compare cluster performance
for cid, stats in data['cluster_results'].items():
    print(f"\nCluster {cid} ({'DOMINANT' if stats['is_dominant'] else 'ALT'}):")
    print(f"  Members: {stats['members']}")
    print(f"  Precision: {stats['avg_precision']:.2%}")
    print(f"  F1: {stats['avg_f1']:.2%}")
    print(f"  Expectancy: {stats['avg_expectancy']:.5f}")
```

### 3. Find Best/Worst Assets
```python
import pandas as pd

# Load CSV
df = pd.read_csv('artifacts/hierarchical_fleet_results.csv')

# Sort by expectancy
print("\nBest Performers:")
print(df.nlargest(3, 'Expectancy'))

print("\nWorst Performers:")
print(df.nsmallest(3, 'Expectancy'))
```

---

## Troubleshooting

### Error: "Insufficient data"
**Solution:** Reduce `--days` or `--minutes`
```bash
python run_hierarchical_fleet.py --days 365  # Use 1 year instead of 2
```

### Error: "Memory error"
**Solution:** Reduce folds or use hourly data
```bash
python run_hierarchical_fleet.py --days 365 --folds 3
```

### Warning: "Asset-aware mode not activated"
**Check:** Ensure asset_id column exists in data
**Fix:** This should be automatic, check data loading

### Low precision on specific asset
**Analysis:** Check JSON telemetry for that asset
**Action:** May need asset-specific tuning

---

## Next Steps After Validation

### 1. If Results are Good (All profitable, Precision > 55%)
✅ **Deploy to paper trading**
✅ **Monitor live performance**
✅ **Compare with backtest**

### 2. If Results Need Improvement
🔧 **Try 5-minute data** (`--minutes 150`)
🔧 **Adjust clusters** (`--clusters 2` or `--clusters 4`)
🔧 **Try different market factor** (`--factor-method mean`)
🔧 **Analyze JSON telemetry** for weak experts

### 3. Deep Analysis
📊 **Load JSON in Python/Jupyter**
📊 **Visualize per-asset performance**
📊 **Compare expert predictions**
📊 **Identify systematic biases**

---

## Example Workflow

```bash
# Step 1: Quick test (15 min)
python run_hierarchical_fleet.py --days 365 --clusters 2 --folds 3

# Step 2: Full hourly validation (45 min)
python run_hierarchical_fleet.py --days 730 --clusters 3 --folds 5

# Step 3: High-resolution 5-min (60 min)
python run_hierarchical_fleet.py --minutes 150 --clusters 3 --folds 5

# Step 4: Analyze results
python
>>> import json
>>> with open('artifacts/hierarchical_fleet_telemetry.json') as f:
...     data = json.load(f)
>>> # Analyze...
```

---

## Success Metrics

### Minimum Acceptable
- ✅ All assets profitable (Expectancy > 0)
- ✅ Average precision > 55%
- ✅ No temporal leakage (validated by Panel Data structure)

### Good Performance
- ✅ Average precision > 60%
- ✅ SOL precision > 58%
- ✅ BNB precision > 60%
- ✅ All clusters profitable

### Excellent Performance
- ✅ Average precision > 65%
- ✅ All assets > 60% precision
- ✅ Expectancy > 0.01 for all assets
- ✅ F1 score > 0.55

---

## Ready to Start!

**Recommended first command:**
```bash
python run_hierarchical_fleet.py --days 730 --clusters 3 --folds 5
```

This will:
- ✅ Load 2 years of hourly data
- ✅ Auto-cluster into 3 groups
- ✅ Extract market factor from dominant cluster
- ✅ Train asset-aware experts
- ✅ Save comprehensive telemetry
- ✅ Report all metrics

**Expected time:** 30-60 minutes

**Output:** CSV + JSON with complete metrics for analysis

🚀 **Let's validate the system!**
