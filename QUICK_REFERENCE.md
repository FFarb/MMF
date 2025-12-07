# Quick Reference: Scout & Filter Pipeline

## Running the Pipeline

### Basic Execution
```bash
python run_deep_research.py
```

### Expected Output
```
========================================================================
          MULTI-ASSET NEURO-SYMBOLIC TRADING SYSTEM
          (Smart Horizon & Scout Assembly Mode)
========================================================================

[1] SCOUT PHASE (Feature Selection on Leader)

    >> Processing BTCUSDT (ID: 0)...
       Generating Strategic (H1) features...
       Generating Execution (M5) features...
       Merging Contexts...
       Applying Chaos Physics...
       -> Saved: temp_processed_assets\BTCUSDT.parquet | Shape: (52000, 2000)

    Running Alpha Council on Scout (BTCUSDT)...
    [Alpha Council] Structuring 2000 raw features...
    [Alpha Council] Identified 15 structural blocks.
    SCOUT SELECTED 25 FEATURES: ['rsi_14', 'macd_signal', ...]

[2] FLEET PHASE (Processing Remaining Assets)

    >> Processing ETHUSDT (ID: 1)...
       -> Saved filtered shard: temp_processed_assets\ETHUSDT.parquet
    
    >> Processing XRPUSDT (ID: 2)...
       -> Saved filtered shard: temp_processed_assets\XRPUSDT.parquet
    
    [... 8 more assets ...]

[3] GLOBAL ASSEMBLY
    Global Tensor Assembled: (572000, 30)

[4] MIXED MODE TRAINING
    Training Config: {'n_estimators': 500, 'epochs': 100}
    [Training progress...]

[5] VALIDATION & SNAPSHOT
    Snapshot saved.
    [Threshold tuning results...]
```

## Configuration Tuning

### Memory Still Too High?
**Reduce M5 lookback further:**
```python
# In run_deep_research.py, line 31
M5_LOOKBACK_DAYS = 90  # Down from 180
```

### Want More Features?
**Increase feature budget:**
```python
# In run_deep_research.py, line 161
selected_alphas = council.screen_features(
    df_council[candidates], 
    y_council, 
    n_features=35  # Up from 25
)
```

### Adjust Clustering Sensitivity
**Tighter clusters (more blocks):**
```python
# In src/features/alpha_council.py, line 97
clusters = self._get_correlation_clusters(X, threshold=0.3)  # Down from 0.5
```

**Looser clusters (fewer blocks):**
```python
clusters = self._get_correlation_clusters(X, threshold=0.7)  # Up from 0.5
```

### Change Redundancy Threshold
**More diverse features:**
```python
# In src/features/alpha_council.py, line 87
if abs(df[f].corr(df[existing])) > 0.75:  # Down from 0.85
```

**Allow more similar features:**
```python
if abs(df[f].corr(df[existing])) > 0.95:  # Up from 0.85
```

## Debugging Common Issues

### Issue 1: Scout Phase Fails
**Symptom:**
```
CRITICAL: Scout failed. Exiting.
```

**Diagnosis:**
```python
# Check if data is being fetched
python -c "
from src.data_loader import MarketDataLoader
loader = MarketDataLoader(interval='5')
loader.symbol = 'BTCUSDT'
df = loader.get_data(days_back=180)
print(f'M5 Shape: {df.shape}')
"
```

**Solutions:**
1. Check internet connection
2. Verify API credentials in `.env`
3. Check cache directory permissions
4. Try different symbol as scout: `ASSET_LIST = ["ETHUSDT", ...]`

### Issue 2: AlphaCouncil Returns No Features
**Symptom:**
```
[Alpha Council] Identified 0 structural blocks.
```

**Diagnosis:**
```python
# Check feature variance
python -c "
import pandas as pd
df = pd.read_parquet('temp_processed_assets/BTCUSDT.parquet')
print(df.var().describe())
"
```

**Solutions:**
1. Check for all-zero columns: `df.columns[df.var() == 0]`
2. Reduce clustering threshold to 0.3
3. Increase feature budget to 50

### Issue 3: Fleet Assets Missing Features
**Symptom:**
```
[ERROR] Saving shard ETHUSDT: KeyError: 'rsi_14'
```

**Diagnosis:**
```python
# Compare scout vs fleet columns
python -c "
import pandas as pd
scout = pd.read_parquet('temp_processed_assets/BTCUSDT.parquet')
fleet = pd.read_parquet('temp_processed_assets/ETHUSDT.parquet')
missing = set(scout.columns) - set(fleet.columns)
print(f'Missing: {missing}')
"
```

**Solutions:**
1. Check if `SignalFactory` is deterministic
2. Verify both assets have same data quality
3. Add feature existence check before filtering

### Issue 4: MemoryError Still Occurs
**Symptom:**
```
MemoryError: Unable to allocate array
```

**Diagnosis:**
```python
# Check shard sizes
python -c "
from pathlib import Path
for f in Path('temp_processed_assets').glob('*.parquet'):
    size_mb = f.stat().st_size / 1024 / 1024
    print(f'{f.name}: {size_mb:.2f} MB')
"
```

**Solutions:**
1. Reduce M5_LOOKBACK_DAYS to 90 or 60
2. Reduce n_features to 15
3. Process fewer assets (comment out some in SYMBOLS)
4. Use sampling: `df.iloc[::2]` (every other row)

### Issue 5: Poor Model Performance
**Symptom:**
```
Validation Accuracy: 0.52 (expected >0.60)
```

**Diagnosis:**
```python
# Check feature correlation matrix
python -c "
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_parquet('temp_processed_assets/BTCUSDT.parquet')
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.savefig('feature_correlation.png')
print('Saved feature_correlation.png')
"
```

**Solutions:**
1. Increase n_features to 35-50
2. Use looser clustering threshold (0.7)
3. Check if physics features are included
4. Verify label quality: `y.value_counts()`

## Monitoring & Profiling

### Memory Usage
```python
# Add to run_deep_research.py after each phase
import psutil
process = psutil.Process()
mem_mb = process.memory_info().rss / 1024 / 1024
print(f"Memory: {mem_mb:.2f} MB")
```

### Execution Time
```python
# Add timing decorators
import time

def timed_phase(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.2f}s")
        return result
    return wrapper

@timed_phase
def process_single_asset(...):
    ...
```

### Feature Selection Quality
```python
# After scout phase
python -c "
import pandas as pd
import numpy as np

df = pd.read_parquet('temp_processed_assets/BTCUSDT.parquet')
corr_matrix = df.corr().abs()

# Check off-diagonal correlations
np.fill_diagonal(corr_matrix.values, 0)
max_corr = corr_matrix.max().max()
mean_corr = corr_matrix.mean().mean()

print(f'Max inter-feature correlation: {max_corr:.3f}')
print(f'Mean inter-feature correlation: {mean_corr:.3f}')
print(f'Target: Max < 0.85, Mean < 0.30')
"
```

## Advanced Usage

### Multi-Asset Scout
Use top 3 assets for feature selection:
```python
# In run_deep_research.py, replace SCOUT PHASE with:
scout_dfs = []
for i in range(3):
    df = process_single_asset(ASSET_LIST[i], i, loader, factory)
    if df is not None:
        scout_dfs.append(df)

df_scout = pd.concat(scout_dfs)
# ... continue with AlphaCouncil
```

### Incremental Processing
Process assets in batches to avoid crashes:
```python
# In run_deep_research.py
BATCH_SIZE = 3
for batch_start in range(1, len(ASSET_LIST), BATCH_SIZE):
    batch = ASSET_LIST[batch_start:batch_start + BATCH_SIZE]
    for asset_idx, symbol in enumerate(batch, start=batch_start):
        # ... process asset
    gc.collect()
    time.sleep(1)  # Cool down
```

### Custom Feature Blocks
Manually define feature groups:
```python
# In src/features/alpha_council.py
def screen_features(self, X, y, n_features=25, custom_blocks=None):
    if custom_blocks:
        clusters = custom_blocks
    else:
        clusters = self._get_correlation_clusters(X)
    # ... continue

# Usage:
custom_blocks = {
    1: ['rsi_14', 'rsi_21', 'rsi_28'],  # Momentum block
    2: ['macd', 'macd_signal', 'macd_hist'],  # Trend block
    3: ['bb_upper', 'bb_lower', 'bb_width'],  # Volatility block
}
selected = council.screen_features(X, y, custom_blocks=custom_blocks)
```

## Validation Checklist

After running the pipeline, verify:

- [ ] **Scout completed**: `temp_processed_assets/BTCUSDT.parquet` exists
- [ ] **Fleet completed**: 10 more parquet files in `temp_processed_assets/`
- [ ] **Schema consistency**: All parquets have same columns
- [ ] **Memory usage**: Peak RAM < 2GB
- [ ] **Feature diversity**: Mean inter-feature correlation < 0.30
- [ ] **Model trained**: `artifacts/money_machine_snapshot.parquet` exists
- [ ] **Cleanup worked**: `temp_processed_assets/` deleted after completion
- [ ] **Performance**: Validation accuracy > 0.55

## Quick Fixes

### Reset Everything
```bash
rm -rf temp_processed_assets artifacts __pycache__ src/__pycache__
python run_deep_research.py
```

### Inspect Intermediate Results
```python
# Don't delete temp files - comment out cleanup
# In run_deep_research.py, line 271:
# cleanup_temp_dir()  # Comment this out

# Then inspect:
import pandas as pd
scout = pd.read_parquet('temp_processed_assets/BTCUSDT.parquet')
print(scout.info())
print(scout.describe())
```

### Test AlphaCouncil Standalone
```python
from src.features.alpha_council import AlphaCouncil
import pandas as pd
import numpy as np

# Generate synthetic data
X = pd.DataFrame(np.random.randn(1000, 100))
y = pd.Series(np.random.randint(0, 2, 1000))

council = AlphaCouncil()
selected = council.screen_features(X, y, n_features=10)
print(f"Selected: {selected}")
```

## Performance Benchmarks

Expected timings on typical hardware (16GB RAM, 8-core CPU):

| Phase | Time | Memory |
|-------|------|--------|
| Scout Processing | 2-3 min | 800 MB |
| AlphaCouncil | 30-60 sec | 1.2 GB |
| Fleet Processing (10 assets) | 8-12 min | 500 MB |
| Global Assembly | 5-10 sec | 300 MB |
| Model Training | 3-5 min | 600 MB |
| **Total** | **15-20 min** | **Peak: 1.2 GB** |

If your timings are significantly different:
- **Slower**: Check disk I/O, reduce feature count
- **Faster**: Great! Consider increasing n_features
- **More memory**: Reduce M5_LOOKBACK_DAYS or n_features
- **Less memory**: You can increase batch size or history

## Support

If issues persist:
1. Check `REFACTORING_SUMMARY.md` for architecture overview
2. Check `IMPLEMENTATION_COMPARISON.md` for detailed changes
3. Review git history: `git log --oneline`
4. Rollback if needed: `git checkout pre-scout-filter`

---
**Last Updated**: 2025-11-21  
**Version**: Scout & Filter v1.0
