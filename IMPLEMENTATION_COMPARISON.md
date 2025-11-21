# Implementation Comparison: Before vs After

## Architecture Overview

### BEFORE: Monolithic Assembly
```
For each asset (11 total):
    ├─ Load M5 data (730 days = 210K rows)
    ├─ Load H1 data (730 days = 17.5K rows)
    ├─ Generate 2000+ features
    ├─ Merge M5 + H1
    ├─ Apply physics
    └─ Append to global list

Concatenate all assets → 18GB tensor
Run AlphaCouncil → MemoryError ❌
```

### AFTER: Scout & Filter
```
SCOUT (Asset #1 only):
    ├─ Load M5 data (180 days = 52K rows)
    ├─ Load H1 data (730 days = 17.5K rows)
    ├─ Generate 2000+ features
    ├─ Run AlphaCouncil → Select 25 features
    └─ Save filtered parquet (30 columns)

FLEET (Assets #2-11):
    For each asset:
        ├─ Load M5 data (180 days)
        ├─ Load H1 data (730 days)
        ├─ Generate 2000+ features
        ├─ Filter to scout schema (30 columns)
        ├─ Save filtered parquet
        └─ Delete full dataframe, gc.collect()

ASSEMBLY:
    ├─ Load all filtered parquets
    └─ Concatenate → 500MB tensor ✅
```

## Code Changes Detail

### `src/features/alpha_council.py`

| Component | Before | After |
|-----------|--------|-------|
| **Philosophy** | Voting ensemble | Block-diagonal structure |
| **Method** | 3 independent experts | Hierarchical clustering |
| **Experts** | Lasso + RF + MI | Correlation clusters |
| **Selection** | ≥2 votes required | Proportional block sampling |
| **Diversity** | Not guaranteed | Enforced (max 85% intra-block corr) |
| **Lines of Code** | 163 | 134 |
| **Dependencies** | sklearn only | sklearn + scipy |

#### Key Algorithm Changes

**OLD: Voting System**
```python
def screen_features(X, y):
    # Expert A: Lasso coefficients
    lasso_scores = fit_lasso(X, y)
    
    # Expert B: Random Forest importance
    rf_scores = fit_rf(X, y)
    
    # Expert C: Mutual Information
    mi_scores = mutual_info(X, y)
    
    # Vote: Keep if ≥2 experts agree
    votes = sum([
        top_50_percent(lasso_scores),
        top_50_percent(rf_scores),
        top_50_percent(mi_scores)
    ])
    
    return features_with_votes >= 2
```

**NEW: Block-Diagonal**
```python
def screen_features(X, y, n_features=25):
    # 1. Find correlation structure
    clusters = hierarchical_clustering(
        correlation_matrix(X),
        threshold=0.5
    )
    
    # 2. Rank blocks by predictive power
    block_scores = [
        mutual_info(block_mean, y)
        for block in clusters
    ]
    
    # 3. Allocate budget proportionally
    for block in sorted_blocks:
        allocation = n_features * (block_score / total_score)
        
        # 4. Select diverse leaders from block
        leaders = remove_redundant(
            block_features,
            max_correlation=0.85
        )
        
        selected.extend(leaders[:allocation])
    
    return selected[:n_features]
```

### `run_deep_research.py`

| Component | Before | After |
|-----------|--------|-------|
| **M5 History** | 730 days | 180 days |
| **H1 History** | 730 days | 730 days (unchanged) |
| **Processing** | Parallel (all assets) | Sequential (scout → fleet) |
| **Feature Selection** | After assembly | Before assembly (scout only) |
| **Memory Strategy** | Accumulate in RAM | Disk-based sharding |
| **Schema** | Variable per asset | Fixed (scout-defined) |
| **Lines of Code** | 248 | 275 |

#### Pipeline Flow Changes

**OLD: Parallel Processing**
```python
def run_pipeline():
    dfs = []
    for asset in ASSET_LIST:
        df = process_asset(asset)  # Full feature set
        dfs.append(df)
    
    global_df = pd.concat(dfs)  # ❌ MemoryError here
    
    survivors = AlphaCouncil().screen_features(global_df, y)
    X = global_df[survivors]
```

**NEW: Scout → Fleet → Assembly**
```python
def run_pipeline():
    # Phase 1: Scout
    scout_df = process_asset(ASSET_LIST[0])
    selected_features = AlphaCouncil().screen_features(scout_df, y)
    scout_df[selected_features].to_parquet("scout.parquet")
    del scout_df
    
    # Phase 2: Fleet
    for asset in ASSET_LIST[1:]:
        full_df = process_asset(asset)
        filtered_df = full_df[selected_features]
        filtered_df.to_parquet(f"{asset}.parquet")
        del full_df, filtered_df
        gc.collect()
    
    # Phase 3: Assembly
    global_df = pd.concat([
        pd.read_parquet(f) 
        for f in parquet_files
    ])  # ✅ Only 30 columns, fits in RAM
```

## Memory Footprint Analysis

### Per-Asset Memory Usage

| Stage | Before | After | Reduction |
|-------|--------|-------|-----------|
| **M5 Raw Data** | 210K × 6 × 8 bytes = 10 MB | 52K × 6 × 8 bytes = 2.5 MB | 75% |
| **H1 Raw Data** | 17.5K × 6 × 8 bytes = 840 KB | 17.5K × 6 × 8 bytes = 840 KB | 0% |
| **M5 Features** | 210K × 1000 × 8 bytes = 1.6 GB | 52K × 1000 × 8 bytes = 400 MB | 75% |
| **H1 Features** | 17.5K × 1000 × 8 bytes = 140 MB | 17.5K × 1000 × 8 bytes = 140 MB | 0% |
| **Merged Features** | 210K × 2000 × 8 bytes = 3.2 GB | 52K × 2000 × 8 bytes = 800 MB | 75% |
| **Filtered Features** | N/A | 52K × 30 × 4 bytes = 6 MB | 99.8% |
| **Saved to Disk** | 3.2 GB (float64) | 6 MB (float32) | 99.8% |

### Global Assembly Memory

| Metric | Before | After | Reduction |
|--------|--------|-------|-----------|
| **Total Rows** | 11 × 210K = 2.31M | 11 × 52K = 572K | 75% |
| **Total Columns** | ~2000 | 30 | 98.5% |
| **Data Type** | float64 (8 bytes) | float32 (4 bytes) | 50% |
| **Total Memory** | 2.31M × 2000 × 8 = 36 GB | 572K × 30 × 4 = 68 MB | 99.8% |
| **Peak RAM** | 36 GB (crashed) | ~500 MB | 98.6% |

## Performance Expectations

### Time Complexity

| Operation | Before | After | Change |
|-----------|--------|-------|--------|
| **Data Loading** | O(11 × 730 days) | O(1 × 730 + 10 × 180) | 50% faster |
| **Feature Generation** | O(11 × 2000 × 210K) | O(11 × 2000 × 52K) | 75% faster |
| **Feature Selection** | O(2000 × 2.31M) | O(2000 × 52K) | 97% faster |
| **Assembly** | O(11 × 3.2 GB) | O(11 × 6 MB) | 99% faster |
| **Total Pipeline** | ~60 min (crashed) | ~15 min (estimated) | 75% faster |

### Quality Metrics

| Metric | Before | After | Expected Change |
|--------|--------|-------|-----------------|
| **Feature Diversity** | Unknown | Guaranteed | ✅ Better |
| **Inter-feature Correlation** | Variable | Max 85% intra-block | ✅ Better |
| **Predictive Power** | High | High (top blocks) | ≈ Same |
| **Model Generalization** | Good | Better (less overfitting) | ✅ Better |
| **Training Speed** | Slow (2000 features) | Fast (30 features) | 60× faster |

## Risk Assessment

### Potential Issues

1. **Scout Bias**: First asset (BTCUSDT) may not represent all assets
   - **Mitigation**: BTC is most liquid, likely has best feature coverage
   - **Alternative**: Use multi-asset scout (top 3 assets)

2. **Feature Missing**: Some assets may lack scout-selected features
   - **Mitigation**: Fill with 0.0 (implemented)
   - **Validation**: Check for excessive zeros in fleet assets

3. **Reduced History**: 180 days M5 may miss long-term patterns
   - **Mitigation**: H1 still has 730 days for macro context
   - **Validation**: Compare performance to baseline

4. **Clustering Instability**: Hierarchical clustering may vary with data
   - **Mitigation**: Fixed random_state, deterministic linkage
   - **Validation**: Run multiple times, check consistency

### Rollback Triggers

If any of these occur, revert to previous version:
- [ ] Scout phase fails to select features
- [ ] Fleet assets have >50% zero-filled features
- [ ] Global assembly still triggers MemoryError
- [ ] Model performance drops >10% vs baseline
- [ ] Selected features show >90% correlation

## Testing Protocol

### Unit Tests
```python
# Test 1: AlphaCouncil clustering
def test_clustering():
    X = generate_correlated_features(n=100, blocks=5)
    council = AlphaCouncil()
    clusters = council._get_correlation_clusters(X)
    assert len(clusters) == 5

# Test 2: Leader selection
def test_leader_selection():
    X = generate_redundant_features(n=10, redundancy=0.9)
    council = AlphaCouncil()
    leaders = council._apply_leader_follower_constraint(X, X.columns)
    assert len(leaders) < 10

# Test 3: Scout schema propagation
def test_schema_propagation():
    scout_features = ["feat_1", "feat_2", "asset_id", "close"]
    fleet_df = generate_asset_data(features=["feat_1", "feat_2", "feat_3"])
    filtered = fleet_df[scout_features]
    assert filtered.shape[1] == 4
```

### Integration Tests
```python
# Test 4: End-to-end pipeline
def test_full_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock data loader
        loader = MockMarketDataLoader()
        
        # Run pipeline
        run_pipeline()
        
        # Validate outputs
        assert Path("artifacts/money_machine_snapshot.parquet").exists()
        assert not TEMP_DIR.exists()  # Cleanup worked
```

### Manual Validation
1. **Memory Monitoring**: Run with Task Manager open
2. **Feature Inspection**: Check correlation matrix of selected features
3. **Performance Comparison**: Compare metrics to baseline
4. **Shard Validation**: Inspect saved parquet files for consistency

## Migration Guide

### Step 1: Backup Current State
```bash
git add -A
git commit -m "Pre-refactor checkpoint"
git tag pre-scout-filter
```

### Step 2: Update Dependencies
```bash
pip install scipy  # Required for hierarchical clustering
```

### Step 3: Run Refactored Pipeline
```bash
python run_deep_research.py
```

### Step 4: Validate Results
```bash
# Check memory usage
python -c "import psutil; print(f'RAM: {psutil.virtual_memory().percent}%')"

# Inspect selected features
python -c "
from pathlib import Path
import pandas as pd
df = pd.read_parquet('temp_processed_assets/BTCUSDT.parquet')
print(f'Columns: {df.columns.tolist()}')
print(f'Shape: {df.shape}')
"
```

### Step 5: Compare Performance
```bash
# Load validation snapshot
python -c "
import pandas as pd
df = pd.read_parquet('artifacts/money_machine_snapshot.parquet')
print(df.describe())
"
```

## Conclusion

The Scout & Filter refactoring transforms the pipeline from a memory-intensive monolithic approach to a scalable, disk-based architecture. By selecting features BEFORE global assembly and limiting M5 history to 180 days, we achieve:

✅ **99.8% memory reduction** (36 GB → 68 MB)  
✅ **Guaranteed feature diversity** (block-diagonal structure)  
✅ **Faster training** (30 features vs 2000)  
✅ **Scalable to 100+ assets** (disk-based sharding)  

The trade-offs are minimal:
- Slight scout bias (mitigated by using BTC as representative)
- Reduced M5 history (compensated by full H1 history)
- Sequential processing (faster overall due to reduced data volume)

**Recommendation**: Deploy to production after validation testing.
