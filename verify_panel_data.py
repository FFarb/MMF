"""
Quick verification script to test Panel Data structure in global validation.

This script loads a small sample of data and verifies:
1. Timestamp sorting is correct
2. TimeSeriesSplit creates valid folds
3. No temporal leakage between train/val
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

# Create mock panel data
np.random.seed(42)

# 3 assets, 100 timestamps each
assets = ['BTC', 'ETH', 'SOL']
timestamps = pd.date_range('2024-01-01', periods=100, freq='1H')

data = []
for asset in assets:
    for ts in timestamps:
        data.append({
            'timestamp': ts,
            'asset_id': asset,
            'feature1': np.random.randn(),
            'feature2': np.random.randn(),
            'target': np.random.randint(0, 2),
        })

df = pd.DataFrame(data)

print("=" * 72)
print("PANEL DATA STRUCTURE VERIFICATION")
print("=" * 72)

# Test 1: Before sorting (wrong)
print("\n[Test 1] BEFORE SORTING (WRONG):")
df_wrong = df.copy()
print(f"  First 10 rows:")
print(df_wrong[['timestamp', 'asset_id']].head(10))

# Test 2: After sorting (correct)
print("\n[Test 2] AFTER SORTING (CORRECT):")
df_correct = df.copy()
df_correct.sort_values(by=['timestamp', 'asset_id'], inplace=True)
df_correct.reset_index(drop=True, inplace=True)
print(f"  First 10 rows:")
print(df_correct[['timestamp', 'asset_id']].head(10))

# Test 3: Verify monotonic timestamp
print("\n[Test 3] TIMESTAMP MONOTONICITY:")
is_monotonic = df_correct['timestamp'].is_monotonic_increasing
print(f"  Timestamps monotonically increasing: {is_monotonic} {'PASS' if is_monotonic else 'FAIL'}")

# Test 4: TimeSeriesSplit validation
print("\n[Test 4] TIME SERIES SPLIT VALIDATION:")
X = df_correct.drop(columns=['target'])
y = df_correct['target']
timestamp_col = df_correct['timestamp']

tscv = TimeSeriesSplit(n_splits=3)

all_valid = True
for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
    train_max_time = timestamp_col.iloc[train_idx].max()
    val_min_time = timestamp_col.iloc[val_idx].min()
    
    is_valid = train_max_time < val_min_time
    all_valid = all_valid and is_valid
    
    print(f"  Fold {fold_idx}:")
    print(f"    Train: {timestamp_col.iloc[train_idx].min()} to {train_max_time}")
    print(f"    Val:   {val_min_time} to {timestamp_col.iloc[val_idx].max()}")
    print(f"    Valid: {is_valid} {'PASS' if is_valid else 'FAIL - TEMPORAL LEAKAGE!'}")

print(f"\n  Overall: {'PASS - ALL FOLDS VALID' if all_valid else 'FAIL - TEMPORAL LEAKAGE DETECTED'}")

# Test 5: Asset alignment
print("\n[Test 5] ASSET ALIGNMENT:")
for asset in df_correct['asset_id'].unique():
    asset_df = df_correct[df_correct['asset_id'] == asset]
    print(f"  {asset}: {asset_df['timestamp'].min()} to {asset_df['timestamp'].max()} ({len(asset_df)} samples)")

print("\n" + "=" * 72)
print("VERIFICATION COMPLETE")
print("=" * 72)
