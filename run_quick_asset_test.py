"""
Quick validation test on a single cluster to verify asset-aware experts.

This script tests asset-aware functionality on Cluster 1 (the problematic one)
to quickly verify that SOL and BNB performance improves.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

# Run hierarchical fleet but only on specific assets
from run_hierarchical_fleet import run_hierarchical_fleet_training

# Test on Cluster 1 assets (the problematic ones)
TEST_ASSETS = [
    'BNBUSDT',
    'SOLUSDT', 
    'XRPUSDT',
]

print("=" * 72)
print("QUICK ASSET-AWARE VALIDATION TEST")
print("Testing on Cluster 1 assets: BNB, SOL, XRP")
print("=" * 72)

# Run with fewer folds for speed
results = run_hierarchical_fleet_training(
    assets=TEST_ASSETS,
    n_clusters=1,  # Single cluster
    n_folds=3,     # Fewer folds for speed
    history_days=365,  # 1 year for speed
    max_frac_diff_d=0.65,
    market_factor_method='mean',  # Faster than PCA
)

print("\n" + "=" * 72)
print("QUICK TEST COMPLETE")
print("=" * 72)
print("\nCheck if SOL and BNB precision improved!")
print("Target: SOL > 58%, BNB > 60%")
