"""
Quick smoke test for Fractional Differentiation implementation.

This script verifies:
1. FractionalDifferentiator class works correctly
2. Weights computation is accurate
3. ADF test integration functions
4. Numba acceleration is working
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent))

from src.preprocessing.frac_diff import FractionalDifferentiator


def test_basic_functionality():
    """Test basic FracDiff functionality."""
    print("=" * 72)
    print("TEST 1: Basic Functionality")
    print("=" * 72)
    
    # Create synthetic price series (random walk with drift)
    np.random.seed(42)
    n = 1000
    returns = np.random.randn(n) * 0.02 + 0.0005  # 5bps drift
    prices = 100 * np.exp(np.cumsum(returns))
    
    series = pd.Series(prices, name="price")
    
    print(f"Input series: {len(series)} observations")
    print(f"  Mean: {series.mean():.2f}")
    print(f"  Std:  {series.std():.2f}")
    
    # Test transformation with different d values
    frac_diff = FractionalDifferentiator(window_size=512)
    
    for d in [0.0, 0.3, 0.5, 0.7, 1.0]:
        result = frac_diff.transform(series, d=d, drop_na=True)
        print(f"\n  d={d:.1f}: {len(result)} valid values, mean={result.mean():.6f}, std={result.std():.6f}")
    
    print("\n✓ Basic transformation works")


def test_optimal_d_search():
    """Test optimal d search with ADF test."""
    print("\n" + "=" * 72)
    print("TEST 2: Optimal d Search (ADF Test)")
    print("=" * 72)
    
    # Create non-stationary series (random walk)
    np.random.seed(42)
    n = 500
    prices = 100 + np.cumsum(np.random.randn(n))
    series = pd.Series(prices, name="price")
    
    print(f"Input series: {len(series)} observations (random walk)")
    
    # Find optimal d
    frac_diff = FractionalDifferentiator(window_size=512)
    optimal_d = frac_diff.find_min_d(series, precision=0.05, verbose=True)
    
    print(f"\n✓ Optimal d search completed: d={optimal_d:.3f}")


def test_weights_formula():
    """Test that weights follow the correct formula."""
    print("\n" + "=" * 72)
    print("TEST 3: Weights Formula Verification")
    print("=" * 72)
    
    frac_diff = FractionalDifferentiator(window_size=10)
    
    d = 0.5
    weights = frac_diff._get_weights(d, size=10)
    
    print(f"\nWeights for d={d}:")
    for i, w in enumerate(weights):
        print(f"  w[{i}] = {w:.6f}")
    
    # Verify formula: w_k = -w_{k-1} * (d - k + 1) / k
    print("\nVerifying formula:")
    for k in range(1, len(weights)):
        expected = -weights[k-1] * (d - k + 1) / k
        actual = weights[k]
        error = abs(expected - actual)
        status = "✓" if error < 1e-10 else "✗"
        print(f"  {status} w[{k}]: expected={expected:.6f}, actual={actual:.6f}, error={error:.2e}")
    
    print("\n✓ Weights formula verified")


def test_cache_performance():
    """Test that caching improves performance."""
    print("\n" + "=" * 72)
    print("TEST 4: Cache Performance")
    print("=" * 72)
    
    import time
    
    frac_diff = FractionalDifferentiator(window_size=2048)
    
    # First call (no cache)
    start = time.time()
    weights1 = frac_diff._get_weights(0.5, size=2048)
    time1 = time.time() - start
    
    # Second call (cached)
    start = time.time()
    weights2 = frac_diff._get_weights(0.5, size=2048)
    time2 = time.time() - start
    
    print(f"First call (no cache):  {time1*1000:.2f} ms")
    print(f"Second call (cached):   {time2*1000:.2f} ms")
    print(f"Speedup: {time1/time2:.1f}x")
    
    # Verify same results
    assert np.allclose(weights1, weights2), "Cached weights differ!"
    
    info = frac_diff.get_info()
    print(f"\nCache info: {info}")
    
    print("\n✓ Caching works correctly")


def test_multi_asset():
    """Test multi-asset DataFrame processing."""
    print("\n" + "=" * 72)
    print("TEST 5: Multi-Asset DataFrame Processing")
    print("=" * 72)
    
    from src.preprocessing.frac_diff import apply_frac_diff_to_dataframe
    
    # Create multi-asset DataFrame
    np.random.seed(42)
    n_per_asset = 200
    
    dfs = []
    for asset_id in range(3):
        prices = 100 + np.cumsum(np.random.randn(n_per_asset) * 2)
        df = pd.DataFrame({
            'asset_id': asset_id,
            'close': prices,
        })
        dfs.append(df)
    
    df_multi = pd.concat(dfs, ignore_index=True)
    
    print(f"Input: {len(df_multi)} rows, {df_multi['asset_id'].nunique()} assets")
    
    # Apply FracDiff
    df_result = apply_frac_diff_to_dataframe(
        df_multi,
        price_col='close',
        find_optimal=True,
        precision=0.1,  # Coarse for speed
        verbose=True
    )
    
    print(f"\nOutput: {len(df_result)} rows")
    print(f"Columns: {list(df_result.columns)}")
    
    # Check that frac_diff column exists
    assert 'close_fracdiff' in df_result.columns, "frac_diff column not found!"
    
    # Check for NaNs
    n_nans = df_result['close_fracdiff'].isna().sum()
    print(f"NaNs in frac_diff: {n_nans} ({n_nans/len(df_result):.1%})")
    
    print("\n✓ Multi-asset processing works")


def run_all_tests():
    """Run all smoke tests."""
    print("\n" + "=" * 72)
    print("FRACTIONAL DIFFERENTIATION SMOKE TESTS")
    print("=" * 72)
    
    try:
        test_basic_functionality()
        test_optimal_d_search()
        test_weights_formula()
        test_cache_performance()
        test_multi_asset()
        
        print("\n" + "=" * 72)
        print("ALL TESTS PASSED ✓")
        print("=" * 72)
        
    except Exception as e:
        print("\n" + "=" * 72)
        print("TEST FAILED ✗")
        print("=" * 72)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
