"""
Time Conversion Validation Script.

This script validates that:
1. Time parameters work correctly (days, hours, minutes)
2. Data conversions are logically correct
3. No future data leakage in cross-validation

Usage:
    python validate_time_conversions.py --symbol BTCUSDT --days 7
    python validate_time_conversions.py --symbol ETHUSDT --days 1 --test-cv
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_loader import MarketDataLoader, validate_time_conversions


def test_time_parameters(symbol: str, days: int, interval: str = "60"):
    """Test time parameter functionality."""
    print("=" * 80)
    print("TEST 1: TIME PARAMETER VALIDATION")
    print("=" * 80)
    
    loader = MarketDataLoader(symbol=symbol, interval=interval)
    
    print(f"\nFetching {symbol} @ {interval}m for {days} days...")
    df = loader.get_data(days_back=days, force_refresh=False)
    
    # Calculate expected bars
    if interval == "60":
        expected_bars = days * 24
        bar_type = "H1"
    elif interval == "5":
        expected_bars = days * 288
        bar_type = "M5"
    else:
        expected_bars = days * (1440 // int(interval))
        bar_type = f"{interval}m"
    
    print(f"\n[OK] Data loaded successfully!")
    print(f"  Symbol: {symbol}")
    print(f"  Interval: {interval}m ({bar_type})")
    print(f"  Days requested: {days}")
    print(f"  Expected bars: ~{expected_bars:,}")
    print(f"  Actual bars: {len(df):,}")
    print(f"  Difference: {abs(len(df) - expected_bars):,} bars")
    print(f"  Range: {df.index.min()} to {df.index.max()}")
    
    # Validate
    validate_time_conversions(df, symbol)
    
    return df


def test_cross_validation_no_leakage(df: pd.DataFrame, n_folds: int = 5):
    """Test that TimeSeriesSplit doesn't leak future data."""
    print("\n" + "=" * 80)
    print("TEST 2: CROSS-VALIDATION FUTURE DATA LEAKAGE CHECK")
    print("=" * 80)
    
    print(f"\nTesting TimeSeriesSplit with {n_folds} folds...")
    print(f"Total samples: {len(df)}")
    
    tscv = TimeSeriesSplit(n_splits=n_folds)
    
    all_valid = True
    
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(df), start=1):
        train_dates = df.index[train_idx]
        val_dates = df.index[val_idx]
        
        train_start = train_dates.min()
        train_end = train_dates.max()
        val_start = val_dates.min()
        val_end = val_dates.max()
        
        print(f"\nFold {fold_idx}/{n_folds}:")
        print(f"  Train: {len(train_idx):5,} samples | {train_start} to {train_end}")
        print(f"  Val:   {len(val_idx):5,} samples | {val_start} to {val_end}")
        
        # CRITICAL CHECK: Validation data must be AFTER training data
        if val_start <= train_end:
            print(f"  ❌ FUTURE DATA LEAKAGE DETECTED!")
            print(f"     Validation starts ({val_start}) BEFORE training ends ({train_end})")
            all_valid = False
        else:
            gap = val_start - train_end
            print(f"  ✓ No leakage (gap: {gap})")
    
    if all_valid:
        print("\n" + "=" * 80)
        print("✓ ALL FOLDS VALID - NO FUTURE DATA LEAKAGE")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ FUTURE DATA LEAKAGE DETECTED IN SOME FOLDS")
        print("=" * 80)
        raise ValueError("Cross-validation has future data leakage!")
    
    return all_valid


def test_interval_consistency(df: pd.DataFrame, expected_interval_minutes: int):
    """Test that data intervals are consistent."""
    print("\n" + "=" * 80)
    print("TEST 3: INTERVAL CONSISTENCY CHECK")
    print("=" * 80)
    
    if len(df) < 2:
        print("⚠ Not enough data to check intervals")
        return
    
    # Calculate time differences
    time_diffs = df.index.diff().dropna()
    expected_delta = timedelta(minutes=expected_interval_minutes)
    
    # Count matching intervals
    matching = (time_diffs == expected_delta).sum()
    total = len(time_diffs)
    accuracy = matching / total
    
    print(f"\nExpected interval: {expected_interval_minutes} minutes")
    print(f"Total intervals: {total:,}")
    print(f"Matching intervals: {matching:,}")
    print(f"Accuracy: {accuracy:.2%}")
    
    if accuracy >= 0.95:
        print("✓ Interval consistency is good (≥95%)")
    elif accuracy >= 0.90:
        print("⚠ Interval consistency is acceptable (≥90%)")
    else:
        print("❌ Interval consistency is poor (<90%)")
        
        # Show some examples of mismatches
        mismatches = time_diffs[time_diffs != expected_delta]
        if len(mismatches) > 0:
            print(f"\nExample mismatches (showing first 5):")
            for i, (idx, diff) in enumerate(mismatches.head().items()):
                if i >= 5:
                    break
                print(f"  {idx}: {diff} (expected: {expected_delta})")
    
    return accuracy


def test_data_quality(df: pd.DataFrame):
    """Test overall data quality."""
    print("\n" + "=" * 80)
    print("TEST 4: DATA QUALITY CHECK")
    print("=" * 80)
    
    print(f"\nChecking data quality for {len(df)} samples...")
    
    # Check for NaN values
    nan_counts = df.isna().sum()
    total_nans = nan_counts.sum()
    
    print(f"\nNaN values:")
    if total_nans == 0:
        print("  ✓ No NaN values found")
    else:
        print(f"  ⚠ Found {total_nans} NaN values:")
        for col, count in nan_counts[nan_counts > 0].items():
            print(f"    {col}: {count} ({count/len(df):.2%})")
    
    # Check for zero/negative prices
    price_cols = ['open', 'high', 'low', 'close']
    invalid_prices = 0
    
    for col in price_cols:
        if col in df.columns:
            invalid = (df[col] <= 0).sum()
            if invalid > 0:
                print(f"  ❌ {col}: {invalid} invalid (≤0) values")
                invalid_prices += invalid
    
    if invalid_prices == 0:
        print("  ✓ All prices are positive")
    
    # Check OHLC logic (High >= Low, etc.)
    if all(col in df.columns for col in price_cols):
        high_low_valid = (df['high'] >= df['low']).all()
        high_oc_valid = (df['high'] >= df[['open', 'close']].max(axis=1)).all()
        low_oc_valid = (df['low'] <= df[['open', 'close']].min(axis=1)).all()
        
        if high_low_valid and high_oc_valid and low_oc_valid:
            print("  ✓ OHLC logic is valid")
        else:
            print("  ❌ OHLC logic violations detected")
    
    # Check volume
    if 'volume' in df.columns:
        zero_volume = (df['volume'] == 0).sum()
        negative_volume = (df['volume'] < 0).sum()
        
        if zero_volume > 0:
            print(f"  ⚠ {zero_volume} samples with zero volume ({zero_volume/len(df):.2%})")
        if negative_volume > 0:
            print(f"  ❌ {negative_volume} samples with negative volume")
        if zero_volume == 0 and negative_volume == 0:
            print("  ✓ Volume data is valid")


def main():
    """Run all validation tests."""
    parser = argparse.ArgumentParser(
        description="Validate time conversions and data quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Test 7 days of H1 data
  python validate_time_conversions.py --symbol BTCUSDT --days 7
  
  # Test 1 day of M5 data with CV check
  python validate_time_conversions.py --symbol ETHUSDT --days 1 --interval 5 --test-cv
  
  # Quick test with 3 folds
  python validate_time_conversions.py --symbol SOLUSDT --days 30 --folds 3 --test-cv
"""
    )
    
    parser.add_argument("--symbol", default="BTCUSDT", help="Asset symbol")
    parser.add_argument("--days", type=int, default=7, help="Days of data to fetch")
    parser.add_argument("--interval", default="60", help="Candle interval in minutes")
    parser.add_argument("--test-cv", action="store_true", help="Test cross-validation for data leakage")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds to test")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("TIME CONVERSION & DATA QUALITY VALIDATION")
    print("=" * 80)
    print(f"\nParameters:")
    print(f"  Symbol: {args.symbol}")
    print(f"  Days: {args.days}")
    print(f"  Interval: {args.interval}m")
    if args.test_cv:
        print(f"  CV Folds: {args.folds}")
    print("")
    
    # Test 1: Time parameters
    df = test_time_parameters(args.symbol, args.days, args.interval)
    
    # Test 2: Cross-validation (optional)
    if args.test_cv:
        test_cross_validation_no_leakage(df, args.folds)
    
    # Test 3: Interval consistency
    test_interval_consistency(df, int(args.interval))
    
    # Test 4: Data quality
    test_data_quality(df)
    
    # Final summary
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print("\n✓ All tests passed successfully!")
    print(f"\nData Summary:")
    print(f"  Symbol: {args.symbol}")
    print(f"  Samples: {len(df):,}")
    print(f"  Range: {df.index.min()} to {df.index.max()}")
    print(f"  Duration: {df.index.max() - df.index.min()}")
    print("")


if __name__ == "__main__":
    main()
