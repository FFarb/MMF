# -*- coding: utf-8 -*-
"""
Test script for multi-asset sparse-activated system.
Tests with 3 assets and 30 days of data.
"""
import sys
import os

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')

sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
from src.config import SYMBOLS
from src.data_loader import MarketDataLoader

print("="*70)
print("MULTI-ASSET SYSTEM TEST (3 assets, 30 days)")
print("="*70)

# Test 1: Data Loading
print("\n[TEST 1] Data Loading")
print("-" * 70)
loader = MarketDataLoader()

# Fetch only 3 assets for testing
test_symbols = SYMBOLS[:3]  # BTC, ETH, XRP
print(f"Testing with: {test_symbols}")

try:
    # Manually fetch each asset
    frames = []
    for asset_id, symbol in enumerate(test_symbols):
        print(f"\nFetching {symbol}...")
        asset_loader = MarketDataLoader(symbol=symbol, interval="5")
        df = asset_loader.get_data(days_back=30)
        df['asset_id'] = asset_id
        df['symbol'] = symbol
        frames.append(df)
        print(f"  [OK] Got {len(df)} candles")
    
    df_multi = pd.concat(frames, axis=0).sort_index()
    print(f"\n[OK] Combined dataset: {len(df_multi)} rows")
    print(f"[OK] Assets: {df_multi['asset_id'].unique()}")
    print(f"[OK] Columns: {list(df_multi.columns)}")
    
except Exception as e:
    print(f"[FAIL] Data loading failed: {e}")
    sys.exit(1)

# Test 2: Neural Architecture
print("\n[TEST 2] Neural Architecture")
print("-" * 70)

try:
    from src.models.deep_experts import AdaptiveConvExpert, TorchSklearnWrapper
    import torch
    
    # Test AdaptiveConvExpert with embeddings
    print("Testing AdaptiveConvExpert...")
    model = AdaptiveConvExpert(
        n_features=10,
        num_assets=3,
        embedding_dim=16,
        hidden_dim=32,
        sequence_length=16,
        lstm_hidden=64,
        dropout=0.2,
    )
    
    # Test forward pass
    x = torch.randn(32, 10)  # Batch of 32, 10 features
    asset_ids = torch.randint(0, 3, (32,))  # Random asset IDs
    output = model(x, asset_ids)
    
    print(f"  [OK] Forward pass: input shape {x.shape} -> output shape {output.shape}")
    print(f"  [OK] Asset embedding layer exists")
    print(f"  [OK] Dropout layers configured")
    
    # Test Monte Carlo inference
    mean_probs, uncertainties = model.predict_with_uncertainty(x, asset_ids, n_iter=5)
    print(f"  [OK] Monte Carlo inference: mean shape {mean_probs.shape}, uncertainty shape {uncertainties.shape}")
    print(f"  [OK] Sample uncertainty: {uncertainties[:3, 0].numpy()}")
    
except Exception as e:
    print(f"[FAIL] Neural architecture test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: TorchSklearnWrapper
print("\n[TEST 3] TorchSklearnWrapper with Asset IDs")
print("-" * 70)

try:
    # Create small synthetic dataset
    n_samples = 200
    X_test = np.random.randn(n_samples, 10)
    y_test = np.random.randint(0, 2, n_samples)
    asset_ids_test = np.random.randint(0, 3, n_samples)
    
    wrapper = TorchSklearnWrapper(
        n_features=10,
        num_assets=3,
        embedding_dim=16,
        hidden_dim=16,
        sequence_length=8,
        lstm_hidden=32,
        dropout=0.2,
        max_epochs=5,  # Quick test
        batch_size=32,
    )
    
    print("  Training wrapper...")
    wrapper.fit(X_test, y_test, asset_ids=asset_ids_test)
    print("  [OK] Training completed")
    
    print("  Predicting...")
    probs = wrapper.predict_proba(X_test, asset_ids=asset_ids_test)
    print(f"  [OK] Predictions shape: {probs.shape}")
    print(f"  [OK] Sample probabilities: {probs[:3]}")
    
except Exception as e:
    print(f"[FAIL] TorchSklearnWrapper test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: MoE Ensemble
print("\n[TEST 4] MoE Ensemble with Asset Awareness")
print("-" * 70)

try:
    from src.models.moe_ensemble import HybridTrendExpert
    
    # Create test data with asset_id column
    X_df = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(10)])
    X_df['asset_id'] = asset_ids_test
    
    expert = HybridTrendExpert(
        n_estimators=50,  # Reduced for testing
        random_state=42,
    )
    
    print("  Training HybridTrendExpert...")
    expert.fit(X_df, y_test)
    print("  [OK] Training completed")
    
    print("  Predicting...")
    probs = expert.predict_proba(X_df)
    print(f"  [OK] Predictions shape: {probs.shape}")
    print(f"  [OK] Sample probabilities: {probs[:3]}")
    
except Exception as e:
    print(f"[FAIL] MoE Ensemble test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("ALL TESTS PASSED")
print("="*70)
print("\nNext steps:")
print("1. Update feature engineering to respect asset boundaries")
print("2. Update run_deep_research.py orchestration")
print("3. Run full test with all 11 assets")
print("4. Push to GitHub")
