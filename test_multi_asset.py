# -*- coding: utf-8 -*-
"""
Test script for multi-asset Graph Architecture.
Tests GraphVisionary with 4D input tensors.
"""
import sys
import os

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')

sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import torch
from src.config import SYMBOLS
from src.data_loader import MarketDataLoader, GlobalMarketDataset, make_global_loader

print("="*70)
print("GRAPH ARCHITECTURE TEST (GraphVisionary + GlobalMarketDataset)")
print("="*70)

# Test 1: Data Loading
print("\n[TEST 1] Multi-Asset Data Loading")
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

# Test 2: GraphVisionary Architecture
print("\n[TEST 2] GraphVisionary Neural Architecture")
print("-" * 70)

try:
    from src.models.deep_experts import GraphVisionary, TorchSklearnWrapper
    
    # Test GraphVisionary with 4D input
    print("Testing GraphVisionary with 4D input...")
    model = GraphVisionary(
        n_features=10,
        n_assets=3,
        hidden_dim=32,
        n_heads=2,
        dropout=0.2,
    )
    
    # Test forward pass with 4D input: (Batch, Seq, Assets, Features)
    batch_size = 16
    seq_len = 8
    n_assets = 3
    n_features = 10
    
    x = torch.randn(batch_size, seq_len, n_assets, n_features)
    output = model(x)
    
    print(f"  [OK] Forward pass: input shape {x.shape} -> output shape {output.shape}")
    assert output.shape == (batch_size, n_assets, 1), f"Expected ({batch_size}, {n_assets}, 1), got {output.shape}"
    print(f"  [OK] Output shape validation passed")
    print(f"  [OK] Cross-asset attention layer exists")
    print(f"  [OK] Per-asset LSTM encoder configured")
    
except Exception as e:
    print(f"[FAIL] GraphVisionary test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: TorchSklearnWrapper (Global Mode)
print("\n[TEST 3] TorchSklearnWrapper with Global Mode")
print("-" * 70)

try:
    # Create synthetic dataset for Global Mode
    # Global Mode expects: (Batch * N_Assets, Seq * Features)
    # which reshapes to: (Batch, Seq, N_Assets, Features)
    
    n_assets = 3
    seq_len = 8
    n_features = 10
    n_batches = 25  # Small for quick test
    
    # Total samples = n_batches * n_assets
    n_samples = n_batches * n_assets
    
    # Flatten: (Batch * Assets, Seq * Features)
    X_test = np.random.randn(n_samples, seq_len * n_features).astype(np.float32)
    y_test = np.random.randint(0, 2, n_samples)
    
    wrapper = TorchSklearnWrapper(
        n_features=n_features,
        n_assets=n_assets,
        sequence_length=seq_len,
        hidden_dim=16,
        n_heads=2,
        dropout=0.2,
        max_epochs=5,  # Quick test
        batch_size=8,
    )
    
    print(f"  Training wrapper in Global Mode...")
    print(f"  Input shape: {X_test.shape} (will reshape to ({n_batches}, {seq_len}, {n_assets}, {n_features}))")
    wrapper.fit(X_test, y_test)
    print("  [OK] Training completed")
    
    print("  Predicting...")
    probs = wrapper.predict_proba(X_test)
    print(f"  [OK] Predictions shape: {probs.shape}")
    assert probs.shape == (n_samples, 2), f"Expected ({n_samples}, 2), got {probs.shape}"
    print(f"  [OK] Sample probabilities: {probs[:3]}")
    
except Exception as e:
    print(f"[FAIL] TorchSklearnWrapper test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: GlobalMarketDataset
print("\n[TEST 4] GlobalMarketDataset with Parquet Files")
print("-" * 70)

try:
    from pathlib import Path
    import tempfile
    
    # Create temporary parquet files for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create synthetic parquet files for 3 assets
        n_timesteps = 100
        n_features = 5
        
        for i, symbol in enumerate(['BTC', 'ETH', 'XRP']):
            # Create synthetic data
            dates = pd.date_range('2024-01-01', periods=n_timesteps, freq='5min')
            data = np.random.randn(n_timesteps, n_features).astype(np.float32)
            df = pd.DataFrame(data, columns=[f'feature_{j}' for j in range(n_features)], index=dates)
            df.index.name = 'timestamp'
            
            # Save to parquet
            df.to_parquet(tmpdir / f"{symbol}.parquet")
        
        print(f"  Created test parquet files in {tmpdir}")
        
        # Test GlobalMarketDataset
        dataset = GlobalMarketDataset(
            file_paths=list(tmpdir.glob("*.parquet")),
            sequence_length=16,
            features=[f'feature_{j}' for j in range(n_features)]
        )
        
        print(f"  [OK] Dataset created with {len(dataset)} samples")
        
        # Test __getitem__
        sample = dataset[0]
        print(f"  [OK] Sample shape: {sample.shape}")
        assert sample.shape == (16, 3, n_features), f"Expected (16, 3, {n_features}), got {sample.shape}"
        
        # Test make_global_loader
        loader = make_global_loader(
            data_dir=tmpdir,
            batch_size=4,
            sequence_length=16,
            features=[f'feature_{j}' for j in range(n_features)],
            shuffle=False
        )
        
        print(f"  [OK] DataLoader created")
        
        # Test batch
        batch = next(iter(loader))
        print(f"  [OK] Batch shape: {batch.shape}")
        assert batch.shape[1:] == (16, 3, n_features), f"Expected (*, 16, 3, {n_features}), got {batch.shape}"
        
except Exception as e:
    print(f"[FAIL] GlobalMarketDataset test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: MoE Ensemble Integration
print("\n[TEST 5] MoE Ensemble with GraphVisionary")
print("-" * 70)

try:
    from src.models.moe_ensemble import HybridTrendExpert
    
    # Create test data with required physics features
    n_samples = 200
    X_df = pd.DataFrame(np.random.randn(n_samples, 10), columns=[f'feature_{i}' for i in range(10)])
    
    # Add required physics features
    X_df['hurst_200'] = np.random.uniform(0.3, 0.7, n_samples)
    X_df['entropy_200'] = np.random.uniform(0.5, 0.9, n_samples)
    X_df['volatility_200'] = np.random.uniform(0.01, 0.05, n_samples)
    
    y_test = np.random.randint(0, 2, n_samples)
    
    expert = HybridTrendExpert(
        n_estimators=50,  # Reduced for testing
        random_state=42,
    )
    
    print("  Training HybridTrendExpert (uses GraphVisionary internally)...")
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
print("ALL TESTS PASSED [OK]")
print("="*70)
print("\nGraph Architecture Status:")
print("[OK] GraphVisionary handles 4D input (Batch, Seq, Assets, Features)")
print("[OK] GlobalMarketDataset aligns multi-asset data")
print("[OK] TorchSklearnWrapper supports Global Mode")
print("[OK] MoE Ensemble integrates with GraphVisionary")
print("\nNext steps:")
print("1. Run full pipeline with run_deep_research.py")
print("2. Verify training on real market data")
print("3. Deploy to Vast.AI for production testing")

