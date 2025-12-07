
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.data_loader import GlobalMarketDataset, make_global_loader
from src.models.deep_experts import GraphVisionary, TorchSklearnWrapper
from src.models.moe_ensemble import HybridTrendExpert
from src.config import N_ASSETS

def create_dummy_data(data_dir: Path, n_assets: int = 3, n_samples: int = 100):
    data_dir.mkdir(parents=True, exist_ok=True)
    start_date = pd.Timestamp("2023-01-01")
    timestamps = [start_date + pd.Timedelta(minutes=5*i) for i in range(n_samples)]
    
    file_paths = []
    for i in range(n_assets):
        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": np.random.rand(n_samples),
            "high": np.random.rand(n_samples),
            "low": np.random.rand(n_samples),
            "close": np.random.rand(n_samples),
            "volume": np.random.rand(n_samples),
            "feature_1": np.random.rand(n_samples),
            "feature_2": np.random.rand(n_samples),
        })
        path = data_dir / f"asset_{i}.parquet"
        df.to_parquet(path)
        file_paths.append(path)
    return file_paths

def test_data_loader():
    print("\n--- Testing GlobalMarketDataset ---")
    data_dir = Path("temp_verification_data")
    if data_dir.exists():
        shutil.rmtree(data_dir)
    
    file_paths = create_dummy_data(data_dir, n_assets=N_ASSETS, n_samples=100)
    
    dataset = GlobalMarketDataset(file_paths, sequence_length=10, features=["feature_1", "feature_2"])
    print(f"Dataset length: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")
    # Expected: (Seq_Len, N_Assets, N_Features)
    assert sample.shape == (10, N_ASSETS, 2)
    print("GlobalMarketDataset shape check passed!")
    
    loader = make_global_loader(data_dir, batch_size=4, sequence_length=10, features=["feature_1", "feature_2"])
    batch = next(iter(loader))
    print(f"Batch shape: {batch.shape}")
    # Expected: (Batch, Seq_Len, N_Assets, N_Features)
    assert batch.shape == (4, 10, N_ASSETS, 2)
    print("DataLoader batch shape check passed!")
    
    return batch

def test_graph_visionary(batch):
    print("\n--- Testing GraphVisionary ---")
    # Batch shape: (B, S, N, F)
    B, S, N, F = batch.shape
    
    model = GraphVisionary(n_features=F, n_assets=N, hidden_dim=16, n_heads=2)
    output = model(batch)
    print(f"Output shape: {output.shape}")
    # Expected: (B, N, 1)
    assert output.shape == (B, N, 1)
    print("GraphVisionary forward pass passed!")

def test_hybrid_expert():
    print("\n--- Testing HybridTrendExpert Integration ---")
    # Mock data for sklearn interface
    # HybridTrendExpert expects 2D input (Samples, Features)
    # But internally we hacked it to use GraphVisionary which expects 4D or reshaped 2D.
    
    # Let's simulate the input that HybridTrendExpert would receive.
    # If we have 100 samples, and N_ASSETS=11 (from config), and features=2.
    # Total features = 2 * 11? No, usually features are per asset.
    # If we are training a global model, we might pass a flattened version of the global state?
    # OR we pass (Samples, Features) where Samples = TimeSteps * Assets?
    
    # In `HybridTrendExpert.fit`, we added logic:
    # if X.size == expected_size: reshape...
    
    # Let's try passing a flattened 4D tensor.
    # (B, S, N, F) -> flattened
    B = 10
    S = 16
    N = N_ASSETS
    F = 5
    
    # We simulate (Batch * Assets, Seq * Features)
    # Total samples = B * N
    # Features per sample = S * F
    
    X_input = np.random.rand(B * N, S * F).astype(np.float32)
    y_input = np.random.randint(0, 2, size=(B * N,))
    
    expert = HybridTrendExpert()
    
    print(f"Testing fit with X shape {X_input.shape}...")
    try:
        expert.fit(X_input, y_input)
        print("HybridTrendExpert fit passed!")
    except Exception as e:
        print(f"HybridTrendExpert fit failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        batch = test_data_loader()
        test_graph_visionary(batch)
        test_hybrid_expert()
        print("\nAll verification tests passed!")
    except Exception as e:
        print(f"\nVerification failed: {e}")
        exit(1)
    finally:
        if Path("temp_verification_data").exists():
            shutil.rmtree("temp_verification_data")
