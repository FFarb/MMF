import numpy as np
import pandas as pd
import shutil
from pathlib import Path
from src.features.tensor_flex import TensorFlexFeatureRefiner
import src.config as cfg

def test_tensor_flex_v2():
    print(">>> Running Tensor-Flex v2 Smoke Test...")
    
    # 1. Generate Synthetic Data
    rng = np.random.default_rng(42)
    n_samples = 500
    
    # Create 3 distinct clusters of features
    # Cluster 1: 10 features, highly correlated
    base1 = rng.normal(size=n_samples)
    c1 = pd.DataFrame({f"c1_{i}": base1 + 0.05 * rng.normal(size=n_samples) for i in range(10)})
    
    # Cluster 2: 10 features, highly correlated
    base2 = rng.normal(size=n_samples)
    c2 = pd.DataFrame({f"c2_{i}": base2 + 0.05 * rng.normal(size=n_samples) for i in range(10)})
    
    # Cluster 3: 10 features, highly correlated
    base3 = rng.normal(size=n_samples)
    c3 = pd.DataFrame({f"c3_{i}": base3 + 0.05 * rng.normal(size=n_samples) for i in range(10)})
    
    # Noise: 5 features
    noise = pd.DataFrame({f"noise_{i}": rng.normal(size=n_samples) for i in range(5)})
    
    X = pd.concat([c1, c2, c3, noise], axis=1)
    
    # Target: Depends on c1 and c2, but not c3 or noise
    y_signal = base1 + 0.5 * base2
    y = (y_signal > 0).astype(int)
    y = pd.Series(y, index=X.index)
    
    print(f"Data Shape: {X.shape}")
    
    # 2. Setup Refiner
    artifacts_dir = Path("temp_smoke_test_artifacts")
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)
    artifacts_dir.mkdir()
    
    # Force v2 mode via config patch (though we pass args explicitly too)
    cfg.TENSOR_FLEX_MODE = "v2"
    
    refiner = TensorFlexFeatureRefiner(
        max_cluster_size=8, # Force splitting of 10-feature clusters
        corr_threshold=0.85,
        min_latents=3,
        max_latents=10,
        supervised_weight=0.5,
        artifacts_dir=artifacts_dir
    )
    
    # 3. Fit
    print("Fitting refiner...")
    refiner.fit(X, y)
    
    # 4. Verify Clusters
    print(f"Clusters found: {len(refiner.clusters_)}")
    print(f"Cluster sizes: {[len(c) for c in refiner.clusters_]}")
    
    # We expect at least 3 clusters (c1, c2, c3) + noise features might be singletons or small clusters.
    # Since max_cluster_size=8 and c1 has 10 features, c1 should be split into at least 2 clusters.
    # Same for c2, c3.
    # So we expect > 3 clusters.
    assert len(refiner.clusters_) > 3, "Failed to split large clusters!"
    
    # 5. Verify Latents
    print(f"Selected Latents: {len(refiner.selected_feature_names_)}")
    print(f"Latent Names: {refiner.selected_feature_names_}")
    
    assert len(refiner.selected_feature_names_) >= 3, "Failed to select min_latents!"
    
    # 6. Verify Report
    report_path = artifacts_dir / "tensor_flex_report.json"
    assert report_path.exists(), "Report file not generated!"
    
    import json
    with open(report_path, "r") as f:
        report = json.load(f)
        
    print("Report Summary:")
    print(json.dumps(report["global_selector"], indent=2))
    
    # Check if stability scores are present
    latents_info = report["global_selector"]["latents"]
    assert len(latents_info) > 0
    assert "stability" in latents_info[0]
    assert "importance_score" in latents_info[0]
    
    print("\n>>> Smoke Test PASSED!")
    
    # Cleanup
    shutil.rmtree(artifacts_dir)

if __name__ == "__main__":
    test_tensor_flex_v2()
