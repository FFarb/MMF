import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_loader import MarketDataLoader
from src.features import SignalFactory
from src.models.moe_ensemble import MixtureOfExpertsEnsemble

def run_validation():
    print("=== Physics-Guided MoE Validation ===")
    
    # 1. Load Data
    print("\n[1] Loading Data...")
    try:
        loader = MarketDataLoader()
        # Fetch enough data for windows (504 + buffer)
        df = loader.get_data(days_back=100) 
        if df.empty:
            print("Error: No data loaded.")
            return
        print(f"Loaded {len(df)} rows.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Generate Signals
    print("\n[2] Generating Signals...")
    factory = SignalFactory()
    features = factory.generate_signals(df)
    print(f"Generated features shape: {features.shape}")
    
    # 3. Check Features
    print("\n[3] Checking Stability Features...")
    required = ["stability_theta", "stability_acf", "stability_warning"]
    missing = [c for c in required if c not in features.columns]
    if missing:
        print(f"FAILED: Missing columns {missing}")
        return
    else:
        print("SUCCESS: All stability features found.")
        print(features[required].describe())

    # 4. Train MoE
    print("\n[4] Training Mixture of Experts...")
    # Create a dummy target (Next day direction)
    features["target"] = (np.sign(features["close"].shift(-1) - features["close"]) + 1) // 2
    
    # Drop NaNs created by target shift
    train_df = features.dropna()
    
    # Select features (exclude non-feature columns)
    exclude = ["target", "timestamp", "open", "high", "low", "close", "volume", "asset_id"]
    X = train_df.drop(columns=exclude, errors="ignore")
    y = train_df["target"]
    
    print(f"Training on {len(X)} samples...")
    moe = MixtureOfExpertsEnsemble()
    moe.fit(X, y)
    print("Training complete.")
    
    # 5. Telemetry Check
    print("\n[5] Checking Telemetry & Correlations...")
    
    # Extract Physics Matrix
    physics_cols = list(moe.physics_features)
    print(f"Physics Features used: {physics_cols}")
    
    # Ensure columns exist
    missing_physics = [c for c in physics_cols if c not in X.columns]
    if missing_physics:
        print(f"Error: Missing physics columns in X: {missing_physics}")
        return

    physics_matrix = X[physics_cols].to_numpy()
    
    # Get Gating Weights
    weights = moe._gating_weights(physics_matrix)
    
    # Trend Expert is index 0
    trend_weights = weights[:, 0]
    stress_weights = weights[:, 2]
    theta = X["stability_theta"].values
    
    # Correlation
    corr_trend = np.corrcoef(theta, trend_weights)[0, 1]
    corr_stress = np.corrcoef(theta, stress_weights)[0, 1]
    
    print(f"Correlation (Theta vs Trend Weight):  {corr_trend:.4f}")
    print(f"Correlation (Theta vs Stress Weight): {corr_stress:.4f}")
    
    # Check Expectation: High Theta (Stable) -> High Trend Weight
    if corr_trend > 0:
        print("SUCCESS: Positive correlation with Trend Expert (Physics Gating working).")
    else:
        print("WARNING: Correlation with Trend Expert is not positive.")
        
    # Check Hard Gating (Low Theta -> High Stress)
    # Filter for low theta
    low_theta_mask = theta < 0.005
    if np.any(low_theta_mask):
        avg_stress_low_theta = np.mean(stress_weights[low_theta_mask])
        avg_stress_normal = np.mean(stress_weights[~low_theta_mask])
        print(f"Avg Stress Weight (Theta < 0.005): {avg_stress_low_theta:.4f}")
        print(f"Avg Stress Weight (Theta >= 0.005): {avg_stress_normal:.4f}")
        
        if avg_stress_low_theta > avg_stress_normal:
             print("SUCCESS: Hard Gating active (Stress Expert boosted in low theta).")
        else:
             print("WARNING: Hard Gating might not be active.")
    else:
        print("Note: No low theta samples found to verify hard gating.")

if __name__ == "__main__":
    run_validation()
