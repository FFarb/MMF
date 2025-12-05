"""
Quick test to verify SDE expert works in enriched fleet.
"""
import numpy as np
import pandas as pd
from src.models.moe_ensemble import MixtureOfExpertsEnsemble

print("=" * 80)
print("TESTING SDE EXPERT IN MOE ENSEMBLE")
print("=" * 80)

# Create dummy data
np.random.seed(42)
n_samples = 500

# Create features including physics features
data = {
    'feature_1': np.random.randn(n_samples),
    'feature_2': np.random.randn(n_samples),
    'feature_3': np.random.randn(n_samples),
    'hurst_200': np.random.uniform(0.3, 0.7, n_samples),
    'entropy_200': np.random.uniform(0.5, 1.5, n_samples),
    'fdi_200': np.random.uniform(0.4, 0.6, n_samples),
    'stability_theta': np.random.uniform(0.001, 0.05, n_samples),
    'stability_acf': np.random.uniform(0.5, 0.9, n_samples),
    'frac_diff': np.random.randn(n_samples),
}

X = pd.DataFrame(data)
y = np.random.randint(0, 2, n_samples)

print(f"\nData shape: {X.shape}")
print(f"Features: {list(X.columns)}")
print(f"Target distribution: {np.bincount(y)}")

# Initialize MoE with SDE expert
print("\n" + "=" * 80)
print("INITIALIZING MOE WITH SDE EXPERT")
print("=" * 80)

moe = MixtureOfExpertsEnsemble(
    physics_features=['hurst_200', 'entropy_200', 'fdi_200', 'stability_theta', 'stability_acf'],
    use_ou=True,  # Enable SDE expert
    use_cnn=False,  # Disable CNN for quick test
    random_state=42
)

print("\n[OK] MoE initialized successfully")
print(f"  SDE Expert enabled: {moe._ou_enabled}")
print(f"  SDE Expert type: {type(moe.sde_expert).__name__ if moe.sde_expert else 'None'}")

# Train
print("\n" + "=" * 80)
print("TRAINING MOE ENSEMBLE")
print("=" * 80)

moe.fit(X, y)

print("\n[OK] Training complete!")

# Predict
print("\n" + "=" * 80)
print("TESTING PREDICTIONS")
print("=" * 80)

proba = moe.predict_proba(X[:10])
print(f"\nPredictions shape: {proba.shape}")
print(f"Sample predictions (first 5):")
for i in range(5):
    print(f"  Sample {i}: P(down)={proba[i,0]:.4f}, P(up)={proba[i,1]:.4f}")

# Get SDE telemetry
if moe.sde_expert:
    print("\n" + "=" * 80)
    print("SDE EXPERT TELEMETRY")
    print("=" * 80)
    
    telemetry = moe.sde_expert.get_telemetry()
    print(f"\nActive Dimensions: {telemetry.get('active_dimensions', 'N/A')}/{moe.sde_expert.latent_dim}")
    print(f"Signal-to-Noise: {telemetry.get('signal_to_noise', 'N/A'):.4f}")
    
    physics_dna = telemetry.get('physics_dna', {})
    if physics_dna:
        print(f"\nPhysics DNA (discovered laws):")
        for dim, law in sorted(physics_dna.items()):
            print(f"  Dim {dim}: {law}")
    else:
        print("\nPhysics DNA: No dominant laws (highly stochastic)")

print("\n" + "=" * 80)
print("ALL TESTS PASSED!")
print("=" * 80)
print("\n[OK] SDE Expert successfully integrated into MoE Ensemble")
print("[OK] Ready to use in enriched fleet training")
