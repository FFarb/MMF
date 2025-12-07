"""
Integration Example: Adding SDE Expert to MoE Ensemble.

This script demonstrates how to integrate the LaP-SDE expert
into the existing Mixture of Experts ensemble.

The SDE expert specializes in:
- Uncertainty quantification
- Stochastic market regimes
- Adaptive dimensionality reduction
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_loader import load_multi_asset_data
from src.features.alpha_council import AlphaCouncil
from src.models.sde_expert import SDEExpert
from src.config import TRAIN_SPLIT, RANDOM_SEED


def main():
    """Demonstrate SDE expert integration."""
    
    print("=" * 80)
    print("SDE EXPERT INTEGRATION EXAMPLE")
    print("=" * 80)
    
    # 1. Load and prepare data
    print("\n[1/4] Loading data...")
    df = load_multi_asset_data()
    
    # Create target if needed
    if 'target' not in df.columns:
        df['target'] = (df['return_1h'] > 0).astype(int)
    
    df = df.dropna(subset=['target'])
    
    # Feature selection
    print("\n[2/4] Running Alpha Council...")
    council = AlphaCouncil()
    df_selected = council.fit_transform(df, df['target'].values)
    
    # Prepare features
    feature_cols = [col for col in df_selected.columns 
                   if col not in ['target', 'asset_id']]
    
    if 'asset_id' in df_selected.columns:
        feature_cols.append('asset_id')
        df_selected['asset_id'] = pd.Categorical(df_selected['asset_id']).codes
    
    X = df_selected[feature_cols]
    y = df_selected['target'].values
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=TRAIN_SPLIT, random_state=RANDOM_SEED, stratify=y
    )
    
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Test: {len(X_test):,} samples")
    print(f"  Features: {X.shape[1]}")
    
    # 2. Train SDE Expert
    print("\n[3/4] Training SDE Expert...")
    print("-" * 80)
    
    sde_expert = SDEExpert(
        latent_dim=64,
        hidden_dims=[512, 256, 128],
        lr=0.001,
        epochs=50,  # Reduced for demo
        beta_kl=1.0,
        lambda_sparse=0.01,
        time_steps=10,
        random_state=RANDOM_SEED
    )
    
    sde_expert.fit(X_train, y_train)
    
    # 3. Evaluate with Uncertainty
    print("\n[4/4] Evaluating with Uncertainty Quantification...")
    print("-" * 80)
    
    # Get predictions with uncertainty
    y_pred_proba, uncertainty = sde_expert.predict_with_uncertainty(X_test)
    y_pred = (y_pred_proba[:, 1] >= 0.5).astype(int)
    
    # Overall performance
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    
    print(f"\n  Overall Performance:")
    print(f"    ROC AUC: {auc:.4f}")
    print(f"    Mean Uncertainty: {uncertainty.mean():.4f}")
    
    # Performance by uncertainty quartile
    print(f"\n  Performance by Uncertainty Quartile:")
    
    quartiles = np.percentile(uncertainty, [25, 50, 75])
    
    for i, label in enumerate(['Q1 (Low σ)', 'Q2', 'Q3', 'Q4 (High σ)']):
        if i == 0:
            mask = uncertainty <= quartiles[0]
        elif i == 3:
            mask = uncertainty > quartiles[2]
        else:
            mask = (uncertainty > quartiles[i-1]) & (uncertainty <= quartiles[i])
        
        if mask.sum() > 0:
            acc = (y_pred[mask] == y_test[mask]).mean()
            n_samples = mask.sum()
            mean_unc = uncertainty[mask].mean()
            
            print(f"    {label}: Acc={acc:.3f}, N={n_samples:4d}, σ={mean_unc:.4f}")
    
    # Uncertainty-filtered trading
    print(f"\n  Uncertainty-Filtered Trading Strategy:")
    
    # Only trade on bottom quartile (most confident)
    confidence_threshold = quartiles[0]
    high_confidence_mask = uncertainty < confidence_threshold
    
    filtered_acc = (y_pred[high_confidence_mask] == y_test[high_confidence_mask]).mean()
    filtered_auc = roc_auc_score(y_test[high_confidence_mask], 
                                  y_pred_proba[high_confidence_mask, 1])
    
    print(f"    Confidence Threshold: σ < {confidence_threshold:.4f}")
    print(f"    Samples Traded: {high_confidence_mask.sum()} / {len(X_test)} "
          f"({100*high_confidence_mask.sum()/len(X_test):.1f}%)")
    print(f"    Filtered Accuracy: {filtered_acc:.4f}")
    print(f"    Filtered ROC AUC: {filtered_auc:.4f}")
    print(f"    Improvement: {(filtered_auc - auc):.4f}")
    
    # Telemetry
    print(f"\n  Model Telemetry:")
    telemetry = sde_expert.get_telemetry()
    
    print(f"    Active Dimensions: {telemetry['active_dimensions']} / 64")
    print(f"    Signal-to-Noise: {telemetry['signal_to_noise']:.4f}")
    
    if telemetry['physics_dna']:
        print(f"    Physics DNA:")
        for dim, law in sorted(telemetry['physics_dna'].items())[:5]:
            print(f"      Dim {dim}: {law}")
        if len(telemetry['physics_dna']) > 5:
            print(f"      ... and {len(telemetry['physics_dna']) - 5} more")
    
    # 4. Integration Pattern
    print("\n" + "=" * 80)
    print("INTEGRATION PATTERN FOR MoE ENSEMBLE")
    print("=" * 80)
    
    integration_code = """
# In src/models/moe_ensemble.py:

from src.models.sde_expert import SDEExpert

class MixtureOfExpertsEnsemble:
    def __init__(self, ...):
        # ... existing experts ...
        
        # Add SDE Expert (Stochastic Specialist)
        self.experts['sde'] = SDEExpert(
            latent_dim=64,
            hidden_dims=[512, 256, 128],
            epochs=100,
            beta_kl=1.0,
            lambda_sparse=0.01,
            random_state=self.random_state
        )
    
    def fit(self, X, y, sample_weight=None):
        # ... train other experts ...
        
        # Train SDE expert
        print("  [6/6] Training SDE Expert (Stochastic)...")
        self.experts['sde'].fit(X, y)
        
        # Get uncertainty for gating
        _, uncertainty = self.experts['sde'].predict_with_uncertainty(X)
        
        # Use uncertainty in gating network
        # Low uncertainty → Higher weight for SDE
        sde_weight = 1.0 / (1.0 + uncertainty)
        
        # ... update gating weights ...
    
    def predict_proba(self, X):
        # ... get predictions from all experts ...
        
        # Get SDE predictions with uncertainty
        sde_proba, sde_uncertainty = self.experts['sde'].predict_with_uncertainty(X)
        
        # Adjust SDE weight based on uncertainty
        sde_weight = 1.0 / (1.0 + sde_uncertainty)
        
        # Weighted ensemble
        # ... combine predictions ...
    """
    
    print(integration_code)
    
    print("\n" + "=" * 80)
    print("INTEGRATION COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. SDE expert provides uncertainty quantification")
    print("  2. Filter trades by uncertainty for higher precision")
    print("  3. Use uncertainty in gating network weights")
    print("  4. Monitor Latent Prism and Drift DNA for market insights")
    print("")


if __name__ == "__main__":
    main()
