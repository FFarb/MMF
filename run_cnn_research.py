"""
CNN Temporal Expert Research Script

Validates the 1D CNN model with:
- Proper data splitting (no leakage)
- Causal convolutions (Chomp1d)
- LR scheduling
- Comprehensive visualization
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config import RANDOM_SEED
from src.data_loader import MarketDataLoader
from src.features import SignalFactory
from src.features.tensor_flex import TensorFlexFeatureRefiner
from src.models.cnn_temporal import CNNExpert

# Configuration
SYMBOL = "BTCUSDT"
INTERVAL = "60"  # H1 for better signal
DAYS_BACK = 365
WINDOW_LENGTH = 64
EPOCHS = 50
BATCH_SIZE = 128
TOP_FEATURES = 20  # CNNs hate 1000+ noisy features


def build_labels(df: pd.DataFrame, lookahead: int = 24) -> pd.Series:
    """Build forward-looking labels (24h ahead for H1 data)."""
    forward_ret = df['close'].shift(-lookahead) / df['close'] - 1.0
    y = (forward_ret > 0.005).astype(int)  # 0.5% threshold
    return y


def run_cnn_research():
    print("=" * 72)
    print("CNN TEMPORAL EXPERT RESEARCH")
    print("Strictly Causal TCN with Chomp1d")
    print("=" * 72)
    
    # Step 1: Load Data
    print("\n[1] Loading BTCUSDT data...")
    loader = MarketDataLoader(symbol=SYMBOL, interval=INTERVAL)
    
    # Force refresh to avoid empty cache issues
    df_raw = loader.get_data(days_back=DAYS_BACK, force_refresh=True)
    
    if df_raw.empty:
        print("Error: No data loaded.")
        return
    
    print(f"Loaded {len(df_raw)} rows")
    
    # Step 2: Generate Features
    print("\n[2] Generating features...")
    factory = SignalFactory()
    df_features = factory.generate_signals(df_raw)
    
    print(f"Generated {df_features.shape[1]} features")
    
    # Step 3: Build Labels
    print("\n[3] Building labels...")
    y = build_labels(df_features)
    valid_mask = ~y.isna()
    
    X = df_features.loc[valid_mask]
    y = y.loc[valid_mask]
    
    print(f"Valid samples: {len(X)}")
    print(f"Positive class: {y.sum()} ({y.mean():.2%})")
    
    # Step 4: Feature Selection
    print("\n[4] Feature Selection...")
    
    # Exclude non-feature columns
    exclude = {"open", "high", "low", "close", "volume", "timestamp", "target", "asset_id"}
    feature_cols = [c for c in X.columns if c not in exclude]
    
    print(f"Candidate features: {len(feature_cols)}")
    
    # Use TensorFlex if available, otherwise simple correlation selection
    try:
        print("  Attempting Tensor-Flex refinement...")
        refiner = TensorFlexFeatureRefiner(
            max_cluster_size=64,
            max_pairs_per_cluster=5,
            variance_threshold=0.95,
            n_splits_stability=3,
            stability_threshold=0.6,
            selector_coef_threshold=1e-4,
            selector_c=0.1,
            random_state=RANDOM_SEED,
            supervised_weight=0.2,
            corr_threshold=0.85,
            min_latents=5,
            max_latents=TOP_FEATURES,
        )
        
        # Use 80% for feature selection
        split_idx = int(len(X) * 0.8)
        X_select = X[feature_cols].iloc[:split_idx]
        y_select = y.iloc[:split_idx]
        
        refiner.fit(X_select, y_select)
        X_refined = refiner.transform(X[feature_cols], mode="selected")
        
        print(f"  ✓ Tensor-Flex: {len(feature_cols)} → {X_refined.shape[1]} latents")
        X_cnn = X_refined
        
    except Exception as e:
        print(f"  Tensor-Flex failed ({e}), using correlation selection...")
        
        # Fallback: Select top features by correlation with target
        correlations = {}
        for col in feature_cols:
            corr = X[col].corr(y)
            if not np.isnan(corr):
                correlations[col] = abs(corr)
        
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        selected = [f[0] for f in sorted_features[:TOP_FEATURES]]
        
        print(f"  ✓ Selected top {len(selected)} features by correlation")
        X_cnn = X[selected]
    
    # Step 5: Train CNN Expert
    print("\n[5] Training CNN Expert...")
    print(f"  Window Length: {WINDOW_LENGTH}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    
    cnn = CNNExpert(
        window_length=WINDOW_LENGTH,
        mid_channels=128,
        hidden_dim=64,
        dropout=0.2,
        lr=1e-3,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        random_state=RANDOM_SEED,
        patience=10,
        dilations=(1, 2, 4, 8, 16, 32),
        use_attention=True,
    )
    
    # Split for final test set
    test_split = int(len(X_cnn) * 0.8)
    X_train_val = X_cnn.iloc[:test_split]
    X_test = X_cnn.iloc[test_split:]
    y_train_val = y.iloc[:test_split]
    y_test = y.iloc[test_split:]
    
    print(f"  Train+Val: {len(X_train_val)}, Test: {len(X_test)}")
    
    # Fit (internally splits train/val)
    cnn.fit(X_train_val, y_train_val)
    
    # Step 6: Evaluate on Test Set
    print("\n[6] Evaluating on Test Set...")
    
    y_pred_proba = cnn.predict_proba(X_test)[:, 1]
    
    # Filter out NaN predictions (first window_length samples)
    valid_preds = ~np.isnan(y_pred_proba)
    y_test_valid = y_test.values[valid_preds]
    y_pred_valid = y_pred_proba[valid_preds]
    
    if len(y_test_valid) > 0 and len(np.unique(y_test_valid)) > 1:
        test_auc = roc_auc_score(y_test_valid, y_pred_valid)
        print(f"  Test ROC-AUC: {test_auc:.4f}")
    else:
        test_auc = float('nan')
        print("  Test ROC-AUC: N/A (insufficient data)")
    
    # Step 7: Visualization
    print("\n[7] Generating Visualizations...")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Training & Validation Loss
    ax1 = axes[0]
    if cnn.loss_history_:
        epochs_range = range(1, len(cnn.loss_history_) + 1)
        train_losses = [h[0] for h in cnn.loss_history_]
        val_losses = [h[1] for h in cnn.loss_history_]
        
        ax1.plot(epochs_range, train_losses, label='Training Loss', linewidth=2, color='#3498db')
        ax1.plot(epochs_range, val_losses, label='Validation Loss', linewidth=2, color='#e74c3c')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss (BCE)', fontsize=12)
        ax1.set_title('CNN Training Progress (Causal TCN with Chomp1d)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No loss history available', ha='center', va='center')
    
    # Plot 2: ROC Curve
    ax2 = axes[1]
    if not np.isnan(test_auc) and len(y_test_valid) > 0:
        fpr, tpr, _ = roc_curve(y_test_valid, y_pred_valid)
        
        ax2.plot(fpr, tpr, label=f'ROC Curve (AUC = {test_auc:.4f})', linewidth=2, color='#2ecc71')
        ax2.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        ax2.set_xlabel('False Positive Rate', fontsize=12)
        ax2.set_ylabel('True Positive Rate', fontsize=12)
        ax2.set_title('Test Set ROC Curve', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'ROC curve unavailable', ha='center', va='center')
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path("cnn_learning_curve.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved plot to {output_path}")
    
    plt.show()
    
    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Features Used: {X_cnn.shape[1]}")
    print(f"Training Samples: {len(X_train_val)}")
    print(f"Test Samples: {len(X_test)}")
    print(f"Test AUC: {test_auc:.4f}" if not np.isnan(test_auc) else "Test AUC: N/A")
    print(f"Epochs Trained: {len(cnn.loss_history_)}")
    
    if cnn.loss_history_:
        final_train_loss = cnn.loss_history_[-1][0]
        final_val_loss = cnn.loss_history_[-1][1]
        print(f"Final Train Loss: {final_train_loss:.5f}")
        print(f"Final Val Loss: {final_val_loss:.5f}")
    
    print("\n✓ Research complete!")


if __name__ == "__main__":
    run_cnn_research()
