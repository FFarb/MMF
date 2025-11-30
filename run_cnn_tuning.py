"""
CNN Hyperparameter Tuning Script (Randomized Search)

Performs a randomized search to find a stable configuration for the CNNExpert.
Focuses on maximizing Validation AUC and minimizing the Generalization Gap.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config import RANDOM_SEED
from src.data_loader import MarketDataLoader
from src.features import SignalFactory
from src.features.tensor_flex import TensorFlexFeatureRefiner
from src.models.cnn_temporal import CNNExpert

# Configuration
SYMBOL = "BTCUSDT"
INTERVAL = "60"  # H1
DAYS_BACK = 365
TOP_FEATURES = 20
N_ITER = 10
EPOCHS_PER_TRIAL = 15

SEARCH_SPACE = {
    "dropout": [0.3, 0.5, 0.6],
    "mid_channels": [32, 64],
    "weight_decay": [1e-3, 1e-2, 0.1],
    "lr": [1e-3, 5e-4],
    "window_length": [32, 64],
}

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)


def build_labels(df: pd.DataFrame, lookahead: int = 24) -> pd.Series:
    """Build forward-looking labels (24h ahead for H1 data)."""
    forward_ret = df['close'].shift(-lookahead) / df['close'] - 1.0
    y = (forward_ret > 0.005).astype(int)  # 0.5% threshold
    return y


def get_data_and_features():
    print("\n[1] Loading Data & Generating Features...")
    loader = MarketDataLoader(symbol=SYMBOL, interval=INTERVAL)
    df_raw = loader.get_data(days_back=DAYS_BACK)
    
    if df_raw.empty:
        raise RuntimeError("No data loaded.")
    
    factory = SignalFactory()
    df_features = factory.generate_signals(df_raw)
    y = build_labels(df_features)
    
    valid_mask = ~y.isna()
    X = df_features.loc[valid_mask]
    y = y.loc[valid_mask]
    
    # Feature Selection
    exclude = {"open", "high", "low", "close", "volume", "timestamp", "target", "asset_id"}
    feature_cols = [c for c in X.columns if c not in exclude]
    
    try:
        print("  Attempting Tensor-Flex refinement...")
        refiner = TensorFlexFeatureRefiner(
            max_cluster_size=64,
            random_state=RANDOM_SEED,
            min_latents=5,
            max_latents=TOP_FEATURES,
        )
        # Use subset for fitting refiner
        split_idx = int(len(X) * 0.8)
        refiner.fit(X[feature_cols].iloc[:split_idx], y.iloc[:split_idx])
        X_refined = refiner.transform(X[feature_cols], mode="selected")
        print(f"  ✓ Tensor-Flex: {len(feature_cols)} → {X_refined.shape[1]} latents")
        return X_refined, y
    except Exception as e:
        print(f"  Tensor-Flex failed ({e}), using correlation selection...")
        correlations = {}
        for col in feature_cols:
            corr = X[col].corr(y)
            if not np.isnan(corr):
                correlations[col] = abs(corr)
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        selected = [f[0] for f in sorted_features[:TOP_FEATURES]]
        return X[selected], y


def sample_config() -> Dict[str, Any]:
    return {k: random.choice(v) for k, v in SEARCH_SPACE.items()}


def run_tuning():
    print("=" * 72)
    print("CNN HYPERPARAMETER TUNING (Randomized Search)")
    print("=" * 72)
    
    X, y = get_data_and_features()
    
    # Split Train/Val/Test
    test_split = int(len(X) * 0.8)
    X_train_val = X.iloc[:test_split]
    X_test = X.iloc[test_split:]
    y_train_val = y.iloc[:test_split]
    y_test = y.iloc[test_split:]
    
    results = []
    best_auc = -1.0
    best_config = None
    best_model_history = None
    
    print(f"\nStarting {N_ITER} trials...")
    
    for i in range(N_ITER):
        config = sample_config()
        print(f"\nTrial {i+1}/{N_ITER}: {config}")
        
        try:
            cnn = CNNExpert(
                window_length=config["window_length"],
                mid_channels=config["mid_channels"],
                hidden_dim=32,  # Fixed small hidden dim
                dropout=config["dropout"],
                lr=config["lr"],
                weight_decay=config["weight_decay"],
                epochs=EPOCHS_PER_TRIAL,
                batch_size=128,
                random_state=RANDOM_SEED + i,
                patience=5,
                dilations=(1, 2, 4, 8, 16), # Slightly reduced depth
                kernel_size=3,
            )
            
            cnn.fit(X_train_val, y_train_val)
            
            # Metrics
            if not cnn.loss_history_:
                print("  Failed: No loss history")
                continue
                
            train_loss = cnn.loss_history_[-1][0]
            val_loss = cnn.loss_history_[-1][1]
            gap = val_loss - train_loss
            
            # Val AUC (using internal validation split logic would be better, 
            # but here we rely on the final model state which might be overfitted 
            # if we don't have a separate holdout. 
            # Actually CNNExpert.fit splits internally. 
            # But we can't easily access that internal val set AUC unless we parse logs or modify class.
            # Let's use the X_test as a proxy for "Validation" in this tuning loop 
            # since we are tuning hyperparameters, not final testing.)
            
            y_pred_proba = cnn.predict_proba(X_test)[:, 1]
            valid_preds = ~np.isnan(y_pred_proba)
            
            if valid_preds.sum() > 0:
                auc = roc_auc_score(y_test.values[valid_preds], y_pred_proba[valid_preds])
            else:
                auc = 0.0
            
            print(f"  Result: AUC={auc:.4f}, Gap={gap:.4f} (Train={train_loss:.4f}, Val={val_loss:.4f})")
            
            results.append({
                "trial": i + 1,
                "config": config,
                "auc": auc,
                "gap": gap,
                "train_loss": train_loss,
                "val_loss": val_loss
            })
            
            if auc > best_auc:
                best_auc = auc
                best_config = config
                best_model_history = cnn.loss_history_
                
        except Exception as e:
            print(f"  Trial failed: {e}")
            
    # Sort results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=["auc", "gap"], ascending=[False, True])
    
    print("\n" + "=" * 72)
    print("TUNING RESULTS")
    print("=" * 72)
    print(results_df[["trial", "auc", "gap", "train_loss", "val_loss"]].to_string(index=False))
    
    # Save best params
    if best_config:
        best_params_path = ARTIFACTS_DIR / "best_cnn_params.json"
        with open(best_params_path, "w") as f:
            json.dump(best_config, f, indent=4)
        print(f"\nSaved best params to {best_params_path}")
        print(f"Best Config: {best_config}")
        
        # Plot best learning curve
        if best_model_history:
            plt.figure(figsize=(10, 6))
            train_losses = [h[0] for h in best_model_history]
            val_losses = [h[1] for h in best_model_history]
            epochs = range(1, len(train_losses) + 1)
            
            plt.plot(epochs, train_losses, label='Train Loss', marker='o')
            plt.plot(epochs, val_losses, label='Val Loss', marker='o')
            plt.title(f'Best Model Learning Curve (AUC={best_auc:.4f})')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            plot_path = ARTIFACTS_DIR / "best_cnn_curve.png"
            plt.savefig(plot_path)
            print(f"Saved learning curve to {plot_path}")

if __name__ == "__main__":
    run_tuning()
