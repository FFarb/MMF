"""
Robust Training Pipeline with Time-Series Cross-Validation.

This script implements rigorous statistical validation for the MMF trading system:
- Time-Series Cross-Validation (No Look-Ahead Bias)
- Physics-Aware Sample Weighting (Ignore Chaos Periods)
- Bootstrap Confidence Intervals
- Expectancy-Based Evaluation (Profit over Accuracy)
- Three-Layer Brain Tensor-Flex (Marchenko-Pastur + Effective Rank + Sharpe Proxy)
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config import (
    DAYS_BACK,
    SYMBOLS,
    TP_PCT,
    SL_PCT,
    RANDOM_SEED,
)
import src.config as cfg

# --- CRITICAL CONFIG OVERRIDES ---
cfg.USE_TENSOR_FLEX = True
cfg.TENSOR_FLEX_MODE = "v2"
cfg.TENSOR_FLEX_MIN_LATENTS = 5
print(f"!!! FORCED CONFIG: USE_TENSOR_FLEX = {cfg.USE_TENSOR_FLEX} !!!")

from src.data_loader import MarketDataLoader
from src.features import SignalFactory
from src.features.tensor_flex import TensorFlexFeatureRefiner
from src.models.moe_ensemble import MixtureOfExpertsEnsemble

# Import data assembly logic from run_deep_research
from run_deep_research import (
    process_single_asset,
    cleanup_temp_dir,
    TEMP_DIR,
    PHYSICS_COLUMNS,
    LABEL_LOOKAHEAD,
    LABEL_THRESHOLD,
)


def build_labels(df: pd.DataFrame) -> pd.Series:
    """Build forward-looking labels."""
    if 'asset_id' in df.columns:
        forward_ret = df.groupby('asset_id')['close'].shift(-LABEL_LOOKAHEAD) / df['close'] - 1.0
    else:
        forward_ret = df['close'].shift(-LABEL_LOOKAHEAD) / df['close'] - 1.0
    y = (forward_ret > LABEL_THRESHOLD).astype(int)
    return y


def assemble_global_dataset(
    asset_list: List[str],
    loader: MarketDataLoader,
    factory: SignalFactory,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Assemble global multi-asset dataset using Scout & Fleet strategy.
    
    Returns
    -------
    df_global : pd.DataFrame
        Combined dataset with all assets
    feature_schema : List[str]
        List of feature column names
    """
    print("\n" + "=" * 72)
    print("STEP 1: DATA ASSEMBLY (Scout & Fleet)")
    print("=" * 72)
    
    cleanup_temp_dir()
    generated_files = []
    
    # Scout Phase: First asset defines the schema
    print(f"\n[Scout] Processing {asset_list[0]}...")
    df_scout = process_single_asset(asset_list[0], 0, loader, factory)
    
    if df_scout is None:
        raise RuntimeError("Scout asset failed to load.")
    
    # Define schema (exclude OHLCV, keep features + physics + asset_id)
    exclude = {"open", "high", "low", "close", "volume", "timestamp", "target"}
    feature_schema = [c for c in df_scout.columns if c not in exclude]
    
    print(f"[Scout] Schema defined: {len(feature_schema)} columns")
    
    # Save scout
    save_path = TEMP_DIR / f"{asset_list[0]}.parquet"
    df_scout[feature_schema + ['close']].to_parquet(save_path, compression='snappy')
    generated_files.append(save_path)
    
    del df_scout
    gc.collect()
    
    # Fleet Phase: Process remaining assets
    print(f"\n[Fleet] Processing {len(asset_list) - 1} remaining assets...")
    for asset_idx, symbol in enumerate(asset_list[1:], start=1):
        df_asset = process_single_asset(symbol, asset_idx, loader, factory)
        
        if df_asset is not None:
            # Ensure schema compatibility
            for col in feature_schema:
                if col not in df_asset.columns:
                    df_asset[col] = 0.0
            
            # Add close for label generation
            cols_to_save = feature_schema + ['close']
            cols_to_save = [c for c in cols_to_save if c in df_asset.columns]
            
            save_path = TEMP_DIR / f"{symbol}.parquet"
            df_asset[cols_to_save].to_parquet(save_path, compression='snappy')
            generated_files.append(save_path)
            print(f"  ✓ {symbol}")
        
        del df_asset
        gc.collect()
    
    # Global Assembly
    print(f"\n[Assembly] Merging {len(generated_files)} shards...")
    dfs = []
    for f in generated_files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"  Warning: Corrupt shard {f}: {e}")
    
    if not dfs:
        raise RuntimeError("No valid data shards.")
    
    df_global = pd.concat(dfs).sort_index()
    print(f"[Assembly] Global dataset: {df_global.shape}")
    
    return df_global, feature_schema


def create_physics_sample_weights(X: pd.DataFrame) -> np.ndarray:
    """
    Create sample weights based on stability warnings.
    
    Chaos periods (stability_warning == 1) get weight 0.0.
    Stable periods get weight 1.0.
    """
    if "stability_warning" not in X.columns:
        print("  [Warning] stability_warning not found. Using uniform weights.")
        return np.ones(len(X))
    
    warnings = X["stability_warning"].values
    weights = np.ones(len(X))
    weights[warnings == 1] = 0.0  # Zero out chaos periods
    
    n_chaos = (warnings == 1).sum()
    n_stable = (warnings == 0).sum()
    
    print(f"  [Physics Weighting] Stable: {n_stable}, Chaos (ignored): {n_chaos}")
    
    return weights


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_iterations: int = 50,
    random_state: int = 42,
) -> dict:
    """
    Calculate bootstrap confidence intervals for metrics.
    
    Returns
    -------
    dict with keys:
        - precision_mean, precision_5th, precision_95th
        - recall_mean, recall_5th, recall_95th
        - expectancy_mean, expectancy_5th, expectancy_95th
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)
    
    precisions = []
    recalls = []
    expectancies = []
    
    for _ in range(n_iterations):
        # Resample with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        y_t = y_true[indices]
        y_p = y_pred[indices]
        
        # Calculate metrics
        tp = ((y_p == 1) & (y_t == 1)).sum()
        fp = ((y_p == 1) & (y_t == 0)).sum()
        fn = ((y_p == 0) & (y_t == 1)).sum()
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Expectancy
        exp = (prec * TP_PCT) - ((1 - prec) * SL_PCT)
        
        precisions.append(prec)
        recalls.append(rec)
        expectancies.append(exp)
    
    return {
        "precision_mean": np.mean(precisions),
        "precision_5th": np.percentile(precisions, 5),
        "precision_95th": np.percentile(precisions, 95),
        "recall_mean": np.mean(recalls),
        "recall_5th": np.percentile(recalls, 5),
        "recall_95th": np.percentile(recalls, 95),
        "expectancy_mean": np.mean(expectancies),
        "expectancy_5th": np.percentile(expectancies, 5),
        "expectancy_95th": np.percentile(expectancies, 95),
    }


def run_robust_training(n_folds: int = 5):
    """
    Main training pipeline with rigorous validation.
    """
    print("=" * 72)
    print("ROBUST TRAINING PIPELINE")
    print("Physics-Guided Learning with Time-Series Cross-Validation")
    print("Three-Layer Brain: Marchenko-Pastur + Effective Rank + Sharpe Proxy")
    print("=" * 72)
    
    # Step 1: Data Assembly
    loader = MarketDataLoader(interval="5")
    factory = SignalFactory()
    
    df_global, feature_schema = assemble_global_dataset(SYMBOLS, loader, factory)
    
    # Build labels
    y_global = build_labels(df_global)
    valid_mask = ~y_global.isna()
    
    X = df_global.loc[valid_mask].drop(columns=['close'], errors='ignore')
    y = y_global.loc[valid_mask]
    
    print(f"\n[Labels] Valid samples: {len(X)} / {len(df_global)}")
    print(f"[Labels] Positive class: {y.sum()} ({y.mean():.2%})")
    
    # Check for stability features
    stability_features_present = all(
        col in X.columns for col in ["stability_theta", "stability_acf", "stability_warning"]
    )
    if stability_features_present:
        print("[✓] Stability Physics features detected")
    else:
        print("[!] Stability Physics features NOT found")
    
    # Step 2: Time-Series Cross-Validation
    print("\n" + "=" * 72)
    print(f"STEP 2: TIME-SERIES CROSS-VALIDATION ({n_folds} Folds)")
    print("=" * 72)
    
    tscv = TimeSeriesSplit(n_splits=n_folds)
    
    # Identify passthrough columns (physics + asset_id)
    available_physics = [c for c in PHYSICS_COLUMNS if c in X.columns]
    # Add stability features
    stability_cols = ["stability_theta", "stability_acf", "stability_warning"]
    available_physics.extend([c for c in stability_cols if c in X.columns])
    
    passthrough_cols = []
    if "asset_id" in X.columns:
        passthrough_cols.append("asset_id")
    passthrough_cols.extend(available_physics)
    
    tensor_feature_cols = [c for c in X.columns if c not in passthrough_cols]
    
    print(f"[Features] Total: {len(X.columns)}")
    print(f"[Features] Passthrough (Physics + ID): {len(passthrough_cols)}")
    print(f"[Features] Tensor-Flex Candidates: {len(tensor_feature_cols)}")
    
    # Safety Check: Ensure Tensor-Flex is ON
    if not cfg.USE_TENSOR_FLEX:
        print("WARNING: Tensor-Flex is OFF in config! Forcing it ON...")
        cfg.USE_TENSOR_FLEX = True
    
    # Results storage
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        print(f"\n{'─' * 72}")
        print(f"FOLD {fold_idx}/{n_folds}")
        print(f"{'─' * 72}")
        
        X_train_raw = X.iloc[train_idx]
        X_val_raw = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        
        print(f"Train: {len(X_train_raw)} samples | Val: {len(X_val_raw)} samples")
        
        # Step 3: Isolated Feature Refinement (CRITICAL)
        print(f"\n[Fold {fold_idx}] Step 3: Isolated Tensor-Flex Refinement")
        print("  🧠 Three-Layer Brain: Physics → Geometry → Economics")
        
        if cfg.USE_TENSOR_FLEX and tensor_feature_cols:
            print("  Fitting fresh Tensor-Flex on training fold only...")
            
            X_train_tensor = X_train_raw[tensor_feature_cols].copy()
            X_val_tensor = X_val_raw[tensor_feature_cols].copy()
            
            # Fresh refiner for this fold
            refiner = TensorFlexFeatureRefiner(
                max_cluster_size=cfg.TENSOR_FLEX_MAX_CLUSTER_SIZE,
                max_pairs_per_cluster=cfg.TENSOR_FLEX_MAX_PAIRS_PER_CLUSTER,
                variance_threshold=cfg.TENSOR_FLEX_VARIANCE_THRESHOLD,
                n_splits_stability=cfg.TENSOR_FLEX_N_SPLITS_STABILITY,
                stability_threshold=cfg.TENSOR_FLEX_STABILITY_THRESHOLD,
                selector_coef_threshold=cfg.TENSOR_FLEX_SELECTOR_COEF_THRESHOLD,
                selector_c=cfg.TENSOR_FLEX_SELECTOR_C,
                random_state=RANDOM_SEED + fold_idx,  # Different seed per fold
                supervised_weight=cfg.TENSOR_FLEX_SUPERVISED_WEIGHT,
                corr_threshold=cfg.TENSOR_FLEX_CORR_THRESHOLD,
                min_latents=cfg.TENSOR_FLEX_MIN_LATENTS,
                max_latents=cfg.TENSOR_FLEX_MAX_LATENTS,
            )
            
            refiner.fit(X_train_tensor, y_train)
            
            # Transform both sets
            X_train_tf = refiner.transform(X_train_tensor, mode="selected")
            X_val_tf = refiner.transform(X_val_tensor, mode="selected")
            
            # Combine with passthrough
            X_train = pd.concat([X_train_tf, X_train_raw[passthrough_cols]], axis=1)
            X_val = pd.concat([X_val_tf, X_val_raw[passthrough_cols]], axis=1)
            
            print(f"  ✓ Refined: {len(tensor_feature_cols)} → {X_train_tf.shape[1]} latent features")
            
            del refiner, X_train_tensor, X_val_tensor, X_train_tf, X_val_tf
            gc.collect()
        else:
            X_train = X_train_raw
            X_val = X_val_raw
            print("  Tensor-Flex disabled, using raw features.")
        
        # Step 4: Physics-Aware Training
        print(f"\n[Fold {fold_idx}] Step 4: Physics-Aware MoE Training")
        
        # Load best CNN params if available
        cnn_params = None
        best_params_path = Path("artifacts/best_cnn_params.json")
        if best_params_path.exists():
            import json
            with open(best_params_path, "r") as f:
                cnn_params = json.load(f)
            print(f"  Loaded tuned CNN params: {cnn_params}")
        else:
            print("  No tuned CNN params found, using defaults.")
        
        sample_weights = create_physics_sample_weights(X_train)
        
        moe = MixtureOfExpertsEnsemble(
            physics_features=available_physics,
            random_state=RANDOM_SEED,
            use_cnn=True,  # Enable CNN
            cnn_params=cnn_params,
            cnn_epochs=15, # Keep it fast for robust training loop
        )
        
        moe.fit(X_train, y_train, sample_weight=sample_weights)
        
        # Predict on validation
        y_pred_proba = moe.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Step 5: Bootstrap Validation
        print(f"\n[Fold {fold_idx}] Step 5: Bootstrap Validation (50 iterations)")
        
        boot_metrics = bootstrap_metrics(
            y_val.values,
            y_pred,
            n_iterations=50,
            random_state=RANDOM_SEED + fold_idx,
        )
        
        # Store results
        fold_results.append({
            "fold": fold_idx,
            "train_size": len(X_train),
            "val_size": len(X_val),
            "precision_mean": boot_metrics["precision_mean"],
            "precision_5th": boot_metrics["precision_5th"],
            "recall_mean": boot_metrics["recall_mean"],
            "expectancy_mean": boot_metrics["expectancy_mean"],
            "expectancy_5th": boot_metrics["expectancy_5th"],
        })
        
        print(f"  Precision: {boot_metrics['precision_mean']:.2%} "
              f"(5th: {boot_metrics['precision_5th']:.2%})")
        print(f"  Recall:    {boot_metrics['recall_mean']:.2%}")
        print(f"  Expectancy: {boot_metrics['expectancy_mean']:.4f} "
              f"(5th: {boot_metrics['expectancy_5th']:.4f})")
        
        del X_train, X_val, y_train, y_val, moe
        gc.collect()
    
    # Step 6: Reporting
    print("\n" + "=" * 72)
    print("STEP 6: FINAL REPORT")
    print("=" * 72)
    
    results_df = pd.DataFrame(fold_results)
    
    print("\nPer-Fold Results:")
    print(results_df.to_string(index=False))
    
    # Aggregate metrics
    avg_precision = results_df["precision_mean"].mean()
    avg_precision_5th = results_df["precision_5th"].mean()
    avg_expectancy = results_df["expectancy_mean"].mean()
    avg_expectancy_5th = results_df["expectancy_5th"].mean()
    
    print("\n" + "─" * 72)
    print("AGGREGATE METRICS (Across All Folds)")
    print("─" * 72)
    print(f"Average Precision:       {avg_precision:.2%}")
    print(f"Average Precision (5th): {avg_precision_5th:.2%}")
    print(f"Average Expectancy:      {avg_expectancy:.4f}")
    print(f"Average Expectancy (5th): {avg_expectancy_5th:.4f}")
    
    # Pass/Fail Criteria
    print("\n" + "=" * 72)
    print("PRODUCTION READINESS CHECK")
    print("=" * 72)
    
    precision_pass = avg_precision > 0.53
    expectancy_pass = avg_expectancy > 0.0
    
    print(f"✓ Precision > 53%:   {'PASS' if precision_pass else 'FAIL'} ({avg_precision:.2%})")
    print(f"✓ Expectancy > 0:    {'PASS' if expectancy_pass else 'FAIL'} ({avg_expectancy:.4f})")
    
    if precision_pass and expectancy_pass:
        print("\n🎯 MODEL IS PRODUCTION READY")
    else:
        print("\n⚠️  MODEL NEEDS IMPROVEMENT")
    
    # Save results
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    results_df.to_csv(artifacts_dir / "robust_training_results.csv", index=False)
    print(f"\n[Artifacts] Results saved to {artifacts_dir / 'robust_training_results.csv'}")
    
    cleanup_temp_dir()
    
    return results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Robust Training Pipeline with CV")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    args = parser.parse_args()
    
    run_robust_training(n_folds=args.folds)
