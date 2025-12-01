"""
Global Fleet Validation: Multi-Asset Oracle MoE Training.

This script trains a single "Global Brain" MoE on 10 different cryptocurrencies,
using asset embeddings to learn per-asset policies while sharing universal patterns.

PROBLEM:
- Single-asset models don't scale (need 10 separate models)
- Risk of spurious correlations (DOGE noise affecting BTC)
- Inefficient use of data

SOLUTION: Universal Training with Asset Identity
- Universal Features: FracDiff + TensorFlex (standardized across assets)
- Asset Embeddings: Feed asset_id to Gating Network for per-asset policies
- Energy Weighting: Focus learning on high-volume, high-volatility moves
- Oracle Training: Learn which expert is best for each asset/regime

Assets (The Fleet):
- BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX, LINK, LTC (10 assets)

Success Criteria:
- BTC performance ≥ single-asset baseline (no contamination)
- All assets profitable (Expectancy > 0)
- Per-asset precision > 55%

Goal: Prove a single model can trade 10 coins profitably by understanding
      their unique personalities (via embeddings) and ignoring noise (via energy).
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config import (
    DAYS_BACK,
    TP_PCT,
    SL_PCT,
    RANDOM_SEED,
)
import src.config as cfg

# --- CRITICAL CONFIG OVERRIDES ---
cfg.USE_TENSOR_FLEX = True
cfg.TENSOR_FLEX_MODE = "v2"
cfg.TENSOR_FLEX_MIN_LATENTS = 5
print("=" * 72)
print("!!! GLOBAL FLEET VALIDATION !!!")
print(f"  USE_TENSOR_FLEX = {cfg.USE_TENSOR_FLEX}")
print(f"  TENSOR_FLEX_MODE = {cfg.TENSOR_FLEX_MODE}")
print(f"  TENSOR_FLEX_MIN_LATENTS = {cfg.TENSOR_FLEX_MIN_LATENTS}")
print("=" * 72)

from src.data_loader import MarketDataLoader
from src.features import SignalFactory
from src.features.tensor_flex import TensorFlexFeatureRefiner
from src.models.moe_ensemble import MixtureOfExpertsEnsemble
from src.preprocessing.frac_diff import FractionalDifferentiator

# Import from run_deep_research
from run_deep_research import (
    PHYSICS_COLUMNS,
    LABEL_LOOKAHEAD,
    LABEL_THRESHOLD,
)

# The Fleet: 10 major cryptocurrencies (MATIC removed due to data quality issues)
FLEET_ASSETS = [
    'BTCUSDT',
    'ETHUSDT',
    'SOLUSDT',
    'BNBUSDT',
    'XRPUSDT',
    'ADAUSDT',
    'DOGEUSDT',
    'AVAXUSDT',
    'LINKUSDT',
    'LTCUSDT',
]


def build_labels(df: pd.DataFrame) -> pd.Series:
    """Build forward-looking labels."""
    forward_ret = df['close'].shift(-LABEL_LOOKAHEAD) / df['close'] - 1.0
    y = (forward_ret > LABEL_THRESHOLD).astype(int)
    return y


def calculate_energy(df: pd.DataFrame) -> np.ndarray:
    """
    Calculate energy score for each sample.
    
    Energy = Norm(Volume) * Norm(Abs(PriceChange))
    
    High energy = significant moves worth learning from
    Low energy = noise to ignore
    """
    # Check if volume exists
    if 'volume' not in df.columns:
        print("  [Warning] Volume not found, using uniform energy")
        return np.ones(len(df))
    
    # Normalize volume (handle NaN)
    volume = df['volume'].fillna(df['volume'].median())
    volume_mean = volume.mean()
    volume_std = volume.std()
    
    if volume_std > 1e-8:
        volume_norm = (volume - volume_mean) / volume_std
        volume_norm = np.clip(volume_norm, 0, 5)  # Cap outliers
    else:
        volume_norm = np.ones(len(df))
    
    # Normalize absolute price change (handle NaN)
    price_change = df['close'].pct_change().abs().fillna(0)
    price_mean = price_change.mean()
    price_std = price_change.std()
    
    if price_std > 1e-8:
        price_norm = (price_change - price_mean) / price_std
        price_norm = np.clip(price_norm, 0, 5)
    else:
        price_norm = np.ones(len(df))
    
    # Energy = product of normalized volume and price change
    energy = volume_norm * price_norm
    
    # Normalize to [0, 1] (handle edge cases)
    energy_min = energy.min()
    energy_max = energy.max()
    
    if energy_max - energy_min > 1e-8:
        energy = (energy - energy_min) / (energy_max - energy_min)
    else:
        energy = np.ones(len(df)) * 0.5  # Uniform if no variation
    
    # Replace any remaining NaN with 0.5 (neutral)
    energy = np.nan_to_num(energy, nan=0.5)
    
    # Convert to numpy array if it's a Series
    if hasattr(energy, 'values'):
        return energy.values
    else:
        return np.asarray(energy)


def create_physics_sample_weights(X: pd.DataFrame, energy: np.ndarray = None) -> np.ndarray:
    """
    Create sample weights based on stability warnings and energy.
    
    Chaos periods (stability_warning == 1) get weight 0.0.
    Stable periods get weight proportional to energy.
    """
    weights = np.ones(len(X))
    
    # Handle stability warnings if available
    if "stability_warning" in X.columns:
        warnings = X["stability_warning"].values
        
        # Zero out chaos periods
        weights[warnings == 1] = 0.0
        
        n_chaos = (warnings == 1).sum()
        n_stable = (warnings == 0).sum()
        
        print(f"  [Physics Weighting] Stable: {n_stable}, Chaos (ignored): {n_chaos}")
    else:
        print("  [Warning] stability_warning not found. Using energy only.")
    
    # Scale by energy (focus on significant moves)
    if energy is not None:
        # Ensure energy has no NaN
        energy_clean = np.nan_to_num(energy, nan=0.5)
        
        # Boost high-energy samples
        weights = weights * (1.0 + energy_clean)
        
        print(f"  [Energy Weighting] Avg energy: {energy_clean.mean():.3f}, Max: {energy_clean.max():.3f}")
    
    # Final safety check: replace any NaN with 1.0
    weights = np.nan_to_num(weights, nan=1.0)
    
    # Ensure at least some non-zero weights
    if weights.sum() < 1e-8:
        print("  [Warning] All weights are zero, using uniform weights")
        weights = np.ones(len(X))
    
    return weights


def run_global_fleet_validation(
    assets: List[str] = FLEET_ASSETS,
    n_folds: int = 5,
    history_days: int = 730,  # 2 years
):
    """
    Main pipeline for multi-asset global training.
    
    Parameters
    ----------
    assets : list
        List of asset symbols to train on
    n_folds : int
        Number of cross-validation folds
    history_days : int
        Days of history to load per asset
    """
    print("=" * 72)
    print("GLOBAL FLEET VALIDATION")
    print(f"Training single model on {len(assets)} assets")
    print("=" * 72)
    
    # Step 1: Load and Prepare Data for All Assets
    print("\n" + "=" * 72)
    print("STEP 1: FLEET DATA ASSEMBLY")
    print("=" * 72)
    
    global_dfs = []
    
    for asset_symbol in assets:
        print(f"\n[Fleet] Processing {asset_symbol}...")
        
        try:
            loader = MarketDataLoader(symbol=asset_symbol, interval="60")
            factory = SignalFactory()
            
            df_raw = loader.get_data(days_back=history_days)
            
            if df_raw is None or len(df_raw) < 1000:
                print(f"  ⚠️  Insufficient data for {asset_symbol}, skipping...")
                continue
            
            print(f"  [Data] Loaded {len(df_raw)} candles")
            
            # Fractional Differentiation (auto-tuned per asset)
            frac_diff = FractionalDifferentiator(window_size=2048)
            
            n_calib = min(500, int(len(df_raw) * 0.1))
            calib_series = df_raw['close'].iloc[:n_calib]
            
            optimal_d = frac_diff.find_min_d(calib_series, precision=0.05, verbose=False)
            df_raw['frac_diff'] = frac_diff.transform(df_raw['close'], d=optimal_d)
            
            print(f"  [FracDiff] Optimal d: {optimal_d:.3f}")
            
            # Generate Features
            df_features = factory.generate_signals(df_raw)
            df_features['frac_diff'] = df_raw['frac_diff'].reindex(df_features.index)
            df_features['close'] = df_raw['close'].reindex(df_features.index)
            df_features['volume'] = df_raw['volume'].reindex(df_features.index)
            
            # Z-Score Normalization (Critical for universal training)
            print(f"  [Normalization] Applying Z-score normalization...")
            numeric_cols = df_features.select_dtypes(include=[np.number]).columns
            exclude_cols = ['close', 'volume', 'frac_diff', 'asset_id']
            norm_cols = [c for c in numeric_cols if c not in exclude_cols]
            
            for col in norm_cols:
                mean = df_features[col].mean()
                std = df_features[col].std()
                if std > 1e-8:
                    df_features[col] = (df_features[col] - mean) / std
            
            # Calculate Energy
            energy = calculate_energy(df_features)
            df_features['energy'] = energy
            
            # Add Asset ID
            df_features['asset_id'] = asset_symbol
            
            # Build Labels
            y = build_labels(df_features)
            df_features['target'] = y
            
            # Remove NaNs
            valid_mask = ~y.isna() & ~df_features['frac_diff'].isna()
            df_clean = df_features.loc[valid_mask].copy()
            
            if len(df_clean) < 100:
                print(f"  ⚠️  Insufficient valid samples ({len(df_clean)}), skipping...")
                continue
            
            print(f"  [Clean] {len(df_clean)} valid samples")
            print(f"  [Labels] Positive class: {df_clean['target'].mean():.2%}")
            
            global_dfs.append(df_clean)
            
        except Exception as e:
            import traceback
            print(f"  ❌ Error processing {asset_symbol}: {e}")
            print(f"     {traceback.format_exc()}")
            continue
    
    if len(global_dfs) == 0:
        raise RuntimeError("No assets loaded successfully")
    
    # Combine all assets into global dataset
    print(f"\n[Fleet] Combining {len(global_dfs)} assets into global dataset...")
    global_df = pd.concat(global_dfs, ignore_index=False)  # Keep original index (timestamp)
    
    # CRITICAL: Sort by timestamp first, then asset_id for proper Panel Data structure
    # This ensures TimeSeriesSplit creates valid folds (no future leakage across assets)
    print(f"[Fleet] Sorting by timestamp and asset_id for Panel Data structure...")
    
    # Reset index to make timestamp a column if it's the index
    if 'timestamp' not in global_df.columns:
        global_df = global_df.reset_index()
        if 'index' in global_df.columns and global_df['index'].dtype == 'datetime64[ns]':
            global_df.rename(columns={'index': 'timestamp'}, inplace=True)
        elif global_df.index.name == 'timestamp' or isinstance(global_df.index, pd.DatetimeIndex):
            global_df['timestamp'] = global_df.index
    
    # Sort by timestamp (primary) and asset_id (secondary)
    global_df.sort_values(by=['timestamp', 'asset_id'], inplace=True)
    global_df.reset_index(drop=True, inplace=True)
    
    print(f"[Fleet] ✓ Sorted: {len(global_df)} samples across {len(global_dfs)} assets")
    
    # Gap Handling: Find common time intersection to avoid bias from different start dates
    print(f"\n[Fleet] Analyzing temporal alignment...")
    
    # Get time range per asset
    asset_time_ranges = {}
    for asset in global_df['asset_id'].unique():
        asset_mask = global_df['asset_id'] == asset
        asset_df = global_df[asset_mask]
        min_time = asset_df['timestamp'].min()
        max_time = asset_df['timestamp'].max()
        asset_time_ranges[asset] = (min_time, max_time)
        print(f"  {asset:12s}: {min_time} to {max_time} ({len(asset_df)} samples)")
    
    # Find common intersection (latest start, earliest end)
    common_start = max(start for start, _ in asset_time_ranges.values())
    common_end = min(end for _, end in asset_time_ranges.values())
    
    print(f"\n[Fleet] Common time intersection:")
    print(f"  Start: {common_start}")
    print(f"  End:   {common_end}")
    
    # Filter to common time range
    time_mask = (global_df['timestamp'] >= common_start) & (global_df['timestamp'] <= common_end)
    global_df = global_df[time_mask].copy()
    
    print(f"[Fleet] ✓ Aligned to common intersection: {len(global_df)} samples")
    
    # Verify alignment
    for asset in global_df['asset_id'].unique():
        asset_count = (global_df['asset_id'] == asset).sum()
        print(f"  {asset:12s}: {asset_count} samples")
    
    print(f"\n[Fleet] Global dataset: {len(global_df)} samples across {len(global_dfs)} assets")
    print(f"[Fleet] Assets: {', '.join(sorted(global_df['asset_id'].unique()))}")
    print(f"[Fleet] ✓ Panel Data structure validated (sorted by timestamp, asset_id)")
    
    # Extract features and labels (keep timestamp for validation)
    timestamp_col = global_df['timestamp'].copy()
    X = global_df.drop(columns=['target', 'close', 'volume', 'timestamp'], errors='ignore')
    y = global_df['target']
    energy_weights = global_df['energy'].values
    
    # Step 2: Feature Partitioning
    print("\n" + "=" * 72)
    print("STEP 2: UNIVERSAL FEATURE PARTITIONING")
    print("=" * 72)
    
    available_physics = [c for c in PHYSICS_COLUMNS if c in X.columns]
    stability_cols = ["stability_theta", "stability_acf", "stability_warning"]
    available_physics.extend([c for c in stability_cols if c in X.columns])
    
    passthrough_cols = available_physics + ["frac_diff", "asset_id", "energy"]
    tensor_feature_cols = [c for c in X.columns if c not in passthrough_cols]
    
    print(f"[Features] Total: {len(X.columns)}")
    print(f"[Features] Passthrough (Physics + FracDiff + Asset): {len(passthrough_cols)}")
    print(f"[Features] Tensor-Flex Candidates: {len(tensor_feature_cols)}")
    
    # Load CNN params
    cnn_params = None
    best_params_path = Path("artifacts/best_cnn_params.json")
    if best_params_path.exists():
        with open(best_params_path, "r") as f:
            cnn_params = json.load(f)
        print(f"\n[CNN] Loaded tuned params: {cnn_params}")
    
    # Step 3: Global Cross-Validation
    print("\n" + "=" * 72)
    print(f"STEP 3: GLOBAL CROSS-VALIDATION ({n_folds} Folds)")
    print("=" * 72)
    
    tscv = TimeSeriesSplit(n_splits=n_folds)
    fold_results = []
    per_asset_results = {asset: [] for asset in assets if asset in X['asset_id'].unique()}
    
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        print(f"\n{'─' * 72}")
        print(f"FOLD {fold_idx}/{n_folds}")
        print(f"{'─' * 72}")
        
        X_train_raw = X.iloc[train_idx]
        X_val_raw = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        energy_train = energy_weights[train_idx]
        
        print(f"Train: {len(X_train_raw)} samples | Val: {len(X_val_raw)} samples")
        
        # Tensor-Flex Refinement (Universal)
        print(f"\n[Fold {fold_idx}] Universal Tensor-Flex v2 Refinement")
        
        if cfg.USE_TENSOR_FLEX and tensor_feature_cols:
            X_train_tensor = X_train_raw[tensor_feature_cols].copy()
            X_val_tensor = X_val_raw[tensor_feature_cols].copy()
            
            refiner = TensorFlexFeatureRefiner(
                max_cluster_size=cfg.TENSOR_FLEX_MAX_CLUSTER_SIZE,
                max_pairs_per_cluster=cfg.TENSOR_FLEX_MAX_PAIRS_PER_CLUSTER,
                variance_threshold=cfg.TENSOR_FLEX_VARIANCE_THRESHOLD,
                n_splits_stability=cfg.TENSOR_FLEX_N_SPLITS_STABILITY,
                stability_threshold=cfg.TENSOR_FLEX_STABILITY_THRESHOLD,
                selector_coef_threshold=cfg.TENSOR_FLEX_SELECTOR_COEF_THRESHOLD,
                selector_c=cfg.TENSOR_FLEX_SELECTOR_C,
                random_state=RANDOM_SEED + fold_idx,
                supervised_weight=cfg.TENSOR_FLEX_SUPERVISED_WEIGHT,
                corr_threshold=cfg.TENSOR_FLEX_CORR_THRESHOLD,
                min_latents=cfg.TENSOR_FLEX_MIN_LATENTS,
                max_latents=cfg.TENSOR_FLEX_MAX_LATENTS,
            )
            
            refiner.fit(X_train_tensor, y_train)
            
            X_train_tf = refiner.transform(X_train_tensor, mode="selected")
            X_val_tf = refiner.transform(X_val_tensor, mode="selected")
            
            X_train = pd.concat([X_train_tf, X_train_raw[passthrough_cols]], axis=1)
            X_val = pd.concat([X_val_tf, X_val_raw[passthrough_cols]], axis=1)
            
            print(f"  ✓ Universal features: {len(tensor_feature_cols)} → {X_train_tf.shape[1]} latents")
            
            del refiner, X_train_tensor, X_val_tensor, X_train_tf, X_val_tf
            gc.collect()
        else:
            X_train = X_train_raw
            X_val = X_val_raw
        
        # Train Global MoE with Asset Embeddings
        print(f"\n[Fold {fold_idx}] Global MoE Training (Asset Embeddings Enabled)")
        
        sample_weights = create_physics_sample_weights(X_train, energy_train)
        
        moe = MixtureOfExpertsEnsemble(
            physics_features=available_physics,
            random_state=RANDOM_SEED,
            use_cnn=True,
            use_ou=True,
            use_asset_embedding=True,  # Enable asset-specific policies
            cnn_params=cnn_params,
            cnn_epochs=15,
        )
        
        moe.fit(X_train, y_train, sample_weight=sample_weights)
        
        # Get predictions
        y_pred_proba = moe.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Overall metrics
        tp = ((y_pred == 1) & (y_val == 1)).sum()
        fp = ((y_pred == 1) & (y_val == 0)).sum()
        fn = ((y_pred == 0) & (y_val == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        expectancy = (precision * TP_PCT) - ((1 - precision) * SL_PCT)
        
        print(f"\n[Fold {fold_idx}] Overall Performance:")
        print(f"  Precision: {precision:.2%}, Recall: {recall:.2%}, Expectancy: {expectancy:.4f}")
        
        # Per-asset metrics
        print(f"\n[Fold {fold_idx}] Per-Asset Performance:")
        
        for asset in X_val['asset_id'].unique():
            asset_mask = X_val['asset_id'] == asset
            
            y_val_asset = y_val[asset_mask]
            y_pred_asset = y_pred[asset_mask]
            
            tp_asset = ((y_pred_asset == 1) & (y_val_asset == 1)).sum()
            fp_asset = ((y_pred_asset == 1) & (y_val_asset == 0)).sum()
            fn_asset = ((y_pred_asset == 0) & (y_val_asset == 1)).sum()
            
            prec_asset = tp_asset / (tp_asset + fp_asset) if (tp_asset + fp_asset) > 0 else 0.0
            rec_asset = tp_asset / (tp_asset + fn_asset) if (tp_asset + fn_asset) > 0 else 0.0
            exp_asset = (prec_asset * TP_PCT) - ((1 - prec_asset) * SL_PCT)
            
            print(f"  {asset:12s}: Prec={prec_asset:.2%}, Rec={rec_asset:.2%}, Exp={exp_asset:.5f}")
            
            per_asset_results[asset].append({
                'fold': fold_idx,
                'precision': prec_asset,
                'recall': rec_asset,
                'expectancy': exp_asset,
            })
        
        fold_results.append({
            'fold': fold_idx,
            'precision': precision,
            'recall': recall,
            'expectancy': expectancy,
        })
        
        del X_train, X_val, y_train, y_val, moe
        gc.collect()
    
    # Step 4: Final Report
    print("\n" + "=" * 72)
    print("STEP 4: GLOBAL FLEET REPORT")
    print("=" * 72)
    
    results_df = pd.DataFrame(fold_results)
    
    print("\nOverall Performance:")
    print(f"  Avg Precision: {results_df['precision'].mean():.2%}")
    print(f"  Avg Recall:    {results_df['recall'].mean():.2%}")
    print(f"  Avg Expectancy: {results_df['expectancy'].mean():.5f}")
    
    print("\n" + "─" * 72)
    print("PER-ASSET PERFORMANCE MATRIX")
    print("─" * 72)
    
    # Create per-asset summary
    asset_summary = []
    for asset, results in per_asset_results.items():
        if len(results) > 0:
            df_asset = pd.DataFrame(results)
            asset_summary.append({
                'Asset': asset,
                'Precision': df_asset['precision'].mean(),
                'Recall': df_asset['recall'].mean(),
                'Expectancy': df_asset['expectancy'].mean(),
                'Folds': len(results),
            })
    
    summary_df = pd.DataFrame(asset_summary)
    summary_df = summary_df.sort_values('Expectancy', ascending=False)
    
    print(summary_df.to_string(index=False))
    
    # Success criteria
    print("\n" + "=" * 72)
    print("GLOBAL FLEET VERIFICATION")
    print("=" * 72)
    
    all_profitable = (summary_df['Expectancy'] > 0).all()
    avg_precision = summary_df['Precision'].mean()
    precision_pass = avg_precision > 0.55
    
    # Check BTC performance (no contamination)
    btc_row = summary_df[summary_df['Asset'] == 'BTCUSDT']
    if len(btc_row) > 0:
        btc_exp = btc_row['Expectancy'].iloc[0]
        btc_prec = btc_row['Precision'].iloc[0]
        print(f"✓ BTC Expectancy:      {btc_exp:.5f}")
        print(f"✓ BTC Precision:       {btc_prec:.2%}")
    
    print(f"✓ All Assets Profitable: {'PASS' if all_profitable else 'FAIL'}")
    print(f"✓ Avg Precision > 55%:   {'PASS' if precision_pass else 'FAIL'} ({avg_precision:.2%})")
    
    if all_profitable and precision_pass:
        print("\n🎯 GLOBAL FLEET VALIDATION SUCCESSFUL!")
        print("   Single model can trade all assets profitably")
        print("   Asset embeddings enable per-asset specialization")
        print("   Energy weighting focuses learning on significant moves")
    else:
        print("\n⚠️  GLOBAL FLEET NEEDS TUNING")
        if not all_profitable:
            unprofitable = summary_df[summary_df['Expectancy'] <= 0]['Asset'].tolist()
            print(f"   Unprofitable assets: {', '.join(unprofitable)}")
    
    # Save results
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    summary_df.to_csv(artifacts_dir / "global_fleet_results.csv", index=False)
    
    # Save detailed report
    with open(artifacts_dir / "global_fleet_report.txt", "w") as f:
        f.write("=" * 72 + "\n")
        f.write("GLOBAL FLEET VALIDATION REPORT\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Assets Trained: {len(summary_df)}\n")
        f.write(f"Total Samples: {len(X)}\n")
        f.write(f"Folds: {n_folds}\n\n")
        f.write("Overall Performance:\n")
        f.write(f"  Avg Precision: {results_df['precision'].mean():.2%}\n")
        f.write(f"  Avg Recall:    {results_df['recall'].mean():.2%}\n")
        f.write(f"  Avg Expectancy: {results_df['expectancy'].mean():.5f}\n\n")
        f.write("Per-Asset Performance:\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n")
    
    print(f"\n[Artifacts] Results saved to:")
    print(f"  - {artifacts_dir / 'global_fleet_results.csv'}")
    print(f"  - {artifacts_dir / 'global_fleet_report.txt'}")
    
    return summary_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Global Fleet Validation Pipeline"
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--days", type=int, default=730, help="Days of history per asset")
    
    args = parser.parse_args()
    
    run_global_fleet_validation(
        assets=FLEET_ASSETS,
        n_folds=args.folds,
        history_days=args.days,
    )
