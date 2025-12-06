"""
Individual Fleet Training: "Islands" Strategy.

Each asset gets its own isolated MoE model to prevent signal dilution.

Strategy:
- Train separate models for each asset (BTC, ETH, SOL, etc.)
- No clustering, no cross-asset contamination
- Each model learns asset-specific physics and patterns
- Apply forced daily entry policy to prevent zero-trade days

Author: QFC System v3.1 - Individual Asset Specialists
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
print("!!! INDIVIDUAL FLEET TRAINING (Islands Strategy) !!!")
print(f"  USE_TENSOR_FLEX = {cfg.USE_TENSOR_FLEX}")
print(f"  TENSOR_FLEX_MODE = {cfg.TENSOR_FLEX_MODE}")
print(f"  TENSOR_FLEX_MIN_LATENTS = {cfg.TENSOR_FLEX_MIN_LATENTS}")
print("=" * 72)

from src.data_loader import MarketDataLoader
from src.features import SignalFactory
from src.features.tensor_flex import TensorFlexFeatureRefiner
from src.models.moe_ensemble import MixtureOfExpertsEnsemble
from src.preprocessing.frac_diff import FractionalDifferentiator
from src.trading.forced_entry import (
    apply_forced_daily_entry,
    analyze_forced_entry_performance,
    print_forced_entry_report,
)

# Import from run_deep_research
from run_deep_research import (
    PHYSICS_COLUMNS,
    LABEL_LOOKAHEAD,
    LABEL_THRESHOLD,
)

# ==============================================================================
# 🔥 HOTFIX: TARGET HORIZON OVERRIDE
# ==============================================================================
print("=" * 72)
print("⚠️  TRAINING HORIZON OVERRIDE")
print("=" * 72)
print(f"  [CONFIG] Original Lookahead: {LABEL_LOOKAHEAD} bars")
print(f"  [CONFIG] Original Threshold: {LABEL_THRESHOLD}")
print("")
print("  [OVERRIDE] Switching to H1 Intraday Mode...")
LABEL_LOOKAHEAD = 4  # 4 hours (Intraday/Scalp)
LABEL_THRESHOLD = 0.0015  # 0.15% (Relaxed for more volume)
print(f"  [CONFIG] New Lookahead: {LABEL_LOOKAHEAD} bars (4 hours)")
print(f"  [CONFIG] New Threshold: {LABEL_THRESHOLD} (0.15% price move)")
print("=" * 72)

# The Fleet: Individual asset specialists
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
    """Calculate energy score for each sample."""
    if 'volume' not in df.columns:
        return np.ones(len(df))
    
    # Normalize volume
    volume = df['volume'].fillna(df['volume'].median())
    volume_mean = volume.mean()
    volume_std = volume.std()
    
    if volume_std > 1e-8:
        volume_norm = (volume - volume_mean) / volume_std
        volume_norm = np.clip(volume_norm, 0, 5)
    else:
        volume_norm = np.ones(len(df))
    
    # Normalize absolute price change
    price_change = df['close'].pct_change().abs().fillna(0)
    price_mean = price_change.mean()
    price_std = price_change.std()
    
    if price_std > 1e-8:
        price_norm = (price_change - price_mean) / price_std
        price_norm = np.clip(price_norm, 0, 5)
    else:
        price_norm = np.ones(len(df))
    
    # Energy = product
    energy = volume_norm * price_norm
    
    # Normalize to [0, 1]
    energy_min = energy.min()
    energy_max = energy.max()
    
    if energy_max - energy_min > 1e-8:
        energy = (energy - energy_min) / (energy_max - energy_min)
    else:
        energy = np.ones(len(df)) * 0.5
    
    energy = np.nan_to_num(energy, nan=0.5)
    
    if hasattr(energy, 'values'):
        return energy.values
    else:
        return np.asarray(energy)


def create_physics_sample_weights(X: pd.DataFrame, energy: np.ndarray = None) -> np.ndarray:
    """Create sample weights based on stability warnings and energy."""
    weights = np.ones(len(X))
    
    if "stability_warning" in X.columns:
        warnings = X["stability_warning"].values
        weights[warnings == 1] = 0.0
        
        n_chaos = (warnings == 1).sum()
        n_stable = (warnings == 0).sum()
        
        print(f"  [Physics Weighting] Stable: {n_stable}, Chaos (ignored): {n_chaos}")
    
    if energy is not None:
        energy_clean = np.nan_to_num(energy, nan=0.5)
        weights = weights * (1.0 + energy_clean)
    
    weights = np.nan_to_num(weights, nan=1.0)
    
    if weights.sum() < 1e-8:
        weights = np.ones(len(X))
    
    return weights


def run_individual_fleet_training(
    assets: List[str] = FLEET_ASSETS,
    n_folds: int = 5,
    history_days: int = 730,  # 2 years
    holdout_days: int = 0,
    max_frac_diff_d: float = 0.65,
    apply_forced_entry: bool = True,
):
    """
    Train individual MoE models for each asset (Islands Strategy).
    
    Parameters
    ----------
    assets : list
        List of asset symbols to train on
    n_folds : int
        Number of cross-validation folds
    history_days : int
        Days of history (default: 730 = 2 years)
    holdout_days : int
        Days to exclude from training for validation (default: 0)
    max_frac_diff_d : float
        Maximum fractional differentiation order
    apply_forced_entry : bool
        Apply forced daily entry policy (default: True)
    """
    print("=" * 72)
    print("INDIVIDUAL FLEET TRAINING: Islands Strategy")
    print(f"Training {len(assets)} assets with ISOLATED models")
    print(f"History: {history_days} days")
    if holdout_days > 0:
        print(f"Holdout Period: {holdout_days} days")
    print(f"Forced Daily Entry: {'ENABLED' if apply_forced_entry else 'DISABLED'}")
    print("=" * 72)
    
    # Load CNN params
    cnn_params = None
    best_params_path = Path("artifacts/best_cnn_params.json")
    if best_params_path.exists():
        with open(best_params_path, "r") as f:
            cnn_params = json.load(f)
        print(f"\n[CNN] Loaded tuned params: {cnn_params}")
    
    all_results = []
    
    for asset_symbol in assets:
        print("\n" + "=" * 72)
        print(f"ASSET: {asset_symbol} (Individual Model)")
        print("=" * 72)
        
        try:
            # Load data
            loader = MarketDataLoader(symbol=asset_symbol, interval="60")
            factory = SignalFactory()
            
            df_raw = loader.get_data(days_back=history_days)
            
            if df_raw is None or len(df_raw) < 1000:
                print(f"  ⚠️  Insufficient data for {asset_symbol}, skipping...")
                continue
            
            print(f"  [Data] Loaded {len(df_raw)} hourly candles")
            
            # Fractional Differentiation
            frac_diff = FractionalDifferentiator(window_size=2048)
            
            n_calib = min(500, int(len(df_raw) * 0.1))
            calib_series = df_raw['close'].iloc[:n_calib]
            
            optimal_d = frac_diff.find_min_d(calib_series, precision=0.05, verbose=False)
            
            if optimal_d > max_frac_diff_d:
                print(f"  [FracDiff] Capping d: {optimal_d:.3f} → {max_frac_diff_d:.3f}")
                optimal_d = max_frac_diff_d
            
            df_raw['frac_diff'] = frac_diff.transform(df_raw['close'], d=optimal_d)
            
            print(f"  [FracDiff] Optimal d: {optimal_d:.3f}")
            
            # Generate features
            df_features = factory.generate_signals(df_raw)
            df_features['frac_diff'] = df_raw['frac_diff'].reindex(df_features.index)
            df_features['close'] = df_raw['close'].reindex(df_features.index)
            df_features['volume'] = df_raw['volume'].reindex(df_features.index)
            
            # Apply holdout period if specified
            if holdout_days > 0:
                cutoff_date = df_features.index.max() - pd.Timedelta(days=holdout_days)
                df_train = df_features[df_features.index <= cutoff_date].copy()
                df_holdout = df_features[df_features.index > cutoff_date].copy()
                
                print(f"  [Holdout] Training: {len(df_train)} samples (up to {cutoff_date.date()})")
                print(f"  [Holdout] Held out: {len(df_holdout)} samples")
                
                holdout_file = Path(f"artifacts/individual_holdout_{asset_symbol}.csv")
                df_holdout.to_csv(holdout_file)
                print(f"  [Holdout] Saved to: {holdout_file}")
            else:
                df_train = df_features
            
            # Prepare data
            X = df_train.drop(columns=['close', 'volume'], errors='ignore')
            
            # Calculate Energy
            energy = calculate_energy(df_train)
            X['energy'] = energy
            
            # Build Labels
            y = build_labels(df_train)
            X['target'] = y
            
            # Remove NaNs
            valid_mask = ~y.isna() & ~X['frac_diff'].isna()
            X_clean = X.loc[valid_mask].copy()
            
            if len(X_clean) < 100:
                print(f"    ⚠️  Insufficient valid samples ({len(X_clean)}), skipping...")
                continue
            
            print(f"    {asset_symbol}: {len(X_clean)} samples, {X_clean['target'].mean():.2%} positive")
            
            # Extract features and labels
            timestamp_col = X_clean.index.copy()
            X_final = X_clean.drop(columns=['target'], errors='ignore')
            y_final = X_clean['target']
            energy_weights = X_clean['energy'].values
            
            # Feature partitioning
            available_physics = [c for c in PHYSICS_COLUMNS if c in X_final.columns]
            stability_cols = ["stability_theta", "stability_acf", "stability_warning"]
            available_physics.extend([c for c in stability_cols if c in X_final.columns])
            
            passthrough_cols = available_physics + ["frac_diff", "energy"]
            tensor_feature_cols = [c for c in X_final.columns if c not in passthrough_cols]
            
            print(f"\n[Features] Total: {len(X_final.columns)}")
            print(f"[Features] Passthrough: {len(passthrough_cols)}")
            print(f"[Features] Tensor-Flex Candidates: {len(tensor_feature_cols)}")
            
            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=n_folds)
            fold_results = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_final), start=1):
                print(f"\n{'─' * 72}")
                print(f"{asset_symbol} - FOLD {fold_idx}/{n_folds}")
                print(f"{'─' * 72}")
                
                X_train_raw = X_final.iloc[train_idx]
                X_val_raw = X_final.iloc[val_idx]
                y_train = y_final.iloc[train_idx]
                y_val = y_final.iloc[val_idx]
                energy_train = energy_weights[train_idx]
                
                print(f"Train: {len(X_train_raw)} samples | Val: {len(X_val_raw)} samples")
                
                # Tensor-Flex Refinement
                print(f"\n[Fold {fold_idx}] Tensor-Flex v2 Refinement")
                
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
                    
                    print(f"  ✓ Refined: {len(tensor_feature_cols)} → {X_train_tf.shape[1]} latents")
                    
                    del refiner, X_train_tensor, X_val_tensor, X_train_tf, X_val_tf
                    gc.collect()
                else:
                    X_train = X_train_raw
                    X_val = X_val_raw
                
                # Train MoE (Individual Model for this asset)
                print(f"\n[Fold {fold_idx}] {asset_symbol} MoE Training (Individual Model)")
                
                sample_weights = create_physics_sample_weights(X_train, energy_train)
                
                moe = MixtureOfExpertsEnsemble(
                    physics_features=available_physics,
                    random_state=RANDOM_SEED,
                    use_cnn=True,
                    use_ou=True,  # Enable SDE expert
                    use_asset_embedding=False,  # Single asset, no embedding needed
                    cnn_params=cnn_params,
                    cnn_epochs=15,
                )
                
                moe.fit(X_train, y_train, sample_weight=sample_weights)
                
                # Get predictions
                y_pred_proba = moe.predict_proba(X_val)[:, 1]
                
                # Use sniper threshold
                threshold = cfg.META_PROB_THRESHOLD if hasattr(cfg, 'META_PROB_THRESHOLD') else 0.55
                y_pred = (y_pred_proba >= threshold).astype(int)
                
                # Overall metrics
                tp = ((y_pred == 1) & (y_val == 1)).sum()
                fp = ((y_pred == 1) & (y_val == 0)).sum()
                fn = ((y_pred == 0) & (y_val == 1)).sum()
                tn = ((y_pred == 0) & (y_val == 0)).sum()
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
                expectancy = (precision * TP_PCT) - ((1 - precision) * SL_PCT)
                
                print(f"\n[Fold {fold_idx}] {asset_symbol} Performance:")
                print(f"  Precision: {precision:.2%}, Recall: {recall:.2%}, F1: {f1:.2%}")
                print(f"  Accuracy: {accuracy:.2%}, Expectancy: {expectancy:.4f}")
                
                fold_results.append({
                    'asset': asset_symbol,
                    'fold': fold_idx,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'accuracy': accuracy,
                    'expectancy': expectancy,
                    'y_pred_proba': y_pred_proba,
                    'y_pred': y_pred,
                    'timestamps': timestamp_col.iloc[val_idx],
                })
                
                del X_train, X_val, y_train, y_val, moe
                gc.collect()
            
            # Save predictions for this asset
            print(f"\n[Predictions] Saving predictions for {asset_symbol}...")
            
            asset_predictions = []
            for fold_res in fold_results:
                timestamps = fold_res['timestamps']
                probas = fold_res['y_pred_proba']
                preds = fold_res['y_pred']
                
                for ts, prob, pred in zip(timestamps, probas, preds):
                    asset_predictions.append({
                        'timestamp': ts,
                        'probability': prob,
                        'signal': 1 if pred == 1 else -1,
                        'fold': fold_res['fold'],
                    })
            
            if len(asset_predictions) > 0:
                pred_df = pd.DataFrame(asset_predictions)
                pred_df = pred_df.sort_values('timestamp')
                
                # Add close price
                asset_close = df_train['close'].reindex(pred_df['timestamp'])
                pred_df['close'] = asset_close.values
                
                # Apply Forced Daily Entry Policy
                if apply_forced_entry:
                    print(f"\n[ForcedEntry] Applying daily entry policy for {asset_symbol}...")
                    pred_df, forced_stats = apply_forced_daily_entry(
                        pred_df,
                        prob_threshold=threshold,
                    )
                    
                    print(f"  Natural trades: {forced_stats['natural_trades']}")
                    print(f"  Forced trades: {forced_stats['forced_trades']} ({forced_stats['forced_pct']:.1f}% of days)")
                
                pred_file = Path(f"artifacts/individual_predictions_{asset_symbol}.csv")
                pred_df.to_csv(pred_file, index=False)
                print(f"  ✓ Saved {len(pred_df)} predictions to {pred_file}")
            
            # Asset summary
            asset_df_results = pd.DataFrame(fold_results)
            
            print(f"\n{'=' * 72}")
            print(f"{asset_symbol} SUMMARY")
            print(f"{'=' * 72}")
            print(f"  Avg Precision: {asset_df_results['precision'].mean():.2%}")
            print(f"  Avg Recall:    {asset_df_results['recall'].mean():.2%}")
            print(f"  Avg Expectancy: {asset_df_results['expectancy'].mean():.5f}")
            
            all_results.append({
                'asset': asset_symbol,
                'avg_precision': asset_df_results['precision'].mean(),
                'avg_recall': asset_df_results['recall'].mean(),
                'avg_expectancy': asset_df_results['expectancy'].mean(),
                'folds': n_folds,
            })
            
        except Exception as e:
            import traceback
            print(f"  ❌ Error processing {asset_symbol}: {e}")
            print(f"     {traceback.format_exc()}")
            continue
    
    # Final Report
    print("\n" + "=" * 72)
    print("INDIVIDUAL FLEET REPORT")
    print("=" * 72)
    
    if len(all_results) > 0:
        summary_df = pd.DataFrame(all_results)
        summary_df = summary_df.sort_values('avg_expectancy', ascending=False)
        
        print("\n" + summary_df.to_string(index=False))
        
        # Save results
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        
        summary_df.to_csv(artifacts_dir / "individual_fleet_results.csv", index=False)
        
        print(f"\n[Artifacts] Results saved to:")
        print(f"  - {artifacts_dir / 'individual_fleet_results.csv'}")
        
        # Success criteria
        all_profitable = (summary_df['avg_expectancy'] > 0).all()
        avg_precision = summary_df['avg_precision'].mean()
        precision_pass = avg_precision > 0.55
        
        print("\n" + "=" * 72)
        print("INDIVIDUAL FLEET VERIFICATION")
        print("=" * 72)
        print(f"✓ All Assets Profitable: {'PASS' if all_profitable else 'FAIL'}")
        print(f"✓ Avg Precision > 55%:   {'PASS' if precision_pass else 'FAIL'} ({avg_precision:.2%})")
        
        if all_profitable and precision_pass:
            print("\n🎯 INDIVIDUAL FLEET TRAINING SUCCESSFUL!")
            print("   Islands Strategy: Each asset has its own specialized model!")
        else:
            print("\n⚠️  INDIVIDUAL FLEET NEEDS TUNING")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Individual Fleet Training: Islands Strategy"
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--days", type=int, default=730, help="Days of history")
    parser.add_argument("--holdout-days", type=int, default=0, help="Days to exclude from training")
    parser.add_argument("--max-d", type=float, default=0.65, help="Max frac diff order")
    parser.add_argument("--no-forced-entry", action="store_true", help="Disable forced daily entry policy")
    
    args = parser.parse_args()
    
    run_individual_fleet_training(
        assets=FLEET_ASSETS,
        n_folds=args.folds,
        history_days=args.days,
        holdout_days=args.holdout_days,
        max_frac_diff_d=args.max_d,
        apply_forced_entry=not args.no_forced_entry,
    )
