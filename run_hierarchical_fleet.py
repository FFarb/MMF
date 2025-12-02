"""
Hierarchical Fleet Training Pipeline.

This script implements hierarchical multi-asset training where:
1. Assets are automatically clustered by correlation structure
2. Market factor is extracted from dominant cluster (Majors)
3. Market factor is injected as feature to all clusters
4. Separate MoE models are trained per cluster

Problem:
--------
- Global training fails on Altcoins (different physics than BTC)
- Isolated training fails (Alts depend on BTC for context)

Solution:
---------
Hierarchical Training: "Majors lead, Alts follow (but have their own brain)"

- Majors (BTC, ETH): Learn pure trend dynamics
- Alts (SOL, ADA, etc.): Learn alt-specific moves + beta to Bitcoin

Result: Better performance on both Majors and Alts by respecting their different physics.
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
print("!!! HIERARCHICAL FLEET TRAINING !!!")
print(f"  USE_TENSOR_FLEX = {cfg.USE_TENSOR_FLEX}")
print(f"  TENSOR_FLEX_MODE = {cfg.TENSOR_FLEX_MODE}")
print(f"  TENSOR_FLEX_MIN_LATENTS = {cfg.TENSOR_FLEX_MIN_LATENTS}")
print("=" * 72)

from src.data_loader import MarketDataLoader
from src.features import SignalFactory
from src.features.tensor_flex import TensorFlexFeatureRefiner
from src.models.moe_ensemble import MixtureOfExpertsEnsemble
from src.preprocessing.frac_diff import FractionalDifferentiator
from src.analysis.cluster_engine import AssetClusterer, MarketFactorExtractor

# Import from run_deep_research
from run_deep_research import (
    PHYSICS_COLUMNS,
    LABEL_LOOKAHEAD,
    LABEL_THRESHOLD,
)

# The Fleet: 10 major cryptocurrencies
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
    """
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


def run_hierarchical_fleet_training(
    assets: List[str] = FLEET_ASSETS,
    n_clusters: int = 3,
    n_folds: int = 5,
    history_days: int = 730,
    history_days_5min: int = 0,  # NEW: Days of 5-minute data (0 = disabled)
    max_frac_diff_d: float = 0.65,
    market_factor_method: str = 'pca',
):
    """
    Main pipeline for hierarchical multi-asset training.
    
    Parameters
    ----------
    assets : list
        List of asset symbols to train on
    n_clusters : int
        Number of clusters to form
    n_folds : int
        Number of cross-validation folds
    history_days : int
        Days of hourly history to load per asset (default: 730 = 2 years)
    history_days_5min : int
        Days of 5-minute history to load per asset (default: 0 = disabled)
        Recommended: 120-180 days for 5-minute data
    max_frac_diff_d : float
        Maximum fractional differentiation order (cap)
    market_factor_method : str
        Method for market factor extraction ('pca', 'mean', 'weighted_mean')
    """
    # Determine which interval to use
    if history_days_5min > 0:
        interval = "5"
        history = history_days_5min
        print(f"[Config] Using 5-minute data: {history} days")
    else:
        interval = "60"
        history = history_days
        print(f"[Config] Using hourly data: {history} days")
    
    print("=" * 72)
    print("HIERARCHICAL FLEET TRAINING")
    print(f"Training {len(assets)} assets with cluster-based hierarchy")
    print(f"Interval: {interval} minutes, History: {history} days")
    print("=" * 72)
    
    # Step 1: Load and Preprocess Data for All Assets
    print("\n" + "=" * 72)
    print("STEP 1: FLEET DATA LOADING & PREPROCESSING")
    print("=" * 72)
    
    asset_data = {}
    
    for asset_symbol in assets:
        print(f"\n[Fleet] Processing {asset_symbol}...")
        
        try:
            loader = MarketDataLoader(symbol=asset_symbol, interval=interval)
            factory = SignalFactory()
            
            df_raw = loader.get_data(days_back=history)
            
            if df_raw is None or len(df_raw) < 1000:
                print(f"  ⚠️  Insufficient data for {asset_symbol}, skipping...")
                continue
            
            print(f"  [Data] Loaded {len(df_raw)} candles")
            
            # Fractional Differentiation (capped at max_frac_diff_d)
            frac_diff = FractionalDifferentiator(window_size=2048)
            
            n_calib = min(500, int(len(df_raw) * 0.1))
            calib_series = df_raw['close'].iloc[:n_calib]
            
            optimal_d = frac_diff.find_min_d(calib_series, precision=0.05, verbose=False)
            
            # Cap fractional differentiation order
            if optimal_d > max_frac_diff_d:
                print(f"  [FracDiff] Capping d: {optimal_d:.3f} → {max_frac_diff_d:.3f}")
                optimal_d = max_frac_diff_d
            
            df_raw['frac_diff'] = frac_diff.transform(df_raw['close'], d=optimal_d)
            
            print(f"  [FracDiff] Optimal d: {optimal_d:.3f}")
            
            # Generate Features
            df_features = factory.generate_signals(df_raw)
            df_features['frac_diff'] = df_raw['frac_diff'].reindex(df_features.index)
            df_features['close'] = df_raw['close'].reindex(df_features.index)
            df_features['volume'] = df_raw['volume'].reindex(df_features.index)
            
            # Store in asset_data
            asset_data[asset_symbol] = df_features
            
            print(f"  ✓ {len(df_features)} samples, {df_features.shape[1]} features")
            
        except Exception as e:
            import traceback
            print(f"  ❌ Error processing {asset_symbol}: {e}")
            print(f"     {traceback.format_exc()}")
            continue
    
    if len(asset_data) == 0:
        raise RuntimeError("No assets loaded successfully")
    
    print(f"\n[Fleet] ✓ Loaded {len(asset_data)} assets: {', '.join(sorted(asset_data.keys()))}")
    
    # Step 2: Auto-Clustering
    print("\n" + "=" * 72)
    print("STEP 2: AUTOMATIC ASSET CLUSTERING")
    print("=" * 72)
    
    clusterer = AssetClusterer(
        n_clusters=n_clusters,
        method='ward',
        feature_column='frac_diff',
        min_overlap=100,
    )
    
    cluster_result = clusterer.fit(asset_data)
    
    # Step 3: Market Factor Extraction
    print("\n" + "=" * 72)
    print("STEP 3: MARKET FACTOR EXTRACTION FROM DOMINANT CLUSTER")
    print("=" * 72)
    
    dominant_symbols = cluster_result.cluster_members[cluster_result.dominant_cluster_id]
    
    factor_extractor = MarketFactorExtractor(
        method=market_factor_method,
        feature_column='frac_diff',
    )
    
    market_factor = factor_extractor.extract_factor(asset_data, dominant_symbols)
    
    # Step 4: Inject Market Factor into All Assets
    print("\n" + "=" * 72)
    print("STEP 4: MARKET FACTOR INJECTION")
    print("=" * 72)
    
    for asset_symbol, df in asset_data.items():
        # Align market factor to asset timestamps
        df_aligned = df.copy()
        
        # Reindex market factor to match asset timestamps
        market_factor_aligned = market_factor.reindex(df_aligned.index, method='ffill')
        
        # Fill any remaining NaNs with 0 (neutral)
        market_factor_aligned = market_factor_aligned.fillna(0)
        
        df_aligned['market_factor'] = market_factor_aligned
        
        asset_data[asset_symbol] = df_aligned
        
        n_valid = (~market_factor_aligned.isna()).sum()
        print(f"  {asset_symbol:12s}: {n_valid}/{len(df_aligned)} timestamps with market factor")
    
    print(f"\n  ✓ Market factor injected into all {len(asset_data)} assets")
    
    # Step 5: Prepare Data for Training (Per-Cluster)
    print("\n" + "=" * 72)
    print("STEP 5: CLUSTER-BASED TRAINING DATA PREPARATION")
    print("=" * 72)
    
    cluster_datasets = {}
    
    for cluster_id, members in cluster_result.cluster_members.items():
        print(f"\n[Cluster {cluster_id}] Preparing data for: {', '.join(sorted(members))}")
        
        cluster_dfs = []
        
        for asset_symbol in members:
            if asset_symbol not in asset_data:
                continue
            
            df = asset_data[asset_symbol].copy()
            
            # Z-Score Normalization (per asset)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            exclude_cols = ['close', 'volume', 'frac_diff', 'market_factor']
            norm_cols = [c for c in numeric_cols if c not in exclude_cols]
            
            for col in norm_cols:
                mean = df[col].mean()
                std = df[col].std()
                if std > 1e-8:
                    df[col] = (df[col] - mean) / std
            
            # Calculate Energy
            energy = calculate_energy(df)
            df['energy'] = energy
            
            # Add Asset ID
            df['asset_id'] = asset_symbol
            
            # Build Labels
            y = build_labels(df)
            df['target'] = y
            
            # Remove NaNs
            valid_mask = ~y.isna() & ~df['frac_diff'].isna() & ~df['market_factor'].isna()
            df_clean = df.loc[valid_mask].copy()
            
            if len(df_clean) < 100:
                print(f"    ⚠️  {asset_symbol}: Insufficient valid samples ({len(df_clean)}), skipping...")
                continue
            
            print(f"    {asset_symbol:12s}: {len(df_clean)} samples, {df_clean['target'].mean():.2%} positive")
            
            cluster_dfs.append(df_clean)
        
        if len(cluster_dfs) == 0:
            print(f"    ⚠️  No valid data for Cluster {cluster_id}")
            continue
        
        # Combine cluster data (Panel Data structure)
        cluster_df = pd.concat(cluster_dfs, ignore_index=False)
        
        # Sort by timestamp and asset_id (Panel Data structure)
        if 'timestamp' not in cluster_df.columns:
            cluster_df = cluster_df.reset_index()
            if 'index' in cluster_df.columns and cluster_df['index'].dtype == 'datetime64[ns]':
                cluster_df.rename(columns={'index': 'timestamp'}, inplace=True)
            elif isinstance(cluster_df.index, pd.DatetimeIndex):
                cluster_df['timestamp'] = cluster_df.index
        
        cluster_df.sort_values(by=['timestamp', 'asset_id'], inplace=True)
        cluster_df.reset_index(drop=True, inplace=True)
        
        print(f"    ✓ Cluster {cluster_id}: {len(cluster_df)} total samples (Panel Data structure)")
        
        cluster_datasets[cluster_id] = cluster_df
    
    # Step 6: Cluster-Based Cross-Validation Training
    print("\n" + "=" * 72)
    print(f"STEP 6: CLUSTER-BASED TRAINING ({n_folds} Folds per Cluster)")
    print("=" * 72)
    
    # Load CNN params
    cnn_params = None
    best_params_path = Path("artifacts/best_cnn_params.json")
    if best_params_path.exists():
        with open(best_params_path, "r") as f:
            cnn_params = json.load(f)
        print(f"\n[CNN] Loaded tuned params: {cnn_params}")
    
    all_results = []
    cluster_results = {}
    
    for cluster_id, cluster_df in cluster_datasets.items():
        print(f"\n{'=' * 72}")
        print(f"CLUSTER {cluster_id} TRAINING")
        print(f"{'=' * 72}")
        
        members = cluster_result.cluster_members[cluster_id]
        is_dominant = cluster_id == cluster_result.dominant_cluster_id
        
        print(f"Assets: {', '.join(sorted(members))}")
        print(f"Dominant: {'YES' if is_dominant else 'NO'}")
        print(f"Samples: {len(cluster_df)}")
        
        # Extract features and labels
        timestamp_col = cluster_df['timestamp'].copy()
        X = cluster_df.drop(columns=['target', 'close', 'volume', 'timestamp'], errors='ignore')
        y = cluster_df['target']
        energy_weights = cluster_df['energy'].values
        
        # Feature partitioning
        available_physics = [c for c in PHYSICS_COLUMNS if c in X.columns]
        stability_cols = ["stability_theta", "stability_acf", "stability_warning"]
        available_physics.extend([c for c in stability_cols if c in X.columns])
        
        passthrough_cols = available_physics + ["frac_diff", "market_factor", "asset_id", "energy"]
        tensor_feature_cols = [c for c in X.columns if c not in passthrough_cols]
        
        print(f"\n[Features] Total: {len(X.columns)}")
        print(f"[Features] Passthrough: {len(passthrough_cols)}")
        print(f"[Features] Tensor-Flex Candidates: {len(tensor_feature_cols)}")
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=n_folds)
        fold_results = []
        per_asset_results = {asset: [] for asset in members}
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
            print(f"\n{'─' * 72}")
            print(f"CLUSTER {cluster_id} - FOLD {fold_idx}/{n_folds}")
            print(f"{'─' * 72}")
            
            X_train_raw = X.iloc[train_idx]
            X_val_raw = X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]
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
            
            # Train MoE
            print(f"\n[Fold {fold_idx}] Cluster {cluster_id} MoE Training")
            
            sample_weights = create_physics_sample_weights(X_train, energy_train)
            
            moe = MixtureOfExpertsEnsemble(
                physics_features=available_physics,
                random_state=RANDOM_SEED,
                use_cnn=True,
                use_ou=True,
                use_asset_embedding=(len(members) > 1),  # Only if multiple assets
                cnn_params=cnn_params,
                cnn_epochs=15,
            )
            
            moe.fit(X_train, y_train, sample_weight=sample_weights)
            
            # Get predictions and expert telemetry
            y_pred_proba = moe.predict_proba(X_val)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # COMPREHENSIVE TELEMETRY: Get expert-level predictions
            expert_telemetry = {}
            try:
                # Get individual expert predictions if available
                if hasattr(moe, 'experts_'):
                    for expert_name, expert in moe.experts_.items():
                        try:
                            expert_proba = expert.predict_proba(X_val)[:, 1]
                            expert_telemetry[expert_name] = {
                                'predictions': expert_proba,
                                'mean_prob': float(expert_proba.mean()),
                                'std_prob': float(expert_proba.std()),
                            }
                        except Exception as e:
                            print(f"  [Warning] Could not get {expert_name} predictions: {e}")
                
                # Get gating weights if available
                if hasattr(moe, 'gating_network_') and hasattr(moe.gating_network_, 'predict_proba'):
                    try:
                        gating_weights = moe.gating_network_.predict_proba(X_val)
                        expert_telemetry['gating_weights'] = {
                            'mean_weights': gating_weights.mean(axis=0).tolist(),
                            'std_weights': gating_weights.std(axis=0).tolist(),
                        }
                    except Exception as e:
                        print(f"  [Warning] Could not get gating weights: {e}")
            except Exception as e:
                print(f"  [Warning] Expert telemetry collection failed: {e}")
            
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
            
            print(f"\n[Fold {fold_idx}] Cluster {cluster_id} Performance:")
            print(f"  Precision: {precision:.2%}, Recall: {recall:.2%}, F1: {f1:.2%}")
            print(f"  Accuracy: {accuracy:.2%}, Expectancy: {expectancy:.4f}")
            
            # Per-asset metrics
            asset_detailed_metrics = {}
            if len(members) > 1:
                print(f"\n[Fold {fold_idx}] Per-Asset Performance:")
                
                for asset in X_val['asset_id'].unique():
                    asset_mask = X_val['asset_id'] == asset
                    
                    y_val_asset = y_val[asset_mask]
                    y_pred_asset = y_pred[asset_mask]
                    y_proba_asset = y_pred_proba[asset_mask]
                    
                    tp_asset = ((y_pred_asset == 1) & (y_val_asset == 1)).sum()
                    fp_asset = ((y_pred_asset == 1) & (y_val_asset == 0)).sum()
                    fn_asset = ((y_pred_asset == 0) & (y_val_asset == 1)).sum()
                    tn_asset = ((y_pred_asset == 0) & (y_val_asset == 0)).sum()
                    
                    prec_asset = tp_asset / (tp_asset + fp_asset) if (tp_asset + fp_asset) > 0 else 0.0
                    rec_asset = tp_asset / (tp_asset + fn_asset) if (tp_asset + fn_asset) > 0 else 0.0
                    f1_asset = 2 * (prec_asset * rec_asset) / (prec_asset + rec_asset) if (prec_asset + rec_asset) > 0 else 0.0
                    acc_asset = (tp_asset + tn_asset) / (tp_asset + tn_asset + fp_asset + fn_asset) if (tp_asset + tn_asset + fp_asset + fn_asset) > 0 else 0.0
                    exp_asset = (prec_asset * TP_PCT) - ((1 - prec_asset) * SL_PCT)
                    
                    print(f"  {asset:12s}: Prec={prec_asset:.2%}, Rec={rec_asset:.2%}, F1={f1_asset:.2%}, Exp={exp_asset:.5f}")
                    
                    # Store detailed metrics
                    asset_detailed_metrics[asset] = {
                        'tp': int(tp_asset),
                        'fp': int(fp_asset),
                        'fn': int(fn_asset),
                        'tn': int(tn_asset),
                        'precision': float(prec_asset),
                        'recall': float(rec_asset),
                        'f1': float(f1_asset),
                        'accuracy': float(acc_asset),
                        'expectancy': float(exp_asset),
                        'mean_proba': float(y_proba_asset.mean()),
                        'std_proba': float(y_proba_asset.std()),
                        'n_samples': int(len(y_val_asset)),
                    }
                    
                    per_asset_results[asset].append({
                        'cluster_id': cluster_id,
                        'fold': fold_idx,
                        'precision': prec_asset,
                        'recall': rec_asset,
                        'f1': f1_asset,
                        'accuracy': acc_asset,
                        'expectancy': exp_asset,
                        'n_samples': int(len(y_val_asset)),
                    })
            
            fold_results.append({
                'cluster_id': cluster_id,
                'fold': fold_idx,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'expectancy': expectancy,
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'tn': int(tn),
                'n_samples': int(len(y_val)),
                'expert_telemetry': expert_telemetry,
                'asset_metrics': asset_detailed_metrics,
            })
            
            del X_train, X_val, y_train, y_val, moe
            gc.collect()
        
        # Cluster summary
        cluster_df_results = pd.DataFrame(fold_results)
        
        print(f"\n{'=' * 72}")
        print(f"CLUSTER {cluster_id} SUMMARY")
        print(f"{'=' * 72}")
        print(f"  Avg Precision: {cluster_df_results['precision'].mean():.2%}")
        print(f"  Avg Recall:    {cluster_df_results['recall'].mean():.2%}")
        print(f"  Avg Expectancy: {cluster_df_results['expectancy'].mean():.5f}")
        
        cluster_results[cluster_id] = {
            'members': members,
            'is_dominant': is_dominant,
            'avg_precision': cluster_df_results['precision'].mean(),
            'avg_recall': cluster_df_results['recall'].mean(),
            'avg_f1': cluster_df_results['f1'].mean(),
            'avg_accuracy': cluster_df_results['accuracy'].mean(),
            'avg_expectancy': cluster_df_results['expectancy'].mean(),
            'fold_details': fold_results,  # All fold-level metrics
        }
        
        all_results.extend(fold_results)
    
    # Step 7: Final Report
    print("\n" + "=" * 72)
    print("STEP 7: HIERARCHICAL FLEET REPORT")
    print("=" * 72)
    
    print("\n" + "─" * 72)
    print("CLUSTER PERFORMANCE MATRIX")
    print("─" * 72)
    
    for cluster_id, stats in cluster_results.items():
        is_dominant_str = " (DOMINANT)" if stats['is_dominant'] else ""
        print(f"\nCluster {cluster_id}{is_dominant_str}:")
        print(f"  Assets: {', '.join(sorted(stats['members']))}")
        print(f"  Precision:  {stats['avg_precision']:.2%}")
        print(f"  Recall:     {stats['avg_recall']:.2%}")
        print(f"  Expectancy: {stats['avg_expectancy']:.5f}")
    
    # Per-asset summary
    print("\n" + "─" * 72)
    print("PER-ASSET PERFORMANCE SUMMARY")
    print("─" * 72)
    
    asset_summary = []
    for asset, results in per_asset_results.items():
        if len(results) > 0:
            df_asset = pd.DataFrame(results)
            asset_summary.append({
                'Asset': asset,
                'Cluster': df_asset['cluster_id'].iloc[0],
                'Precision': df_asset['precision'].mean(),
                'Recall': df_asset['recall'].mean(),
                'Expectancy': df_asset['expectancy'].mean(),
                'Folds': len(results),
            })
    
    if len(asset_summary) > 0:
        summary_df = pd.DataFrame(asset_summary)
        summary_df = summary_df.sort_values('Expectancy', ascending=False)
        
        print(summary_df.to_string(index=False))
        
        # Save results
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        
        # Save CSV summary
        summary_df.to_csv(artifacts_dir / "hierarchical_fleet_results.csv", index=False)
        
        # Save comprehensive JSON telemetry
        telemetry_data = {
            'config': {
                'n_clusters': n_clusters,
                'n_folds': n_folds,
                'history_days': history_days,
                'history_days_5min': history_days_5min,
                'interval': interval,
                'max_frac_diff_d': max_frac_diff_d,
                'market_factor_method': market_factor_method,
                'assets': assets,
            },
            'cluster_assignments': {
                asset: int(cluster_id) 
                for asset, cluster_id in cluster_result.cluster_map.items()
            },
            'dominant_cluster_id': int(cluster_result.dominant_cluster_id),
            'cluster_results': {
                str(cid): {
                    'members': list(stats['members']),
                    'is_dominant': bool(stats['is_dominant']),
                    'avg_precision': float(stats['avg_precision']),
                    'avg_recall': float(stats['avg_recall']),
                    'avg_f1': float(stats['avg_f1']),
                    'avg_accuracy': float(stats['avg_accuracy']),
                    'avg_expectancy': float(stats['avg_expectancy']),
                    # Note: fold_details contains expert_telemetry with numpy arrays
                    # We'll save a simplified version
                    'n_folds': len(stats['fold_details']),
                }
                for cid, stats in cluster_results.items()
            },
            'per_asset_summary': summary_df.to_dict(orient='records'),
            'per_asset_detailed': {
                asset: results
                for asset, results in per_asset_results.items()
                if len(results) > 0
            },
        }
        
        with open(artifacts_dir / "hierarchical_fleet_telemetry.json", "w") as f:
            json.dump(telemetry_data, f, indent=2)
        
        print(f"\n[Artifacts] Results saved to:")
        print(f"  - {artifacts_dir / 'hierarchical_fleet_results.csv'}")
        print(f"  - {artifacts_dir / 'hierarchical_fleet_telemetry.json'}")
    
    # Success criteria
    print("\n" + "=" * 72)
    print("HIERARCHICAL FLEET VERIFICATION")
    print("=" * 72)
    
    if len(asset_summary) > 0:
        all_profitable = (summary_df['Expectancy'] > 0).all()
        avg_precision = summary_df['Precision'].mean()
        precision_pass = avg_precision > 0.55
        
        print(f"✓ All Assets Profitable: {'PASS' if all_profitable else 'FAIL'}")
        print(f"✓ Avg Precision > 55%:   {'PASS' if precision_pass else 'FAIL'} ({avg_precision:.2%})")
        
        # Check if Alts improved with market factor
        alt_clusters = [cid for cid, stats in cluster_results.items() if not stats['is_dominant']]
        if len(alt_clusters) > 0:
            alt_expectancy = np.mean([cluster_results[cid]['avg_expectancy'] for cid in alt_clusters])
            print(f"✓ Alt Clusters Expectancy: {alt_expectancy:.5f}")
        
        if all_profitable and precision_pass:
            print("\n🎯 HIERARCHICAL FLEET TRAINING SUCCESSFUL!")
            print("   Majors lead, Alts follow (with their own brain)")
            print("   Market factor injection enables context-aware trading")
        else:
            print("\n⚠️  HIERARCHICAL FLEET NEEDS TUNING")
    
    return cluster_results, summary_df if len(asset_summary) > 0 else None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Hierarchical Fleet Training Pipeline"
    )
    parser.add_argument("--clusters", type=int, default=3, help="Number of clusters")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--days", type=int, default=730, help="Days of hourly history per asset")
    parser.add_argument("--minutes", type=int, default=0, help="Days of 5-minute history (0=disabled, recommended: 120-180)")
    parser.add_argument("--max-d", type=float, default=0.65, help="Max frac diff order")
    parser.add_argument("--factor-method", type=str, default="pca", 
                       choices=['pca', 'mean', 'weighted_mean'],
                       help="Market factor extraction method")
    
    args = parser.parse_args()
    
    run_hierarchical_fleet_training(
        assets=FLEET_ASSETS,
        n_clusters=args.clusters,
        n_folds=args.folds,
        history_days=args.days,
        history_days_5min=args.minutes,
        max_frac_diff_d=args.max_d,
        market_factor_method=args.factor_method,
    )
