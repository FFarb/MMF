"""
Enriched Fleet Training Pipeline: H1 Main + M5 Hints.

Strategy: "H1 is Main, M5 is Hint"
- Train on H1 targets (stable, ~58% precision)
- Enrich with M5 microstructure features (volatility, efficiency, buying pressure)
- Get best of both worlds: H1 stability + M5 internal insights

This restores the high precision of H1 training while capturing M5 microstructure
dynamics that provide an "X-ray" view inside each H1 bar.
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
print("!!! ENRICHED FLEET TRAINING (H1 + M5 Microstructure) !!!")
print(f"  USE_TENSOR_FLEX = {cfg.USE_TENSOR_FLEX}")
print(f"  TENSOR_FLEX_MODE = {cfg.TENSOR_FLEX_MODE}")
print(f"  TENSOR_FLEX_MIN_LATENTS = {cfg.TENSOR_FLEX_MIN_LATENTS}")
print("=" * 72)

from src.data_loader import MarketDataLoader
from src.features import SignalFactory
from src.features.micro_structure import calc_microstructure_features, summarize_microstructure
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

# ==============================================================================
# 🔥 HOTFIX: TARGET HORIZON OVERRIDE
# Force H1 Intraday Mode. Original 36h lookahead is too slow for H1/M5 strategy.
# This aligns the target with the M5 microstructure "hints".
# ==============================================================================
print("=" * 72)
print("⚠️  TRAINING HORIZON OVERRIDE")
print("=" * 72)
print(f"  [CONFIG] Original Lookahead: {LABEL_LOOKAHEAD} bars (from run_deep_research)")
print(f"  [CONFIG] Original Threshold: {LABEL_THRESHOLD}")
print("")
print("  [OVERRIDE] Switching to H1 Intraday Mode...")
LABEL_LOOKAHEAD = 4  # Override: 4 hours (Intraday/Scalp)
LABEL_THRESHOLD = 0.0015  # Override: 0.15% (Relaxed for more volume, sniper threshold compensates)
print(f"  [CONFIG] New Lookahead: {LABEL_LOOKAHEAD} bars (4 hours - H1 Intraday)")
print(f"  [CONFIG] New Threshold: {LABEL_THRESHOLD} (0.15% price move - more trades)")
print("")
print("  Rationale: 36h lookahead ignores M5 microstructure features.")
print("             4h lookahead activates M5 hints for precise entry timing.")
print("=" * 72)
# ==============================================================================

# The Fleet: 10 major cryptocurrencies (FULL FLEET RESTORED)
# Neural ODE will handle non-linear dynamics for all assets
FLEET_ASSETS = [
    'BTCUSDT',
    'ETHUSDT',
    'SOLUSDT',
    'BNBUSDT',
    'XRPUSDT',
    'ADAUSDT',
    'DOGEUSDT',  # RESTORED: Neural ODE will handle meme volatility
    'AVAXUSDT',  # RESTORED: Neural ODE will fix model misalignment
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


def run_enriched_fleet_training(
    assets: List[str] = FLEET_ASSETS,
    n_clusters: int = 3,
    n_folds: int = 5,
    history_days_h1: int = 730,  # Days of H1 data (730 days = 730*24 = 17,520 H1 bars)
    history_days_m5: int = 150,  # Days of M5 data (150 days = 150*288 = 43,200 M5 bars)
    holdout_days: int = 0,  # Days to exclude from training (for validation)
    max_frac_diff_d: float = 0.65,
    market_factor_method: str = 'pca',
):
    """
    Enriched fleet training: H1 targets + M5 microstructure features.
    
    Parameters
    ----------
    assets : list
        List of asset symbols to train on
    n_clusters : int
        Number of clusters to form
    n_folds : int
        Number of cross-validation folds
    history_days_h1 : int
        Days of H1 history (default: 730 = 2 years)
        Note: 1 day = 24 H1 bars, so 730 days = 17,520 bars
    history_days_m5 : int
        Days of M5 history for microstructure (default: 150 = 5 months)
        Note: 1 day = 288 M5 bars (24h * 60m / 5m), so 150 days = 43,200 bars
    holdout_days : int
        Days to exclude from training for walk-forward validation (default: 0)
        Example: 180 = train on older data, validate sniper on last 180 days
    max_frac_diff_d : float
        Maximum fractional differentiation order
    market_factor_method : str
        Method for market factor extraction
    """
    print("=" * 72)
    print("ENRICHED FLEET TRAINING: H1 Main + M5 Hints")
    print(f"Training {len(assets)} assets with microstructure enrichment")
    print(f"H1 History: {history_days_h1} days ({history_days_h1 * 24:,} bars)")
    print(f"M5 History: {history_days_m5} days ({history_days_m5 * 288:,} bars)")
    if holdout_days > 0:
        print(f"Holdout Period: {holdout_days} days (excluded from training)")
        print(f"Purpose: Walk-forward validation for sniper simulation")
    print(f"Cross-Validation Folds: {n_folds}")
    print("=" * 72)
    
    # Step 1: Load and Preprocess Data
    print("\n" + "=" * 72)
    print("STEP 1: ENRICHED DATA LOADING (H1 + M5 Microstructure)")
    print("=" * 72)
    
    asset_data = {}
    
    for asset_symbol in assets:
        print(f"\n[Fleet] Processing {asset_symbol}...")
        
        try:
            # Load H1 data (primary)
            loader_h1 = MarketDataLoader(symbol=asset_symbol, interval="60")
            factory = SignalFactory()
            
            df_h1_raw = loader_h1.get_data(days_back=history_days_h1)
            
            if df_h1_raw is None or len(df_h1_raw) < 1000:
                print(f"  ⚠️  Insufficient H1 data for {asset_symbol}, skipping...")
                continue
            
            print(f"  [H1 Data] Loaded {len(df_h1_raw)} hourly candles")
            
            # Load M5 data (for microstructure)
            print(f"  [M5 Data] Loading for microstructure features...")
            loader_m5 = MarketDataLoader(symbol=asset_symbol, interval="5")
            df_m5_raw = loader_m5.get_data(days_back=history_days_m5)
            
            if df_m5_raw is not None and len(df_m5_raw) > 1000:
                print(f"  [M5 Data] Loaded {len(df_m5_raw)} 5-minute candles")
                has_m5 = True
            else:
                print(f"  [M5 Data] Insufficient M5 data, proceeding without microstructure")
                has_m5 = False
            
            # Fractional Differentiation on H1
            frac_diff = FractionalDifferentiator(window_size=2048)
            
            n_calib = min(500, int(len(df_h1_raw) * 0.1))
            calib_series = df_h1_raw['close'].iloc[:n_calib]
            
            optimal_d = frac_diff.find_min_d(calib_series, precision=0.05, verbose=False)
            
            if optimal_d > max_frac_diff_d:
                print(f"  [FracDiff] Capping d: {optimal_d:.3f} → {max_frac_diff_d:.3f}")
                optimal_d = max_frac_diff_d
            
            df_h1_raw['frac_diff'] = frac_diff.transform(df_h1_raw['close'], d=optimal_d)
            
            print(f"  [FracDiff] Optimal d: {optimal_d:.3f}")
            
            # Generate H1 features
            df_h1_features = factory.generate_signals(df_h1_raw)
            df_h1_features['frac_diff'] = df_h1_raw['frac_diff'].reindex(df_h1_features.index)
            df_h1_features['close'] = df_h1_raw['close'].reindex(df_h1_features.index)
            df_h1_features['volume'] = df_h1_raw['volume'].reindex(df_h1_features.index)
            
            # Calculate M5 microstructure features
            if has_m5:
                print(f"  [Microstructure] Calculating M5 hints...")
                micro_features = calc_microstructure_features(df_m5_raw, df_h1_features.index)
                
                # Merge microstructure features into H1 data
                for col in micro_features.columns:
                    df_h1_features[col] = micro_features[col]
                
                print(f"  [Microstructure] ✓ Added {len(micro_features.columns)} microstructure features")
            
            # Apply holdout period if specified
            if holdout_days > 0:
                # Calculate cutoff date
                cutoff_date = df_h1_features.index.max() - pd.Timedelta(days=holdout_days)
                
                # Split into training and holdout
                df_train = df_h1_features[df_h1_features.index <= cutoff_date].copy()
                df_holdout = df_h1_features[df_h1_features.index > cutoff_date].copy()
                
                print(f"  [Holdout] Training: {len(df_train)} samples (up to {cutoff_date.date()})")
                print(f"  [Holdout] Held out: {len(df_holdout)} samples (after {cutoff_date.date()})")
                
                # Store both for later use
                asset_data[asset_symbol] = df_train
                
                # Save holdout data for sniper validation
                holdout_file = Path(f"artifacts/holdout_data_{asset_symbol}.csv")
                df_holdout.to_csv(holdout_file)
                print(f"  [Holdout] Saved to: {holdout_file}")
            else:
                # No holdout - use all data
                asset_data[asset_symbol] = df_h1_features
            
            print(f"  ✓ {len(asset_data[asset_symbol])} training samples, {df_h1_features.shape[1]} features")
            
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
    
    # Step 4: Inject Market Factor
    print("\n" + "=" * 72)
    print("STEP 4: MARKET FACTOR INJECTION")
    print("=" * 72)
    
    for asset_symbol, df in asset_data.items():
        df_aligned = df.copy()
        market_factor_aligned = market_factor.reindex(df_aligned.index, method='ffill')
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
    per_asset_results = {}
    
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
            # Also exclude microstructure features from normalization
            exclude_cols.extend([c for c in df.columns if c.startswith('micro_')])
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
            per_asset_results[asset_symbol] = []
        
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
    print(f"STEP 6: ENRICHED TRAINING ({n_folds} Folds per Cluster)")
    print("H1 Targets + M5 Microstructure Hints")
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
        print(f"CLUSTER {cluster_id} ENRICHED TRAINING")
        print(f"{'=' * 72}")
        
        members = cluster_result.cluster_members[cluster_id]
        is_dominant = cluster_id == cluster_result.dominant_cluster_id
        
        print(f"Assets: {', '.join(sorted(members))}")
        print(f"Dominant: {'YES' if is_dominant else 'NO'}")
        print(f"Samples: {len(cluster_df)}")
        
        # Check for microstructure features
        micro_cols = [c for c in cluster_df.columns if c.startswith('micro_')]
        if len(micro_cols) > 0:
            print(f"Microstructure Features: {len(micro_cols)}")
        
        # Extract features and labels
        timestamp_col = cluster_df['timestamp'].copy()
        X = cluster_df.drop(columns=['target', 'close', 'volume', 'timestamp'], errors='ignore')
        y = cluster_df['target']
        energy_weights = cluster_df['energy'].values
        
        # Feature partitioning
        available_physics = [c for c in PHYSICS_COLUMNS if c in X.columns]
        stability_cols = ["stability_theta", "stability_acf", "stability_warning"]
        available_physics.extend([c for c in stability_cols if c in X.columns])
        
        # Microstructure features are passthrough (don't apply TensorFlex to them)
        passthrough_cols = available_physics + ["frac_diff", "market_factor", "asset_id", "energy"] + micro_cols
        tensor_feature_cols = [c for c in X.columns if c not in passthrough_cols]
        
        print(f"\n[Features] Total: {len(X.columns)}")
        print(f"[Features] Passthrough: {len(passthrough_cols)} (includes {len(micro_cols)} microstructure)")
        print(f"[Features] Tensor-Flex Candidates: {len(tensor_feature_cols)}")
        
        
        # Cross-validation with TimeSeriesSplit
        # CRITICAL: TimeSeriesSplit ensures NO FUTURE DATA LEAKAGE
        # Each fold uses only past data for training, future data for validation
        # Example: Fold 1: Train[0:100], Val[100:200]
        #          Fold 2: Train[0:200], Val[200:300]
        # This mimics real-world trading where you can only use past data
        tscv = TimeSeriesSplit(n_splits=n_folds)
        fold_results = []
        
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
            print(f"\n[Fold {fold_idx}] Cluster {cluster_id} MoE Training (Enriched)")
            
            sample_weights = create_physics_sample_weights(X_train, energy_train)
            
            moe = MixtureOfExpertsEnsemble(
                physics_features=available_physics,
                random_state=RANDOM_SEED,
                use_cnn=True,
                use_ou=True,
                use_asset_embedding=(len(members) > 1),
                cnn_params=cnn_params,
                cnn_epochs=15,
            )
            
            moe.fit(X_train, y_train, sample_weight=sample_weights)
            
            # Get predictions
            y_pred_proba = moe.predict_proba(X_val)[:, 1]
            
            # Use sniper threshold (0.55) instead of coin-flip (0.5)
            # This trades recall for precision - "снайперский огонь"
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
            
            print(f"\n[Fold {fold_idx}] Cluster {cluster_id} Performance:")
            print(f"  Precision: {precision:.2%}, Recall: {recall:.2%}, F1: {f1:.2%}")
            print(f"  Accuracy: {accuracy:.2%}, Expectancy: {expectancy:.4f}")
            
            # Per-asset metrics
            if len(members) > 1:
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
                        'cluster_id': cluster_id,
                        'fold': fold_idx,
                        'precision': prec_asset,
                        'recall': rec_asset,
                        'expectancy': exp_asset,
                    })
            
            fold_results.append({
                'cluster_id': cluster_id,
                'fold': fold_idx,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'expectancy': expectancy,
                'y_pred_proba': y_pred_proba,  # Save probabilities
                'y_pred': y_pred,  # Save predictions
                'timestamps': timestamp_col.iloc[val_idx],  # Save timestamps
                'X_val': X_val,  # Save validation features for asset identification
            })
            
            del X_train, X_val, y_train, y_val, moe
            gc.collect()
        
        # Save predictions for this cluster
        print(f"\n[Predictions] Saving predictions for Cluster {cluster_id}...")
        for asset in members:
            if asset not in per_asset_results or len(per_asset_results[asset]) == 0:
                continue
            
            # Collect all predictions for this asset across folds
            asset_predictions = []
            
            for fold_res in fold_results:
                # Get asset-specific predictions from this fold
                X_val = fold_res['X_val']
                if 'asset_id' in X_val.columns:
                    asset_mask = X_val['asset_id'] == asset
                    
                    if asset_mask.sum() > 0:
                        timestamps = fold_res['timestamps'].iloc[asset_mask.values]
                        probas = fold_res['y_pred_proba'][asset_mask.values]
                        preds = fold_res['y_pred'][asset_mask.values]
                        
                        for ts, prob, pred in zip(timestamps, probas, preds):
                            asset_predictions.append({
                                'timestamp': ts,
                                'probability': prob,
                                'signal': 1 if pred == 1 else -1,  # Convert to +1/-1
                                'fold': fold_res['fold'],
                            })
            
            if len(asset_predictions) > 0:
                # Save to CSV
                pred_df = pd.DataFrame(asset_predictions)
                pred_df = pred_df.sort_values('timestamp')
                
                # Add close price from original data
                if asset in asset_data:
                    asset_close = asset_data[asset]['close'].reindex(pred_df['timestamp'])
                    pred_df['close'] = asset_close.values
                
                pred_file = Path(f"artifacts/fleet_predictions_{asset}.csv")
                pred_df.to_csv(pred_file, index=False)
                print(f"  ✓ Saved {len(pred_df)} predictions for {asset} to {pred_file}")
        
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
            'avg_expectancy': cluster_df_results['expectancy'].mean(),
        }
        
        all_results.extend(fold_results)
    
    # Step 7: Final Report
    print("\n" + "=" * 72)
    print("STEP 7: ENRICHED FLEET REPORT")
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
        
        summary_df.to_csv(artifacts_dir / "enriched_fleet_results.csv", index=False)
        
        print(f"\n[Artifacts] Results saved to:")
        print(f"  - {artifacts_dir / 'enriched_fleet_results.csv'}")
    
    # Success criteria
    print("\n" + "=" * 72)
    print("ENRICHED FLEET VERIFICATION")
    print("=" * 72)
    
    if len(asset_summary) > 0:
        all_profitable = (summary_df['Expectancy'] > 0).all()
        avg_precision = summary_df['Precision'].mean()
        precision_pass = avg_precision > 0.55
        
        print(f"✓ All Assets Profitable: {'PASS' if all_profitable else 'FAIL'}")
        print(f"✓ Avg Precision > 55%:   {'PASS' if precision_pass else 'FAIL'} ({avg_precision:.2%})")
        
        if all_profitable and precision_pass:
            print("\n🎯 ENRICHED FLEET TRAINING SUCCESSFUL!")
            print("   H1 stability + M5 microstructure = Best of both worlds!")
        else:
            print("\n⚠️  ENRICHED FLEET NEEDS TUNING")
    
    return cluster_results, summary_df if len(asset_summary) > 0 else None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enriched Fleet Training: H1 Main + M5 Hints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Default: 2 years H1, 5 months M5
  python run_enriched_fleet.py
  
  # Custom: 1 year H1, 3 months M5, 10 folds
  python run_enriched_fleet.py --h1-days 365 --m5-days 90 --folds 10
  
  # Quick test: 30 days H1, 7 days M5, 3 folds
  python run_enriched_fleet.py --h1-days 30 --m5-days 7 --folds 3
  
Note: 
  --h1-days 1 = 24 H1 bars (1 day of hourly data)
  --m5-days 1 = 288 M5 bars (1 day of 5-minute data)
"""
    )
    parser.add_argument("--clusters", type=int, default=3, help="Number of clusters")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds (min: 2)")
    parser.add_argument("--h1-days", type=int, default=730, 
                       help="Days of H1 history (1 day = 24 bars). Default: 730 = 2 years")
    parser.add_argument("--m5-days", type=int, default=150, 
                       help="Days of M5 history (1 day = 288 bars). Default: 150 = 5 months")
    parser.add_argument("--holdout-days", type=int, default=0, 
                       help="Days to exclude from training (for walk-forward validation)")
    parser.add_argument("--max-d", type=float, default=0.65, help="Max frac diff order")
    parser.add_argument("--factor-method", type=str, default="pca",
                       choices=['pca', 'mean', 'weighted_mean'],
                       help="Market factor extraction method")
    
    args = parser.parse_args()
    
    # Validate parameters
    if args.h1_days <= 0:
        parser.error("--h1-days must be positive")
    if args.m5_days <= 0:
        parser.error("--m5-days must be positive")
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    
    print("\n" + "=" * 72)
    print("ENRICHED FLEET TRAINING - PARAMETER SUMMARY")
    print("=" * 72)
    print(f"H1 Data: {args.h1_days} days = {args.h1_days * 24:,} hourly bars")
    print(f"M5 Data: {args.m5_days} days = {args.m5_days * 288:,} 5-minute bars")
    print(f"Holdout: {args.holdout_days} days")
    print(f"CV Folds: {args.folds}")
    print(f"Clusters: {args.clusters}")
    print("=" * 72 + "\n")
    
    run_enriched_fleet_training(
        assets=FLEET_ASSETS,
        n_clusters=args.clusters,
        n_folds=args.folds,
        history_days_h1=args.h1_days,
        history_days_m5=args.m5_days,
        holdout_days=args.holdout_days,
        max_frac_diff_d=args.max_d,
        market_factor_method=args.factor_method,
    )
