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
    history_days_h1: int = 730,  # H1 history (2 years)
    history_days_m5: int = 150,  # M5 history (5 months)
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
    history_days_m5 : int
        Days of M5 history for microstructure (default: 150 = 5 months)
    max_frac_diff_d : float
        Maximum fractional differentiation order
    market_factor_method : str
        Method for market factor extraction
    """
    print("=" * 72)
    print("ENRICHED FLEET TRAINING: H1 Main + M5 Hints")
    print(f"Training {len(assets)} assets with microstructure enrichment")
    print(f"H1 History: {history_days_h1} days, M5 History: {history_days_m5} days")
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
            
            # Store in asset_data
            asset_data[asset_symbol] = df_h1_features
            
            print(f"  ✓ {len(df_h1_features)} samples, {df_h1_features.shape[1]} features")
            
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
    
    # Step 5-7: Cluster-Based Training
    # NOTE: Copy Steps 5-7 from run_hierarchical_fleet.py (lines 343-700)
    # This includes:
    # - Step 5: Cluster-based data preparation
    # - Step 6: Cross-validation training with MoE
    # - Step 7: Telemetry and reporting
    
    # For now, return the enriched data
    print("\n" + "=" * 72)
    print("ENRICHED DATA PREPARATION COMPLETE")
    print("=" * 72)
    print(f"\nAssets loaded: {len(asset_data)}")
    print(f"Clusters formed: {n_clusters}")
    print(f"Dominant cluster: {cluster_result.dominant_cluster_id}")
    
    # Show microstructure feature summary for first asset
    first_asset = list(asset_data.keys())[0]
    micro_cols = [c for c in asset_data[first_asset].columns if c.startswith('micro_')]
    if len(micro_cols) > 0:
        print(f"\n[Microstructure Features] Found {len(micro_cols)} features:")
        for col in micro_cols:
            print(f"  - {col}")
        
        print(f"\n[Sample] {first_asset} microstructure summary:")
        summarize_microstructure(asset_data[first_asset][micro_cols])
    
    print("\n[ENRICHED FLEET] Data loading complete!")
    print("Next: Add cluster-based training logic from run_hierarchical_fleet.py")
    print("Expected: H1 precision (~58%) + M5 microstructure insights")
    
    return asset_data, cluster_result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enriched Fleet Training: H1 Main + M5 Hints"
    )
    parser.add_argument("--clusters", type=int, default=3, help="Number of clusters")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--h1-days", type=int, default=730, help="Days of H1 history")
    parser.add_argument("--m5-days", type=int, default=150, help="Days of M5 history for microstructure")
    parser.add_argument("--max-d", type=float, default=0.65, help="Max frac diff order")
    parser.add_argument("--factor-method", type=str, default="pca",
                       choices=['pca', 'mean', 'weighted_mean'],
                       help="Market factor extraction method")
    
    args = parser.parse_args()
    
    run_enriched_fleet_training(
        assets=FLEET_ASSETS,
        n_clusters=args.clusters,
        n_folds=args.folds,
        history_days_h1=args.h1_days,
        history_days_m5=args.m5_days,
        max_frac_diff_d=args.max_d,
        market_factor_method=args.factor_method,
    )
