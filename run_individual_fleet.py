"""
Individual Fleet Training v2: H1 + M5 Microstructure with Full Telemetry.

Complete implementation with:
- H1 strategic data + M5 microstructure hints
- Individual models per asset (Islands Strategy)
- Comprehensive telemetry and visualization
- Detailed fold-by-fold analytics

Author: QFC System v3.1 - Individual Asset Specialists
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, confusion_matrix, classification_report
)

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
print("!!! INDIVIDUAL FLEET TRAINING v2 (H1 + M5 + Full Telemetry) !!!")
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
# TARGET HORIZON OVERRIDE
# ==============================================================================
print("=" * 72)
print("TRAINING HORIZON OVERRIDE")
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
    
    volume = df['volume'].fillna(df['volume'].median())
    volume_mean = volume.mean()
    volume_std = volume.std()
    
    if volume_std > 1e-8:
        volume_norm = (volume - volume_mean) / volume_std
        volume_norm = np.clip(volume_norm, 0, 5)
    else:
        volume_norm = np.ones(len(df))
    
    price_change = df['close'].pct_change().abs().fillna(0)
    price_mean = price_change.mean()
    price_std = price_change.std()
    
    if price_std > 1e-8:
        price_norm = (price_change - price_mean) / price_std
        price_norm = np.clip(price_norm, 0, 5)
    else:
        price_norm = np.ones(len(df))
    
    energy = volume_norm * price_norm
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


def extract_m5_microstructure_features(df_m5: pd.DataFrame, df_h1: pd.DataFrame) -> pd.DataFrame:
    """
    Extract microstructure features from M5 data and align to H1 timeframe.
    
    Features extracted:
    - Intrabar volatility (high-low range within H1 bar)
    - Volume profile (volume distribution within H1 bar)
    - Tick imbalance (buy vs sell pressure)
    - Microstructure noise (deviation from H1 close)
    """
    print("  [M5 Microstructure] Extracting features...")
    
    # Resample M5 to H1 and calculate microstructure metrics
    df_m5_copy = df_m5.copy()
    
    # Group by hour
    df_m5_copy['hour'] = df_m5_copy.index.floor('H')
    
    micro_features = []
    
    for hour, group in df_m5_copy.groupby('hour'):
        if len(group) == 0:
            continue
        
        # Intrabar volatility
        intrabar_range = (group['high'].max() - group['low'].min()) / group['close'].iloc[-1]
        
        # Volume profile
        volume_std = group['volume'].std() / (group['volume'].mean() + 1e-8)
        
        # Price momentum within bar
        intrabar_return = (group['close'].iloc[-1] / group['open'].iloc[0]) - 1.0
        
        # Microstructure noise (volatility of returns)
        returns = group['close'].pct_change().dropna()
        micro_volatility = returns.std() if len(returns) > 1 else 0.0
        
        micro_features.append({
            'timestamp': hour,
            'm5_intrabar_range': intrabar_range,
            'm5_volume_std': volume_std,
            'm5_intrabar_return': intrabar_return,
            'm5_micro_volatility': micro_volatility,
        })
    
    micro_df = pd.DataFrame(micro_features)
    micro_df = micro_df.set_index('timestamp')
    
    # Align to H1 index
    micro_aligned = micro_df.reindex(df_h1.index, method='ffill')
    
    print(f"  [M5 Microstructure] Extracted {len(micro_df.columns)} features")
    
    return micro_aligned


def save_fold_visualization(
    asset_symbol: str,
    fold_idx: int,
    timestamps: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    close_prices: np.ndarray,
    threshold: float,
    artifacts_dir: Path,
):
    """
    Save comprehensive fold visualization with decision markers.
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    # Plot 1: Price with predictions
    ax1 = axes[0]
    ax1.plot(timestamps, close_prices, 'k-', label='Close Price', linewidth=1, alpha=0.7)
    
    # Mark true positives, false positives, etc.
    tp_mask = (y_pred == 1) & (y_true == 1)
    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)
    
    ax1.scatter(timestamps[tp_mask], close_prices[tp_mask], 
               c='green', marker='^', s=100, label='True Positive', alpha=0.8, edgecolors='darkgreen')
    ax1.scatter(timestamps[fp_mask], close_prices[fp_mask], 
               c='red', marker='v', s=100, label='False Positive', alpha=0.8, edgecolors='darkred')
    ax1.scatter(timestamps[fn_mask], close_prices[fn_mask], 
               c='orange', marker='x', s=100, label='False Negative', alpha=0.8)
    
    ax1.set_ylabel('Price (USDT)', fontsize=12)
    ax1.set_title(f'{asset_symbol} - Fold {fold_idx} - Decision Analysis', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Prediction probabilities
    ax2 = axes[1]
    ax2.plot(timestamps, y_pred_proba, 'b-', label='Predicted Probability', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.2f})', linewidth=2)
    ax2.fill_between(timestamps, 0, y_pred_proba, where=(y_pred_proba >= threshold), 
                     color='green', alpha=0.2, label='Signal Zone')
    ax2.fill_between(timestamps, 0, y_pred_proba, where=(y_pred_proba < threshold), 
                     color='gray', alpha=0.1, label='No Signal Zone')
    
    ax2.set_ylabel('Probability', fontsize=12)
    ax2.set_ylim([0, 1])
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Confusion matrix heatmap (cumulative)
    ax3 = axes[2]
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, 
               xticklabels=['No Trade', 'Trade'], 
               yticklabels=['Actual No', 'Actual Yes'],
               cbar_kws={'label': 'Count'})
    ax3.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Predicted', fontsize=11)
    ax3.set_ylabel('Actual', fontsize=11)
    
    plt.tight_layout()
    
    viz_file = artifacts_dir / f"{asset_symbol}_fold_{fold_idx}_analysis.png"
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  [Visualization] Saved to {viz_file}")


def save_asset_summary_report(
    asset_symbol: str,
    fold_results: List[Dict],
    artifacts_dir: Path,
):
    """
    Save comprehensive asset summary report with all metrics.
    """
    report_file = artifacts_dir / f"{asset_symbol}_summary_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"{asset_symbol} - INDIVIDUAL MODEL SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall metrics
        df_results = pd.DataFrame(fold_results)
        
        f.write("OVERALL PERFORMANCE:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Average Precision:  {df_results['precision'].mean():.4f} ± {df_results['precision'].std():.4f}\n")
        f.write(f"  Average Recall:     {df_results['recall'].mean():.4f} ± {df_results['recall'].std():.4f}\n")
        f.write(f"  Average F1 Score:   {df_results['f1'].mean():.4f} ± {df_results['f1'].std():.4f}\n")
        f.write(f"  Average Accuracy:   {df_results['accuracy'].mean():.4f} ± {df_results['accuracy'].std():.4f}\n")
        f.write(f"  Average Expectancy: {df_results['expectancy'].mean():.6f} ± {df_results['expectancy'].std():.6f}\n")
        f.write("\n")
        
        # Per-fold breakdown
        f.write("PER-FOLD BREAKDOWN:\n")
        f.write("-" * 80 + "\n")
        for fold_res in fold_results:
            f.write(f"\nFold {fold_res['fold']}:\n")
            f.write(f"  Precision:  {fold_res['precision']:.4f}\n")
            f.write(f"  Recall:     {fold_res['recall']:.4f}\n")
            f.write(f"  F1 Score:   {fold_res['f1']:.4f}\n")
            f.write(f"  Accuracy:   {fold_res['accuracy']:.4f}\n")
            f.write(f"  Expectancy: {fold_res['expectancy']:.6f}\n")
            
            if 'roc_auc' in fold_res:
                f.write(f"  ROC AUC:    {fold_res['roc_auc']:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"  [Report] Saved to {report_file}")


def run_individual_fleet_training(
    assets: List[str] = FLEET_ASSETS,
    n_folds: int = 5,
    history_days_h1: int = 730,  # Days worth of H1 data
    history_days_m5: int = 150,  # Days worth of M5 data
    holdout_days: int = 0,
    max_frac_diff_d: float = 0.65,
    apply_forced_entry: bool = True,
):
    """
    Train individual MoE models for each asset with H1 + M5 microstructure.
    
    Parameters
    ----------
    assets : list
        List of asset symbols to train on
    n_folds : int
        Number of cross-validation folds
    history_days_h1 : int
        Days worth of H1 data (1 day = 24 H1 bars)
    history_days_m5 : int
        Days worth of M5 data (1 day = 288 M5 bars)
    holdout_days : int
        Days to exclude from training for validation
    max_frac_diff_d : float
        Maximum fractional differentiation order
    apply_forced_entry : bool
        Apply forced daily entry policy
    """
    print("=" * 72)
    print("INDIVIDUAL FLEET TRAINING v2: H1 + M5 + Full Telemetry")
    print(f"Training {len(assets)} assets with ISOLATED models")
    print(f"H1 History: {history_days_h1} days ({history_days_h1 * 24:,} bars)")
    print(f"M5 History: {history_days_m5} days ({history_days_m5 * 288:,} bars)")
    if holdout_days > 0:
        print(f"Holdout Period: {holdout_days} days")
    print(f"Forced Daily Entry: {'ENABLED' if apply_forced_entry else 'DISABLED'}")
    print("=" * 72)
    
    # Create artifacts directory
    artifacts_dir = Path("artifacts/individual_fleet")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
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
            # Load H1 data (primary)
            loader_h1 = MarketDataLoader(symbol=asset_symbol, interval="60")
            factory = SignalFactory()
            
            df_h1_raw = loader_h1.get_data(days_back=history_days_h1)
            
            if df_h1_raw is None or len(df_h1_raw) < 1000:
                print(f"  [WARNING] Insufficient H1 data for {asset_symbol}, skipping...")
                continue
            
            print(f"  [H1 Data] Loaded {len(df_h1_raw)} hourly candles")
            
            # Load M5 data (for microstructure)
            print(f"  [M5 Data] Loading for microstructure features...")
            loader_m5 = MarketDataLoader(symbol=asset_symbol, interval="5")
            df_m5_raw = loader_m5.get_data(days_back=history_days_m5)
            
            has_m5 = False
            if df_m5_raw is not None and len(df_m5_raw) > 1000:
                print(f"  [M5 Data] Loaded {len(df_m5_raw)} 5-minute candles")
                has_m5 = True
            else:
                print(f"  [M5 Data] Insufficient data, proceeding without microstructure")
            
            # Fractional Differentiation on H1
            frac_diff = FractionalDifferentiator(window_size=2048)
            
            n_calib = min(500, int(len(df_h1_raw) * 0.1))
            calib_series = df_h1_raw['close'].iloc[:n_calib]
            
            optimal_d = frac_diff.find_min_d(calib_series, precision=0.05, verbose=False)
            
            if optimal_d > max_frac_diff_d:
                print(f"  [FracDiff] Capping d: {optimal_d:.3f} -> {max_frac_diff_d:.3f}")
                optimal_d = max_frac_diff_d
            
            df_h1_raw['frac_diff'] = frac_diff.transform(df_h1_raw['close'], d=optimal_d)
            
            print(f"  [FracDiff] Optimal d: {optimal_d:.3f}")
            
            # Generate H1 features
            df_h1_features = factory.generate_signals(df_h1_raw)
            df_h1_features['frac_diff'] = df_h1_raw['frac_diff'].reindex(df_h1_features.index)
            df_h1_features['close'] = df_h1_raw['close'].reindex(df_h1_features.index)
            df_h1_features['volume'] = df_h1_raw['volume'].reindex(df_h1_features.index)
            
            # Add M5 microstructure features if available
            if has_m5:
                micro_features = extract_m5_microstructure_features(df_m5_raw, df_h1_features)
                for col in micro_features.columns:
                    df_h1_features[col] = micro_features[col]
            
            # Apply holdout period if specified
            if holdout_days > 0:
                cutoff_date = df_h1_features.index.max() - pd.Timedelta(days=holdout_days)
                df_train = df_h1_features[df_h1_features.index <= cutoff_date].copy()
                df_holdout = df_h1_features[df_h1_features.index > cutoff_date].copy()
                
                print(f"  [Holdout] Training: {len(df_train)} samples (up to {cutoff_date.date()})")
                print(f"  [Holdout] Held out: {len(df_holdout)} samples")
                
                holdout_file = artifacts_dir / f"holdout_{asset_symbol}.csv"
                df_holdout.to_csv(holdout_file)
                print(f"  [Holdout] Saved to: {holdout_file}")
            else:
                df_train = df_h1_features
            
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
                print(f"    [WARNING] Insufficient valid samples ({len(X_clean)}), skipping...")
                continue
            
            print(f"    {asset_symbol}: {len(X_clean)} samples, {X_clean['target'].mean():.2%} positive")
            
            # Extract features and labels
            timestamp_col = X_clean.index.copy()
            X_final = X_clean.drop(columns=['target'], errors='ignore')
            y_final = X_clean['target']
            energy_weights = X_clean['energy'].values
            close_prices = df_train['close'].reindex(timestamp_col).values
            
            # Feature partitioning
            available_physics = [c for c in PHYSICS_COLUMNS if c in X_final.columns]
            stability_cols = ["stability_theta", "stability_acf", "stability_warning"]
            available_physics.extend([c for c in stability_cols if c in X_final.columns])
            
            # Add M5 microstructure to passthrough
            m5_cols = [c for c in X_final.columns if c.startswith('m5_')]
            
            passthrough_cols = available_physics + ["frac_diff", "energy"] + m5_cols
            tensor_feature_cols = [c for c in X_final.columns if c not in passthrough_cols]
            
            print(f"\n[Features] Total: {len(X_final.columns)}")
            print(f"[Features] Passthrough: {len(passthrough_cols)} (includes {len(m5_cols)} M5 microstructure)")
            print(f"[Features] Tensor-Flex Candidates: {len(tensor_feature_cols)}")
            
            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=n_folds)
            fold_results = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_final), start=1):
                print(f"\n{'-' * 72}")
                print(f"{asset_symbol} - FOLD {fold_idx}/{n_folds}")
                print(f"{'-' * 72}")
                
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
                    
                    print(f"  [OK] Refined: {len(tensor_feature_cols)} -> {X_train_tf.shape[1]} latents")
                    
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
                
                # Calculate comprehensive metrics
                tp = ((y_pred == 1) & (y_val == 1)).sum()
                fp = ((y_pred == 1) & (y_val == 0)).sum()
                fn = ((y_pred == 0) & (y_val == 1)).sum()
                tn = ((y_pred == 0) & (y_val == 0)).sum()
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
                expectancy = (precision * TP_PCT) - ((1 - precision) * SL_PCT)
                
                # ROC AUC if we have both classes
                try:
                    roc_auc = roc_auc_score(y_val, y_pred_proba)
                except:
                    roc_auc = 0.0
                
                print(f"\n[Fold {fold_idx}] {asset_symbol} Performance:")
                print(f"  Precision: {precision:.2%}, Recall: {recall:.2%}, F1: {f1:.2%}")
                print(f"  Accuracy: {accuracy:.2%}, ROC AUC: {roc_auc:.4f}")
                print(f"  Expectancy: {expectancy:.4f}")
                
                # Save fold visualization
                val_timestamps = timestamp_col.iloc[val_idx]
                val_close_prices = close_prices[val_idx]
                
                save_fold_visualization(
                    asset_symbol=asset_symbol,
                    fold_idx=fold_idx,
                    timestamps=val_timestamps,
                    y_true=y_val.values,
                    y_pred=y_pred,
                    y_pred_proba=y_pred_proba,
                    close_prices=val_close_prices,
                    threshold=threshold,
                    artifacts_dir=artifacts_dir,
                )
                
                fold_results.append({
                    'asset': asset_symbol,
                    'fold': fold_idx,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'accuracy': accuracy,
                    'roc_auc': roc_auc,
                    'expectancy': expectancy,
                    'y_pred_proba': y_pred_proba,
                    'y_pred': y_pred,
                    'timestamps': val_timestamps,
                })
                
                del X_train, X_val, y_train, y_val, moe
                gc.collect()
            
            # Save comprehensive asset summary report
            save_asset_summary_report(asset_symbol, fold_results, artifacts_dir)
            
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
                
                pred_file = artifacts_dir / f"predictions_{asset_symbol}.csv"
                pred_df.to_csv(pred_file, index=False)
                print(f"  [OK] Saved {len(pred_df)} predictions to {pred_file}")
            
            # Asset summary
            asset_df_results = pd.DataFrame(fold_results)
            
            print(f"\n{'=' * 72}")
            print(f"{asset_symbol} SUMMARY")
            print(f"{'=' * 72}")
            print(f"  Avg Precision: {asset_df_results['precision'].mean():.2%}")
            print(f"  Avg Recall:    {asset_df_results['recall'].mean():.2%}")
            print(f"  Avg F1 Score:  {asset_df_results['f1'].mean():.2%}")
            print(f"  Avg ROC AUC:   {asset_df_results['roc_auc'].mean():.4f}")
            print(f"  Avg Expectancy: {asset_df_results['expectancy'].mean():.5f}")
            
            all_results.append({
                'asset': asset_symbol,
                'avg_precision': asset_df_results['precision'].mean(),
                'avg_recall': asset_df_results['recall'].mean(),
                'avg_f1': asset_df_results['f1'].mean(),
                'avg_roc_auc': asset_df_results['roc_auc'].mean(),
                'avg_expectancy': asset_df_results['expectancy'].mean(),
                'folds': n_folds,
            })
            
        except Exception as e:
            import traceback
            print(f"  [ERROR] Error processing {asset_symbol}: {e}")
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
        summary_df.to_csv(artifacts_dir / "fleet_summary.csv", index=False)
        
        print(f"\n[Artifacts] Results saved to:")
        print(f"  - {artifacts_dir / 'fleet_summary.csv'}")
        print(f"  - {artifacts_dir} (fold visualizations and reports)")
        
        # Success criteria
        all_profitable = (summary_df['avg_expectancy'] > 0).all()
        avg_precision = summary_df['avg_precision'].mean()
        precision_pass = avg_precision > 0.55
        
        print("\n" + "=" * 72)
        print("INDIVIDUAL FLEET VERIFICATION")
        print("=" * 72)
        print(f"[OK] All Assets Profitable: {'PASS' if all_profitable else 'FAIL'}")
        print(f"[OK] Avg Precision > 55%:   {'PASS' if precision_pass else 'FAIL'} ({avg_precision:.2%})")
        
        if all_profitable and precision_pass:
            print("\n[SUCCESS] INDIVIDUAL FLEET TRAINING SUCCESSFUL!")
            print("   Islands Strategy: Each asset has its own specialized model!")
        else:
            print("\n[WARNING] INDIVIDUAL FLEET NEEDS TUNING")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Individual Fleet Training v2: H1 + M5 + Full Telemetry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Default: 2 years H1, 5 months M5, 5 folds
  python run_individual_fleet.py
  
  # Custom: 1 year H1, 3 months M5, 10 folds
  python run_individual_fleet.py --h1-days 365 --m5-days 90 --folds 10
  
  # Quick test: 30 days H1, 7 days M5, 3 folds
  python run_individual_fleet.py --h1-days 30 --m5-days 7 --folds 3
  
Note: 
  --h1-days 1 = 24 H1 bars (1 day of hourly data)
  --m5-days 1 = 288 M5 bars (1 day of 5-minute data)
"""
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--h1-days", type=int, default=730, 
                       help="Days worth of H1 data (1 day = 24 bars). Default: 730 = 2 years")
    parser.add_argument("--m5-days", type=int, default=150, 
                       help="Days worth of M5 data (1 day = 288 bars). Default: 150 = 5 months")
    parser.add_argument("--holdout-days", type=int, default=0, help="Days to exclude from training")
    parser.add_argument("--max-d", type=float, default=0.65, help="Max frac diff order")
    parser.add_argument("--no-forced-entry", action="store_true", help="Disable forced daily entry policy")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 72)
    print("INDIVIDUAL FLEET TRAINING - PARAMETER SUMMARY")
    print("=" * 72)
    print(f"H1 Data: {args.h1_days} days = {args.h1_days * 24:,} hourly bars")
    print(f"M5 Data: {args.m5_days} days = {args.m5_days * 288:,} 5-minute bars")
    print(f"Holdout: {args.holdout_days} days")
    print(f"CV Folds: {args.folds}")
    print("=" * 72 + "\n")
    
    run_individual_fleet_training(
        assets=FLEET_ASSETS,
        n_folds=args.folds,
        history_days_h1=args.h1_days,
        history_days_m5=args.m5_days,
        holdout_days=args.holdout_days,
        max_frac_diff_d=args.max_d,
        apply_forced_entry=not args.no_forced_entry,
    )
