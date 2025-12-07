"""
Specialized MoE Verification Script.

This script combines ALL our wins:
1. FractionalDifferentiator (Auto-tuned d)
2. TensorFlex v2 (Auto-tuned latents, "Three-Layer Brain")
3. Specialized MoE (Subtraction Strategy)
4. Physics Gating (Sample weighting)
5. Rigorous 5-Fold CV with Bootstrap

Goal: Achieve stable Precision > 53% by feeding "Clean Data" (FracDiff)
      into "Reliable Models" (HistGBM/CNN).
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import List

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
print("!!! SPECIALIZED MOE PIPELINE !!!")
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


def build_labels(df: pd.DataFrame) -> pd.Series:
    """Build forward-looking labels."""
    forward_ret = df['close'].shift(-LABEL_LOOKAHEAD) / df['close'] - 1.0
    y = (forward_ret > LABEL_THRESHOLD).astype(int)
    return y


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


def run_specialized_moe_pipeline(
    symbol: str = "BTCUSDT",
    n_folds: int = 5,
):
    """
    Main pipeline combining all wins.
    
    Parameters
    ----------
    symbol : str
        Asset symbol to analyze
    n_folds : int
        Number of cross-validation folds
    """
    print("=" * 72)
    print("SPECIALIZED MOE VERIFICATION PIPELINE")
    print("Combining: FracDiff + TensorFlex + Specialized MoE + Physics Gating")
    print("=" * 72)
    
    # Step 1: Load Data
    print("\n" + "=" * 72)
    print("STEP 1: DATA LOADING")
    print("=" * 72)
    
    # Initialize loader with symbol
    loader = MarketDataLoader(symbol=symbol, interval="60")  # 1H candles
    factory = SignalFactory()
    
    print(f"\n[Config] Asset: {symbol}")
    print(f"[Config] Interval: 1H")
    print(f"[Config] Days Back: {DAYS_BACK}")
    print(f"[Config] Folds: {n_folds}")
    
    # Load OHLCV data
    print(f"\n[Data] Loading {symbol}...")
    df_raw = loader.get_data(days_back=DAYS_BACK)
    
    if df_raw is None or len(df_raw) < 1000:
        raise RuntimeError(f"Insufficient data for {symbol}")
    
    print(f"[Data] Loaded {len(df_raw)} candles")
    
    # Step 2: Fractional Differentiation (The "Alpha" Layer)
    print("\n" + "=" * 72)
    print("STEP 2: FRACTIONAL DIFFERENTIATION (Memory Preservation)")
    print("=" * 72)
    
    frac_diff = FractionalDifferentiator(window_size=2048)
    
    # Find optimal d on first 10% (avoid look-ahead)
    n_calib = int(len(df_raw) * 0.1)
    if n_calib < 100:
        n_calib = min(100, len(df_raw))
    
    calib_series = df_raw['close'].iloc[:n_calib]
    
    print(f"[FracDiff] Calibrating on {len(calib_series)} samples...")
    optimal_d = frac_diff.find_min_d(
        calib_series,
        precision=0.05,
        verbose=True
    )
    
    # Apply to full series
    df_raw['frac_diff'] = frac_diff.transform(df_raw['close'], d=optimal_d)
    
    print(f"\n[FracDiff] Optimal d: {optimal_d:.3f}")
    print(f"[FracDiff] ✓ Feature added to dataset")
    
    # Step 3: Generate Standard Features
    print("\n" + "=" * 72)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 72)
    
    print("[Features] Generating technical indicators...")
    df_features = factory.generate_signals(df_raw)
    
    # Add frac_diff back (it gets dropped in generate_signals)
    df_features['frac_diff'] = df_raw['frac_diff'].reindex(df_features.index)
    
    print(f"[Features] Generated {df_features.shape[1]} features")
    
    # Step 4: Build Labels
    print("\n" + "=" * 72)
    print("STEP 4: LABEL GENERATION")
    print("=" * 72)
    
    # Add close back for label generation
    df_features['close'] = df_raw['close'].reindex(df_features.index)
    
    y_global = build_labels(df_features)
    valid_mask = ~y_global.isna()
    
    X = df_features.loc[valid_mask].drop(columns=['close'], errors='ignore')
    y = y_global.loc[valid_mask]
    
    print(f"[Labels] Valid samples: {len(X)} / {len(df_features)}")
    print(f"[Labels] Positive class: {y.sum()} ({y.mean():.2%})")
    
    # Remove rows where frac_diff is NaN
    frac_diff_valid_mask = ~X['frac_diff'].isna()
    n_before = len(X)
    X = X.loc[frac_diff_valid_mask]
    y = y.loc[frac_diff_valid_mask]
    n_after = len(X)
    
    if n_before > n_after:
        print(f"[FracDiff] Dropped {n_before - n_after} rows with NaN frac_diff values")
    
    print(f"[FracDiff] Final dataset: {len(X)} samples")
    
    # Step 5: Feature Partitioning
    print("\n" + "=" * 72)
    print("STEP 5: FEATURE PARTITIONING")
    print("=" * 72)
    
    # Identify passthrough columns (physics + frac_diff)
    available_physics = [c for c in PHYSICS_COLUMNS if c in X.columns]
    stability_cols = ["stability_theta", "stability_acf", "stability_warning"]
    available_physics.extend([c for c in stability_cols if c in X.columns])
    
    passthrough_cols = available_physics + ["frac_diff"]
    tensor_feature_cols = [c for c in X.columns if c not in passthrough_cols]
    
    print(f"[Features] Total: {len(X.columns)}")
    print(f"[Features] Passthrough (Physics + FracDiff): {len(passthrough_cols)}")
    print(f"[Features] Tensor-Flex Candidates: {len(tensor_feature_cols)}")
    
    # Load best CNN params if available
    cnn_params = None
    best_params_path = Path("artifacts/best_cnn_params.json")
    if best_params_path.exists():
        with open(best_params_path, "r") as f:
            cnn_params = json.load(f)
        print(f"\n[CNN] Loaded tuned params: {cnn_params}")
    else:
        print("\n[CNN] No tuned params found, using defaults")
    
    # Step 6: Cross-Validation Loop
    print("\n" + "=" * 72)
    print(f"STEP 6: TIME-SERIES CROSS-VALIDATION ({n_folds} Folds)")
    print("=" * 72)
    
    # Safety Check
    if not cfg.USE_TENSOR_FLEX:
        print("WARNING: Tensor-Flex is OFF! Forcing it ON...")
        cfg.USE_TENSOR_FLEX = True
    
    tscv = TimeSeriesSplit(n_splits=n_folds)
    fold_results = []
    expert_telemetry_all = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        print(f"\n{'─' * 72}")
        print(f"FOLD {fold_idx}/{n_folds}")
        print(f"{'─' * 72}")
        
        X_train_raw = X.iloc[train_idx]
        X_val_raw = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        
        print(f"Train: {len(X_train_raw)} samples | Val: {len(X_val_raw)} samples")
        
        # Tensor-Flex v2 Refinement
        print(f"\n[Fold {fold_idx}] Tensor-Flex v2 Refinement")
        print("  🧠 Three-Layer Brain: Physics → Geometry → Economics")
        
        if cfg.USE_TENSOR_FLEX and tensor_feature_cols:
            print("  Fitting fresh Tensor-Flex on training fold only...")
            
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
            
            print(f"  ✓ Refined: {len(tensor_feature_cols)} → {X_train_tf.shape[1]} latent features")
            
            del refiner, X_train_tensor, X_val_tensor, X_train_tf, X_val_tf
            gc.collect()
        else:
            X_train = X_train_raw
            X_val = X_val_raw
            print("  Tensor-Flex disabled, using raw features.")
        
        # Physics-Aware MoE Training
        print(f"\n[Fold {fold_idx}] Specialized MoE Training")
        
        sample_weights = create_physics_sample_weights(X_train)
        
        moe = MixtureOfExpertsEnsemble(
            physics_features=available_physics,
            random_state=RANDOM_SEED,
            use_cnn=True,
            cnn_params=cnn_params,
            cnn_epochs=15,
        )
        
        moe.fit(X_train, y_train, sample_weight=sample_weights)
        
        # Get expert telemetry
        telemetry = moe.get_expert_telemetry(X_val)
        expert_telemetry_all.append(telemetry)
        
        print(f"\n[Fold {fold_idx}] Expert Weights Distribution:")
        print(f"  Trend:   {telemetry['share_trend']:.2%}")
        print(f"  Range:   {telemetry['share_range']:.2%}")
        print(f"  Stress:  {telemetry['share_stress']:.2%}")
        print(f"  Pattern: {telemetry['share_cnn']:.2%}")
        print(f"  Gating Confidence: {telemetry['gating_confidence']:.2%}")
        
        # Predict on validation
        y_pred_proba = moe.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Bootstrap Validation
        print(f"\n[Fold {fold_idx}] Bootstrap Validation (50 iterations)")
        
        boot_metrics = bootstrap_metrics(
            y_val.values,
            y_pred,
            n_iterations=50,
            random_state=RANDOM_SEED + fold_idx,
        )
        
        fold_results.append({
            "fold": fold_idx,
            "train_size": len(X_train),
            "val_size": len(X_val),
            "precision_mean": boot_metrics["precision_mean"],
            "precision_5th": boot_metrics["precision_5th"],
            "recall_mean": boot_metrics["recall_mean"],
            "expectancy_mean": boot_metrics["expectancy_mean"],
            "expectancy_5th": boot_metrics["expectancy_5th"],
            **telemetry,
        })
        
        print(f"  Precision: {boot_metrics['precision_mean']:.2%} "
              f"(5th: {boot_metrics['precision_5th']:.2%})")
        print(f"  Recall:    {boot_metrics['recall_mean']:.2%}")
        print(f"  Expectancy: {boot_metrics['expectancy_mean']:.4f} "
              f"(5th: {boot_metrics['expectancy_5th']:.4f})") 
        
        del X_train, X_val, y_train, y_val, moe
        gc.collect()
    
    # Step 7: Reporting
    print("\n" + "=" * 72)
    print("STEP 7: FINAL REPORT")
    print("=" * 72)
    
    results_df = pd.DataFrame(fold_results)
    
    print("\nPer-Fold Results:")
    print(results_df[["fold", "precision_mean", "precision_5th", "recall_mean", "expectancy_mean"]].to_string(index=False))
    
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
    
    # Expert activation summary
    print("\n" + "─" * 72)
    print("EXPERT ACTIVATION SUMMARY")
    print("─" * 72)
    avg_trend = results_df["share_trend"].mean()
    avg_range = results_df["share_range"].mean()
    avg_stress = results_df["share_stress"].mean()
    avg_cnn = results_df["share_cnn"].mean()
    avg_confidence = results_df["gating_confidence"].mean()
    
    print(f"Average Trend Weight:   {avg_trend:.2%}")
    print(f"Average Range Weight:   {avg_range:.2%}")
    print(f"Average Stress Weight:  {avg_stress:.2%}")
    print(f"Average Pattern Weight: {avg_cnn:.2%}")
    print(f"Average Gating Confidence: {avg_confidence:.2%}")
    
    # Pass/Fail Criteria
    print("\n" + "=" * 72)
    print("PRODUCTION READINESS CHECK")
    print("=" * 72)
    
    precision_pass = avg_precision > 0.53
    expectancy_pass = avg_expectancy > 0.0
    stability_pass = avg_precision_5th > 0.50  # 5th percentile > 50%
    
    print(f"✓ Precision > 53%:        {'PASS' if precision_pass else 'FAIL'} ({avg_precision:.2%})")
    print(f"✓ Expectancy > 0:         {'PASS' if expectancy_pass else 'FAIL'} ({avg_expectancy:.4f})")
    print(f"✓ Precision (5th) > 50%:  {'PASS' if stability_pass else 'FAIL'} ({avg_precision_5th:.2%})")
    
    if precision_pass and expectancy_pass and stability_pass:
        print("\n🎯 MODEL IS PRODUCTION READY")
        print("   Clean Data (FracDiff) + Reliable Models (HistGBM/CNN) = Alpha!")
    else:
        print("\n⚠️  MODEL NEEDS IMPROVEMENT")
    
    # Save results
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    results_df.to_csv(artifacts_dir / "specialized_moe_results.csv", index=False)
    
    # Save summary report
    report_path = artifacts_dir / "specialized_moe_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write("SPECIALIZED MOE VERIFICATION REPORT\n")
        f.write("=" * 72 + "\n\n")
        
        f.write(f"Asset: {symbol}\n")
        f.write(f"Optimal d: {optimal_d:.3f}\n")
        f.write(f"Folds: {n_folds}\n")
        f.write(f"Total Samples: {len(X)}\n\n")
        
        f.write("AGGREGATE METRICS\n")
        f.write("-" * 72 + "\n")
        f.write(f"Average Precision:       {avg_precision:.4f}\n")
        f.write(f"Average Precision (5th): {avg_precision_5th:.4f}\n")
        f.write(f"Average Expectancy:      {avg_expectancy:.6f}\n")
        f.write(f"Average Expectancy (5th): {avg_expectancy_5th:.6f}\n\n")
        
        f.write("EXPERT ACTIVATION\n")
        f.write("-" * 72 + "\n")
        f.write(f"Trend Weight:   {avg_trend:.4f}\n")
        f.write(f"Range Weight:   {avg_range:.4f}\n")
        f.write(f"Stress Weight:  {avg_stress:.4f}\n")
        f.write(f"Pattern Weight: {avg_cnn:.4f}\n")
        f.write(f"Gating Confidence: {avg_confidence:.4f}\n\n")
        
        f.write("PER-FOLD RESULTS\n")
        f.write("-" * 72 + "\n")
        f.write(results_df.to_string(index=False))
        f.write("\n")
    
    print(f"\n[Artifacts] Results saved to:")
    print(f"  - {artifacts_dir / 'specialized_moe_results.csv'}")
    print(f"  - {artifacts_dir / 'specialized_moe_report.txt'}")
    
    return results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Specialized MoE Verification Pipeline"
    )
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Asset symbol")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    
    args = parser.parse_args()
    
    run_specialized_moe_pipeline(
        symbol=args.symbol,
        n_folds=args.folds,
    )
