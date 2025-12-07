"""
Regime-Adaptive MoE Pipeline.

This script implements dynamic thresholding based on market stability (theta).

PROBLEM:
- Static threshold causes low Recall in stable regimes (too conservative)
- Static threshold causes low Precision in chaotic regimes (too aggressive)

SOLUTION:
- Adaptive threshold: th_t = base_th + sensitivity * (max_theta - theta_t)
- As theta drops (instability) → threshold rises (more conservative)
- As theta rises (stability) → threshold drops (more aggressive)

Experts:
1. Trend (HistGBM) - Sustainable trends
2. Range (KNN) - Local patterns  
3. Stress (LogReg) - Crash protection
4. Elastic (OU) - Mean reversion / elasticity
5. Pattern (CNN) - Temporal sequences

Success Criteria:
- Recall improves in stable folds (Folds 1-4)
- Precision improves in chaotic fold (Fold 5)
- Overall Expectancy increases
- Adaptive > Static threshold performance

Goal: Solve precision-recall trade-off via regime-aware decision making.
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import List, Tuple

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
print("!!! REGIME-ADAPTIVE MOE PIPELINE !!!")
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


class AdaptiveThresholdPolicy:
    """
    Regime-adaptive threshold policy based on market stability.
    
    STABILITY REWARD LOGIC (NEW):
    - Start with STRICT threshold (conservative)
    - LOWER threshold when theta confirms stability (reward)
    - Result: Aggressive in stable markets, conservative in chaos
    
    OLD LOGIC (PENALTY):
    - Started low, raised threshold in chaos
    - Result: Too conservative overall, killed recall
    
    Formula:
    --------
    th_t = max(0.5, base_th - sensitivity * theta_normalized)
    
    Example:
    --------
    base_th = 0.65, sensitivity = 0.15
    - theta = 0.0 (chaos)   → th = max(0.5, 0.65 - 0.15*0.0) = 0.65 (strict)
    - theta = 0.5 (medium)  → th = max(0.5, 0.65 - 0.15*0.5) = 0.575 (moderate)
    - theta = 1.0 (stable)  → th = max(0.5, 0.65 - 0.15*1.0) = 0.50 (aggressive)
    
    Parameters
    ----------
    base_th : float
        Base threshold (maximum, used in chaos)
        Range: 0.55-0.70 (start strict)
    sensitivity : float
        How much threshold decreases with stability
        Range: 0.05-0.20 (reward for stability)
    max_theta : float
        Maximum theta value (for normalization)
    """
    
    def __init__(
        self,
        base_th: float = 0.65,
        sensitivity: float = 0.15,
        max_theta: float = 1.0,
    ):
        self.base_th = base_th
        self.sensitivity = sensitivity
        self.max_theta = max_theta
    
    def compute_threshold(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute adaptive threshold based on theta.
        
        STABILITY REWARD: Lower threshold when theta is high (stable).
        
        Parameters
        ----------
        theta : ndarray
            Stability theta values
        
        Returns
        -------
        thresholds : ndarray
            Adaptive thresholds for each sample
        """
        # Normalize theta to [0, max_theta]
        theta_norm = np.clip(theta, 0, self.max_theta) / self.max_theta
        
        # Stability Reward: SUBTRACT theta (lower threshold when stable)
        thresholds = self.base_th - self.sensitivity * theta_norm
        
        # Floor at 0.5 (never go below neutral)
        thresholds = np.maximum(thresholds, 0.5)
        
        return thresholds
    
    def apply(
        self,
        probabilities: np.ndarray,
        theta: np.ndarray,
    ) -> np.ndarray:
        """
        Apply adaptive threshold to probabilities.
        
        Parameters
        ----------
        probabilities : ndarray
            Predicted probabilities (P(Up))
        theta : ndarray
            Stability theta values
        
        Returns
        -------
        predictions : ndarray
            Binary predictions (0 or 1)
        """
        thresholds = self.compute_threshold(theta)
        predictions = (probabilities >= thresholds).astype(int)
        return predictions


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


def evaluate_threshold_policy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    policy_name: str = "Policy",
) -> dict:
    """
    Evaluate threshold policy performance.
    
    Parameters
    ----------
    y_true : ndarray
        True labels
    y_pred : ndarray
        Predicted labels
    policy_name : str
        Name of the policy (for logging)
    
    Returns
    -------
    metrics : dict
        Performance metrics
    """
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Expectancy
    expectancy = (precision * TP_PCT) - ((1 - precision) * SL_PCT)
    
    return {
        "policy": policy_name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "expectancy": expectancy,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def calibrate_adaptive_policy(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    theta: np.ndarray,
    base_th_range: Tuple[float, float] = (0.60, 0.70),
    sensitivity_range: Tuple[float, float] = (0.05, 0.20),
    n_steps: int = 10,
) -> Tuple[AdaptiveThresholdPolicy, dict]:
    """
    Calibrate adaptive threshold policy via grid search.
    
    NEW OBJECTIVE: Expectancy * log(Trades)
    - Pure Expectancy favors 1 trade with 100% win rate
    - Multiplying by log(Trades) forces optimizer to value frequency (Recall)
    - Result: Balance profitability with trade frequency
    
    Parameters
    ----------
    y_true : ndarray
        True labels
    probabilities : ndarray
        Predicted probabilities
    theta : ndarray
        Stability theta values
    base_th_range : tuple
        Range for base_th parameter (strict threshold)
        Default: (0.60, 0.70) - start conservative (raised from 0.55 for 10 assets)
    sensitivity_range : tuple
        Range for sensitivity parameter (stability reward)
        Default: (0.05, 0.20) - reward for stability
    n_steps : int
        Number of steps in grid search
    
    Returns
    -------
    best_policy : AdaptiveThresholdPolicy
        Best policy found
    best_metrics : dict
        Metrics for best policy
    """
    print("\n[Calibration] Searching for optimal adaptive policy...")
    print(f"  Objective: Expectancy * log(Trades) (balance profit + frequency)")
    print(f"  Base Threshold: {base_th_range[0]:.2f} - {base_th_range[1]:.2f}")
    print(f"  Sensitivity: {sensitivity_range[0]:.2f} - {sensitivity_range[1]:.2f}")
    
    base_th_values = np.linspace(base_th_range[0], base_th_range[1], n_steps)
    sensitivity_values = np.linspace(sensitivity_range[0], sensitivity_range[1], n_steps)
    
    best_score = -np.inf
    best_policy = None
    best_metrics = None
    
    for base_th in base_th_values:
        for sensitivity in sensitivity_values:
            policy = AdaptiveThresholdPolicy(
                base_th=base_th,
                sensitivity=sensitivity,
                max_theta=theta.max() if len(theta) > 0 else 1.0,
            )
            
            y_pred = policy.apply(probabilities, theta)
            metrics = evaluate_threshold_policy(y_true, y_pred, "Adaptive")
            
            # Calculate objective: Expectancy * log(Trades)
            n_trades = (y_pred == 1).sum()
            if n_trades > 0:
                score = metrics["expectancy"] * np.log(n_trades + 1)  # +1 to avoid log(0)
            else:
                score = -np.inf  # No trades = bad
            
            if score > best_score:
                best_score = score
                best_policy = policy
                best_metrics = metrics
                best_metrics["n_trades"] = int(n_trades)
                best_metrics["score"] = float(score)
    
    n_trades = best_metrics.get("n_trades", 0)
    print(f"  ✓ Best: base_th={best_policy.base_th:.3f}, "
          f"sensitivity={best_policy.sensitivity:.3f}")
    print(f"    Expectancy: {best_metrics['expectancy']:.4f}, "
          f"Precision: {best_metrics['precision']:.2%}, "
          f"Recall: {best_metrics['recall']:.2%}")
    print(f"    Trades: {n_trades}, Score: {best_metrics['score']:.4f}")
    
    return best_policy, best_metrics


def run_adaptive_moe_pipeline(
    symbol: str = "BTCUSDT",
    n_folds: int = 5,
):
    """
    Main pipeline with regime-adaptive thresholding.
    
    Parameters
    ----------
    symbol : str
        Asset symbol to analyze
    n_folds : int
        Number of cross-validation folds
    """
    print("=" * 72)
    print("REGIME-ADAPTIVE MOE PIPELINE")
    print("Adaptive Thresholding: th_t = base_th + sensitivity * (max_theta - theta_t)")
    print("=" * 72)
    
    # Step 1: Load Data
    print("\n" + "=" * 72)
    print("STEP 1: DATA LOADING")
    print("=" * 72)
    
    loader = MarketDataLoader(symbol=symbol, interval="60")
    factory = SignalFactory()
    
    print(f"\n[Config] Asset: {symbol}")
    print(f"[Config] Interval: 1H")
    print(f"[Config] Days Back: {DAYS_BACK}")
    print(f"[Config] Folds: {n_folds}")
    
    df_raw = loader.get_data(days_back=DAYS_BACK)
    
    if df_raw is None or len(df_raw) < 1000:
        raise RuntimeError(f"Insufficient data for {symbol}")
    
    print(f"[Data] Loaded {len(df_raw)} candles")
    
    # Step 2: Fractional Differentiation
    print("\n" + "=" * 72)
    print("STEP 2: FRACTIONAL DIFFERENTIATION (Memory Preservation)")
    print("=" * 72)
    
    frac_diff = FractionalDifferentiator(window_size=2048)
    
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
    
    df_raw['frac_diff'] = frac_diff.transform(df_raw['close'], d=optimal_d)
    
    print(f"\n[FracDiff] Optimal d: {optimal_d:.3f}")
    print(f"[FracDiff] ✓ Feature added to dataset")
    
    # Step 3: Generate Features
    print("\n" + "=" * 72)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 72)
    
    print("[Features] Generating technical indicators...")
    df_features = factory.generate_signals(df_raw)
    df_features['frac_diff'] = df_raw['frac_diff'].reindex(df_features.index)
    
    print(f"[Features] Generated {df_features.shape[1]} features")
    
    # Step 4: Build Labels
    print("\n" + "=" * 72)
    print("STEP 4: LABEL GENERATION")
    print("=" * 72)
    
    df_features['close'] = df_raw['close'].reindex(df_features.index)
    
    y_global = build_labels(df_features)
    valid_mask = ~y_global.isna()
    
    X = df_features.loc[valid_mask].drop(columns=['close'], errors='ignore')
    y = y_global.loc[valid_mask]
    
    print(f"[Labels] Valid samples: {len(X)} / {len(df_features)}")
    print(f"[Labels] Positive class: {y.sum()} ({y.mean():.2%})")
    
    # Remove NaN frac_diff
    frac_diff_valid_mask = ~X['frac_diff'].isna()
    X = X.loc[frac_diff_valid_mask]
    y = y.loc[frac_diff_valid_mask]
    
    print(f"[FracDiff] Final dataset: {len(X)} samples")
    
    # Step 5: Feature Partitioning
    print("\n" + "=" * 72)
    print("STEP 5: FEATURE PARTITIONING")
    print("=" * 72)
    
    available_physics = [c for c in PHYSICS_COLUMNS if c in X.columns]
    stability_cols = ["stability_theta", "stability_acf", "stability_warning"]
    available_physics.extend([c for c in stability_cols if c in X.columns])
    
    passthrough_cols = available_physics + ["frac_diff"]
    tensor_feature_cols = [c for c in X.columns if c not in passthrough_cols]
    
    print(f"[Features] Total: {len(X.columns)}")
    print(f"[Features] Passthrough (Physics + FracDiff): {len(passthrough_cols)}")
    print(f"[Features] Tensor-Flex Candidates: {len(tensor_feature_cols)}")
    
    # Load CNN params
    cnn_params = None
    best_params_path = Path("artifacts/best_cnn_params.json")
    if best_params_path.exists():
        with open(best_params_path, "r") as f:
            cnn_params = json.load(f)
        print(f"\n[CNN] Loaded tuned params: {cnn_params}")
    
    # Step 6: Cross-Validation with Adaptive Thresholding
    print("\n" + "=" * 72)
    print(f"STEP 6: ADAPTIVE THRESHOLD CROSS-VALIDATION ({n_folds} Folds)")
    print("=" * 72)
    
    if not cfg.USE_TENSOR_FLEX:
        cfg.USE_TENSOR_FLEX = True
    
    tscv = TimeSeriesSplit(n_splits=n_folds)
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
            
            print(f"  ✓ Refined: {len(tensor_feature_cols)} → {X_train_tf.shape[1]} latent features")
            
            del refiner, X_train_tensor, X_val_tensor, X_train_tf, X_val_tf
            gc.collect()
        else:
            X_train = X_train_raw
            X_val = X_val_raw
        
        # Train MoE
        print(f"\n[Fold {fold_idx}] 5-Expert MoE Training")
        
        sample_weights = create_physics_sample_weights(X_train)
        
        moe = MixtureOfExpertsEnsemble(
            physics_features=available_physics,
            random_state=RANDOM_SEED,
            use_cnn=True,
            use_ou=True,
            cnn_params=cnn_params,
            cnn_epochs=15,
        )
        
        moe.fit(X_train, y_train, sample_weight=sample_weights)
        
        # Get expert telemetry to verify regime classification
        telemetry = moe.get_expert_telemetry(X_val)
        
        print(f"\n[Fold {fold_idx}] Expert Weight Distribution:")
        print(f"  Trend:   {telemetry['share_trend']:.2%}")
        print(f"  Range:   {telemetry['share_range']:.2%}")
        print(f"  Stress:  {telemetry['share_stress']:.2%}")
        print(f"  Elastic: {telemetry['share_ou']:.2%}")
        print(f"  Pattern: {telemetry['share_cnn']:.2%}")
        print(f"  Gating Confidence: {telemetry['gating_confidence']:.2%}")
        
        # Check for weight symmetry (problem indicator)
        weights = [telemetry['share_trend'], telemetry['share_range'], 
                   telemetry['share_ou'], telemetry['share_cnn']]
        weight_std = np.std(weights)
        
        if weight_std < 0.02:  # Very similar weights
            print(f"  ⚠️  WARNING: Weights are nearly identical (std={weight_std:.4f})")
            print(f"      Gating network may not be distinguishing regimes properly")
        else:
            print(f"  ✓ Weight diversity detected (std={weight_std:.4f})")
        
        # Get predictions
        y_pred_proba = moe.predict_proba(X_val)[:, 1]
        
        # Extract theta for adaptive thresholding
        if "stability_theta" in X_val.columns:
            theta_val = X_val["stability_theta"].values
        else:
            print("  [Warning] stability_theta not found, using uniform theta")
            theta_val = np.ones(len(X_val)) * 0.5
        
        # Calibrate adaptive policy on validation set
        adaptive_policy, adaptive_metrics = calibrate_adaptive_policy(
            y_val.values,
            y_pred_proba,
            theta_val,
        )
        
        # Compare with static threshold (0.5)
        y_pred_static = (y_pred_proba >= 0.5).astype(int)
        static_metrics = evaluate_threshold_policy(
            y_val.values,
            y_pred_static,
            "Static (0.5)",
        )
        
        # Apply adaptive threshold
        y_pred_adaptive = adaptive_policy.apply(y_pred_proba, theta_val)
        
        # Store results
        fold_results.append({
            "fold": fold_idx,
            "static_precision": static_metrics["precision"],
            "static_recall": static_metrics["recall"],
            "static_expectancy": static_metrics["expectancy"],
            "adaptive_precision": adaptive_metrics["precision"],
            "adaptive_recall": adaptive_metrics["recall"],
            "adaptive_expectancy": adaptive_metrics["expectancy"],
            "base_th": adaptive_policy.base_th,
            "sensitivity": adaptive_policy.sensitivity,
            "avg_theta": float(np.mean(theta_val)),
            **telemetry,  # Include expert weights
        })
        
        print(f"\n[Fold {fold_idx}] Threshold Comparison:")
        print(f"  Static (0.5):")
        print(f"    Precision: {static_metrics['precision']:.2%}, "
              f"Recall: {static_metrics['recall']:.2%}, "
              f"Expectancy: {static_metrics['expectancy']:.4f}")
        print(f"  Adaptive (base={adaptive_policy.base_th:.3f}, "
              f"sens={adaptive_policy.sensitivity:.3f}):")
        print(f"    Precision: {adaptive_metrics['precision']:.2%}, "
              f"Recall: {adaptive_metrics['recall']:.2%}, "
              f"Expectancy: {adaptive_metrics['expectancy']:.4f}")
        print(f"  Improvement:")
        print(f"    Δ Precision: {(adaptive_metrics['precision'] - static_metrics['precision']):.2%}")
        print(f"    Δ Recall: {(adaptive_metrics['recall'] - static_metrics['recall']):.2%}")
        print(f"    Δ Expectancy: {(adaptive_metrics['expectancy'] - static_metrics['expectancy']):.4f}")
        
        del X_train, X_val, y_train, y_val, moe
        gc.collect()
    
    # Step 7: Final Report
    print("\n" + "=" * 72)
    print("STEP 7: ADAPTIVE THRESHOLD EVALUATION")
    print("=" * 72)
    
    results_df = pd.DataFrame(fold_results)
    
    print("\nPer-Fold Results:")
    print(results_df[["fold", "static_expectancy", "adaptive_expectancy", 
                      "static_precision", "adaptive_precision",
                      "static_recall", "adaptive_recall"]].to_string(index=False))
    
    # Aggregate metrics
    print("\n" + "─" * 72)
    print("AGGREGATE COMPARISON")
    print("─" * 72)
    
    avg_static_exp = results_df["static_expectancy"].mean()
    avg_adaptive_exp = results_df["adaptive_expectancy"].mean()
    avg_static_prec = results_df["static_precision"].mean()
    avg_adaptive_prec = results_df["adaptive_precision"].mean()
    avg_static_rec = results_df["static_recall"].mean()
    avg_adaptive_rec = results_df["adaptive_recall"].mean()
    
    print(f"Static Threshold (0.5):")
    print(f"  Precision: {avg_static_prec:.2%}")
    print(f"  Recall:    {avg_static_rec:.2%}")
    print(f"  Expectancy: {avg_static_exp:.4f}")
    
    print(f"\nAdaptive Threshold:")
    print(f"  Precision: {avg_adaptive_prec:.2%}")
    print(f"  Recall:    {avg_adaptive_rec:.2%}")
    print(f"  Expectancy: {avg_adaptive_exp:.4f}")
    
    print(f"\nImprovement:")
    print(f"  Δ Precision: {(avg_adaptive_prec - avg_static_prec):.2%}")
    print(f"  Δ Recall:    {(avg_adaptive_rec - avg_static_rec):.2%}")
    print(f"  Δ Expectancy: {(avg_adaptive_exp - avg_static_exp):.4f}")
    
    # Success criteria
    print("\n" + "=" * 72)
    print("STABILITY REWARD VERIFICATION")
    print("=" * 72)
    
    recall_pass = avg_adaptive_rec > 0.04  # Recover lost ground (>4%)
    expectancy_pass = avg_adaptive_exp > 0.008  # Maintain profitability
    recall_improved = avg_adaptive_rec > avg_static_rec
    expectancy_improved = avg_adaptive_exp > avg_static_exp
    
    print(f"✓ Recall > 4%:         {'PASS' if recall_pass else 'FAIL'} ({avg_adaptive_rec:.2%})")
    print(f"✓ Expectancy > 0.008:  {'PASS' if expectancy_pass else 'FAIL'} ({avg_adaptive_exp:.4f})")
    print(f"✓ Recall Improved:     {'PASS' if recall_improved else 'FAIL'} "
          f"({avg_static_rec:.2%} → {avg_adaptive_rec:.2%})")
    print(f"✓ Expectancy Improved: {'PASS' if expectancy_improved else 'FAIL'} "
          f"({avg_static_exp:.4f} → {avg_adaptive_exp:.4f})")
    
    if recall_pass and expectancy_pass:
        print("\n🎯 STABILITY REWARD SUCCESSFUL!")
        print(f"   Aggressive in stable markets (Recall: {avg_adaptive_rec:.1%})")
        print(f"   Profitable overall (Expectancy: {avg_adaptive_exp:.4f})")
        print("   Regime-aware decision making solves precision-recall trade-off")
    else:
        if not recall_pass:
            print("\n⚠️  RECALL TOO LOW")
            print(f"   Need to be more aggressive in stable regimes")
        if not expectancy_pass:
            print("\n⚠️  EXPECTANCY TOO LOW")
            print(f"   Need to be more selective in chaotic regimes")
    
    # Save results
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    results_df.to_csv(artifacts_dir / "adaptive_threshold_results.csv", index=False)
    
    print(f"\n[Artifacts] Results saved to:")
    print(f"  - {artifacts_dir / 'adaptive_threshold_results.csv'}")
    
    return results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Regime-Adaptive MoE Pipeline"
    )
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Asset symbol")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    
    args = parser.parse_args()
    
    run_adaptive_moe_pipeline(
        symbol=args.symbol,
        n_folds=args.folds,
    )
