"""
Deep Quant Pipeline Orchestrator (Multi-Asset Bicameral Edition).

This script implements the Neuro-Symbolic "Bicameral" Trading System with multi-asset support:
1) Loads multi-asset data with asset_id tracking
2) Builds features with the Numba-accelerated physics engine (per-asset)
3) Screens features with the Alpha Council voting protocol
4) Trains the Bicameral Hybrid Ensemble (Symbolic + Neural + Meta-Learner)
5) Optimizes trading threshold using Sharpe Proxy metric
6) Reports per-coin performance metrics
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, precision_score, recall_score

from src.analysis.threshold_tuner import run_tuning
from src.config import DAYS_BACK, SYMBOLS
from src.data_loader import MarketDataLoader
from src.features import build_feature_dataset
from src.features.advanced_stats import apply_rolling_physics
from src.features.alpha_council import AlphaCouncil
from src.models.moe_ensemble import MixtureOfExpertsEnsemble
from src.training.meta_controller import TrainingScheduler


PHYSICS_COLUMNS: Sequence[str] = ("hurst_200", "entropy_200", "fdi_200")
LABEL_LOOKAHEAD = 36
LABEL_THRESHOLD = 0.005


def _validate_physics_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Physics engine is missing required columns: {missing}. "
            "Ensure apply_rolling_physics() ran successfully."
        )


def _build_labels(df: pd.DataFrame) -> pd.Series:
    """Build labels per-asset to respect boundaries."""
    if 'asset_id' in df.columns:
        # Multi-asset: calculate per asset
        labels = []
        for asset_id in sorted(df['asset_id'].unique()):
            asset_df = df[df['asset_id'] == asset_id].copy()
            forward_return = asset_df["close"].shift(-LABEL_LOOKAHEAD) / asset_df["close"] - 1.0
            y = (forward_return > LABEL_THRESHOLD).astype(int)
            labels.append(y)
        return pd.concat(labels).sort_index()
    else:
        # Single-asset
        forward_return = df["close"].shift(-LABEL_LOOKAHEAD) / df["close"] - 1.0
        y = (forward_return > LABEL_THRESHOLD).astype(int)
        return y


def run_pipeline() -> None:
    print("=" * 72)
    print("     MULTI-ASSET NEURO-SYMBOLIC BICAMERAL TRADING SYSTEM")
    print("=" * 72)

    # ------------------------------------------------------------------ #
    # 1. Multi-Asset Data + Physics Engine
    # ------------------------------------------------------------------ #
    print("\n[1] MULTI-ASSET DATA & PHYSICS ENGINE")
    
    # Load multi-asset data
    loader = MarketDataLoader()
    print(f"    Loading {len(SYMBOLS)} assets with {DAYS_BACK} days of 5-minute data...")
    df_raw = loader.fetch_all_assets(days_back=DAYS_BACK, force_refresh=False)
    print(f"    Loaded {len(df_raw)} total rows across {df_raw['asset_id'].nunique()} assets")
    
    # Build features (will be added per-asset in future enhancement)
    # For now, we'll use the raw OHLCV data
    df = df_raw.copy()
    
    print("    Applying Numba-accelerated chaos metrics (windows: 100 & 200)...")
    print("    [CRITICAL] Calculating per-asset to prevent feature bleeding...")
    df = apply_rolling_physics(df, windows=[100, 200])
    _validate_physics_columns(df, PHYSICS_COLUMNS)

    # ------------------------------------------------------------------ #
    # 2. Alpha Council Feature Screening
    # ------------------------------------------------------------------ #
    print("\n[2] ALPHA COUNCIL (Feature Selection)")
    raw_labels = _build_labels(df)
    combined = pd.concat([df, raw_labels.rename("target")], axis=1)
    combined = combined.dropna(axis=0)

    y = combined["target"].astype(int)
    exclude_cols = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "timestamp",
        "target",
        "asset_id",  # Exclude from screening but keep in dataset
        "symbol",
        *PHYSICS_COLUMNS,
    }
    candidates = [col for col in combined.columns if col not in exclude_cols]

    print(f"    Screening {len(candidates)} candidate features...")
    council = AlphaCouncil()
    candidate_matrix = combined[candidates]
    survivors = council.screen_features(candidate_matrix, y)
    print(f"    Council elected {len(survivors)} elite features.")

    final_features: List[str] = survivors + list(PHYSICS_COLUMNS)
    
    # Ensure asset_id is passed to model if present
    if "asset_id" in combined.columns:
        final_features.append("asset_id")
        
    X = combined[final_features]
    _validate_physics_columns(X, PHYSICS_COLUMNS)

    # ------------------------------------------------------------------ #
    # 3. Mixed Mode Training
    # ------------------------------------------------------------------ #
    print("\n[3] MIXED MODE TRAINING (Mixture-of-Experts)")
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scheduler = TrainingScheduler()
    entropy_signal = float(X["entropy_200"].iloc[-1])
    volatility_signal = float(
        X["fdi_200"].iloc[-1]
        if "volatility_200" not in X.columns
        else X["volatility_200"].iloc[-1]
    )
    depth = scheduler.suggest_training_depth(entropy_signal, max(volatility_signal, 1e-6))
    print(f"    Meta-Controller recommends: {depth}")

    moe = MixtureOfExpertsEnsemble(
        physics_features=PHYSICS_COLUMNS,
        random_state=42,
        trend_estimators=depth["n_estimators"],
        gating_epochs=depth["epochs"],
    )
    print("    Training Trend / Range / Stress experts with gating network...")
    moe.fit(X_train, y_train)

    # ------------------------------------------------------------------ #
    # 4. System Architecture Report
    # ------------------------------------------------------------------ #
    print("\n[4] SYSTEM ARCHITECTURE")
    print("    Visionary Architecture: 7-Scale Dynamic Convolution (Kernels: 3-129)")
    print("    Analyst Architecture  : Gradient Boosting Decision Trees")
    print("    Arbitration          : Meta-Learner (Volatility-Aware Stacking)")
    
    complexity = moe.get_system_complexity()
    print("\n    System Complexity:")
    print(f"      Gating Network       : {complexity['Gating_Network_Params']:,} params")
    print(f"      Bicameral Trend Expert: {complexity['Bicameral_Trend_Expert']:,} params")
    print(f"      Range Expert (kNN)   : {complexity['Range_Expert_Memory']:,} samples")
    print(f"      Stress Expert (LR)   : {complexity['Stress_Expert_Coefs']:,} coefs")
    print("    " + "-" * 60)
    print(f"      TOTAL PARAMETERS     : {complexity['Total_System_Complexity']:,}")

    # ------------------------------------------------------------------ #
    # 5. Validation & Threshold Optimization
    # ------------------------------------------------------------------ #
    print("\n[5] VALIDATION & THRESHOLD OPTIMIZATION")
    probs = moe.predict_proba(X_test)[:, 1]
    preds = (probs > 0.5).astype(int)
    
    print("\n    GLOBAL PERFORMANCE:")
    report = classification_report(y_test, preds, digits=4)
    print(report)
    
    # Per-coin reporting
    if "asset_id" in X_test.columns:
        print("\n    PER-COIN PERFORMANCE:")
        print("    " + "-" * 60)
        print(f"    {'SYMBOL':<10} | {'PRECISION':<10} | {'RECALL':<10} | {'SAMPLES':<10}")
        print("    " + "-" * 60)
        
        unique_assets = sorted(X_test["asset_id"].unique())
        for asset_id in unique_assets:
            # Map asset_id back to symbol if possible (using SYMBOLS from config)
            symbol = SYMBOLS[int(asset_id)] if int(asset_id) < len(SYMBOLS) else f"Asset {asset_id}"
            
            mask = X_test["asset_id"] == asset_id
            if mask.sum() > 0:
                y_coin = y_test[mask]
                preds_coin = preds[mask]
                
                prec = precision_score(y_coin, preds_coin, zero_division=0)
                rec = recall_score(y_coin, preds_coin, zero_division=0)
                count = len(y_coin)
                
                print(f"    {symbol:<10} | {prec:.2%}    | {rec:.2%}    | {count:<10}")
        print("    " + "-" * 60)

    print("    Sample probabilities:", np.round(probs[:5], 4))

    # Save validation predictions for threshold tuning
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    validation_path = artifacts / "validation_predictions.parquet"
    pd.DataFrame({"probability": probs, "target": y_test.values}).to_parquet(validation_path)
    print(f"\n    Saved validation predictions to: {validation_path}")
    
    # Run threshold optimization
    print("\n    Running Sharpe Proxy threshold optimization...")
    optimal_results = run_tuning(
        validation_path=validation_path,
        output_dir=artifacts,
        avg_win_pct=0.02,
        avg_loss_pct=-0.01,
    )
    
    print("\n" + "=" * 72)
    print("                         SYSTEM READY")
    print("=" * 72)
    print(f"  Optimal Trading Threshold: {optimal_results['optimal_threshold']:.2f}")
    print(f"  Sharpe Proxy            : {optimal_results['optimal_sharpe_proxy']:.4f}")
    print(f"  Expected Precision      : {optimal_results['precision']:.2%}")
    print(f"  Expected Recall         : {optimal_results['recall']:.2%}")
    print("=" * 72)


if __name__ == "__main__":
    run_pipeline()
