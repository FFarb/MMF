"""
Deep Quant Pipeline Orchestrator (Money Machine Edition).

This script is the single source of truth for the research stack:
1) Builds features with the Numba-accelerated physics engine.
2) Screens features with the Alpha Council voting protocol.
3) Trains the Mixture-of-Experts (Mixed Mode) ensemble.
4) Prints diagnostic statistics including the system complexity score.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from src.config import DAYS_BACK
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
    forward_return = df["close"].shift(-LABEL_LOOKAHEAD) / df["close"] - 1.0
    y = (forward_return > LABEL_THRESHOLD).astype(int)
    return y


def run_pipeline() -> None:
    print("=" * 72)
    print("                     MONEY MACHINE: DEEP RESEARCH PIPELINE")
    print("=" * 72)

    # ------------------------------------------------------------------ #
    # 1. Data + Physics Engine
    # ------------------------------------------------------------------ #
    print("\n[1] DATA & PHYSICS ENGINE")
    df = build_feature_dataset(days_back=DAYS_BACK, force_refresh=True)
    print("    Applying Numba-accelerated chaos metrics (windows: 100 & 200)...")
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
        *PHYSICS_COLUMNS,
    }
    candidates = [col for col in combined.columns if col not in exclude_cols]

    print(f"    Screening {len(candidates)} candidate features...")
    council = AlphaCouncil()
    candidate_matrix = combined[candidates]
    survivors = council.screen_features(candidate_matrix, y)
    print(f"    Council elected {len(survivors)} elite features.")

    final_features: List[str] = survivors + list(PHYSICS_COLUMNS)
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
    # 4. System Vital Signs
    # ------------------------------------------------------------------ #
    print("\n[4] SYSTEM VITAL SIGNS")
    complexity = moe.get_system_complexity()
    print(f"    Gating Network Params : {complexity['Gating_Network_Params']:,}")
    print(f"    Trend Expert Nodes    : {complexity['Trend_Expert_Nodes']:,}")
    print(f"    Range Memory Units    : {complexity['Range_Expert_Memory']:,}")
    print(f"    Stress Coefficients   : {complexity['Stress_Expert_Coefs']:,}")
    print("-" * 60)
    print(f"    TOTAL AI PARAMETERS   : {complexity['Total_System_Complexity']:,}")

    # ------------------------------------------------------------------ #
    # 5. Validation Snapshot
    # ------------------------------------------------------------------ #
    print("\n[5] VALIDATION SNAPSHOT")
    probs = moe.predict_proba(X_test)[:, 1]
    preds = (probs > 0.5).astype(int)
    report = classification_report(y_test, preds, digits=4)
    print(report)
    print("    Sample probabilities:", np.round(probs[:5], 4))

    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    summary_path = artifacts / "money_machine_snapshot.parquet"
    pd.DataFrame({"probability": probs, "target": y_test}).to_parquet(summary_path)
    print(f"    Stored validation snapshot at {summary_path}")


if __name__ == "__main__":
    run_pipeline()
