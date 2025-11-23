"""
Deep Quant Pipeline Orchestrator (Smart Horizon & Scout Assembly Edition).
Fixes MemoryError by reducing M5 history and selecting features BEFORE global merge.
"""
from __future__ import annotations

import gc
import shutil
from pathlib import Path
from typing import List, Sequence, Optional

import numpy as np
import pandas as pd

# Import Config
from src.config import DAYS_BACK, SYMBOLS, CACHE_DIR
ASSET_LIST = SYMBOLS

from src.features import SignalFactory
from src.features.advanced_stats import apply_rolling_physics
from src.features.alpha_council import AlphaCouncil
from src.models.moe_ensemble import MixtureOfExpertsEnsemble
from src.training.meta_controller import TrainingScheduler
from src.data_loader import MarketDataLoader
from src.analysis.threshold_tuner import run_tuning

PHYSICS_COLUMNS: Sequence[str] = ("hurst_200", "entropy_200", "fdi_200")
LABEL_LOOKAHEAD = 36
LABEL_THRESHOLD = 0.005
TEMP_DIR = Path("temp_processed_assets")

# --- Config Overrides for Memory Safety ---
M5_LOOKBACK_DAYS = 180  # Limit high-freq data to last 6 months
H1_LOOKBACK_DAYS = DAYS_BACK # Keep deep history for context

def _build_labels(df: pd.DataFrame) -> pd.Series:
    if 'asset_id' in df.columns:
        forward_ret = df.groupby('asset_id')['close'].shift(-LABEL_LOOKAHEAD) / df['close'] - 1.0
    else:
        forward_ret = df['close'].shift(-LABEL_LOOKAHEAD) / df['close'] - 1.0
    y = (forward_ret > LABEL_THRESHOLD).astype(int)
    return y

def cleanup_temp_dir():
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(exist_ok=True)

def get_smart_data(loader: MarketDataLoader, symbol: str, interval: str, days: int) -> pd.DataFrame:
    loader.symbol = symbol
    loader.interval = interval
    try:
        df = loader.get_data(days_back=days)
        return df
    except Exception as e:
        print(f"       [WARNING] Could not fetch {symbol} {interval}: {e}")
        return pd.DataFrame()

def process_single_asset(symbol: str, asset_idx: int, loader: MarketDataLoader, factory: SignalFactory) -> Optional[pd.DataFrame]:
    """
    Loads, generates features, and merges M5/H1 data for a single asset.
    """
    try:
        print(f"\n    >> Processing {symbol} (ID: {asset_idx})...")
        
        # A. Smart Horizon Loading
        df_m5 = get_smart_data(loader, symbol, "5", M5_LOOKBACK_DAYS)
        df_h1 = get_smart_data(loader, symbol, "60", H1_LOOKBACK_DAYS)
        
        if df_m5.empty or df_h1.empty:
            return None

        # B. Feature Generation
        print(f"       Generating Strategic (H1) features...")
        df_h1_features = factory.generate_signals(df_h1)
        df_h1_features = df_h1_features.add_prefix("macro_")
        
        print(f"       Generating Execution (M5) features...")
        df_m5_features = factory.generate_signals(df_m5)
        
        # C. Fractal Merge
        df_m5_features = df_m5_features.sort_index()
        df_h1_features = df_h1_features.sort_index()
        
        df_merged = pd.merge_asof(
            df_m5_features,
            df_h1_features,
            left_index=True,
            right_index=True,
            direction='backward'
        )
        
        # D. Physics
        print(f"       Applying Chaos Physics...")
        df_merged = apply_rolling_physics(df_merged, windows=[100, 200])
        df_merged['asset_id'] = asset_idx
        
        # E. Optimization
        df_merged = df_merged.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Enforce float32
        cols = df_merged.select_dtypes(include=['float64']).columns
        df_merged[cols] = df_merged[cols].astype('float32')
        
        return df_merged
        
    except Exception as e:
        print(f"       [ERROR] {symbol}: {e}")
        return None

def run_pipeline() -> None:
    print("=" * 72)
    print("          MULTI-ASSET NEURO-SYMBOLIC TRADING SYSTEM")
    print("          (Smart Horizon & Scout Assembly Mode)")
    print("=" * 72)

    cleanup_temp_dir()
    loader = MarketDataLoader(interval="5")
    factory = SignalFactory()
    
    generated_files = []
    scout_features: List[str] = []
    
    # ------------------------------------------------------------------ #
    # 1. SCOUT PHASE (First Asset Only)
    # ------------------------------------------------------------------ #
    print("\n[1] SCOUT PHASE (Feature Selection on Leader)")
    scout_symbol = ASSET_LIST[0]
    df_scout = process_single_asset(scout_symbol, 0, loader, factory)
    
    if df_scout is None:
        print("CRITICAL: Scout failed. Exiting.")
        return

    # Run Alpha Council on Scout
    print(f"    Running Alpha Council on Scout ({scout_symbol})...")
    y_scout = _build_labels(df_scout)
    valid_mask = ~y_scout.isna()
    
    # Filter for Council
    df_council = df_scout.loc[valid_mask].copy()
    y_council = y_scout.loc[valid_mask]
    
    exclude = {"open", "high", "low", "close", "volume", "timestamp", "target", "asset_id", *PHYSICS_COLUMNS}
    candidates = [c for c in df_council.columns if c not in exclude]
    
    council = AlphaCouncil()
    selected_alphas = council.screen_features(df_council[candidates], y_council, n_features=25)
    
    # Define Final Schema
    available_physics = [c for c in PHYSICS_COLUMNS if c in df_scout.columns]
    final_schema = selected_alphas + available_physics + ['asset_id', 'close']
    scout_features = final_schema
    
    print(f"    SCOUT SELECTED {len(selected_alphas)} FEATURES: {selected_alphas}")
    
    # Save Scout to Disk (using only filtered schema to save space)
    save_path = TEMP_DIR / f"{scout_symbol}.parquet"
    df_scout[final_schema].to_parquet(save_path, compression='snappy')
    generated_files.append(save_path)
    
    del df_scout, df_council, y_council
    gc.collect()

    # ------------------------------------------------------------------ #
    # 2. FLEET PHASE (Process remaining assets using Scout Schema)
    # ------------------------------------------------------------------ #
    print("\n[2] FLEET PHASE (Processing Remaining Assets)")
    
    for asset_idx, symbol in enumerate(ASSET_LIST[1:], start=1):
        df_asset = process_single_asset(symbol, asset_idx, loader, factory)
        
        if df_asset is not None:
            # Filter columns IMMEDIATELY before saving
            # This ensures the disk files are small and compatible
            try:
                # Ensure all columns exist (fill 0 if missing, though unlikely if logic is same)
                for col in scout_features:
                    if col not in df_asset.columns:
                        df_asset[col] = 0.0
                
                df_filtered = df_asset[scout_features]
                
                save_path = TEMP_DIR / f"{symbol}.parquet"
                df_filtered.to_parquet(save_path, compression='snappy')
                generated_files.append(save_path)
                print(f"       -> Saved filtered shard: {save_path}")
                
            except Exception as e:
                print(f"       [ERROR] Saving shard {symbol}: {e}")
        
        del df_asset
        gc.collect()

    # ------------------------------------------------------------------ #
    # 3. GLOBAL ASSEMBLY
    # ------------------------------------------------------------------ #
    # NOTE: Current approach uses pd.concat to create a DataFrame.
    # This is compatible with GraphVisionary's Global Mode via TorchSklearnWrapper.
    # 
    # FUTURE MIGRATION: Can replace with make_global_loader() to get a DataLoader
    # that yields 4D tensors (Batch, Seq, Assets, Features) directly.
    # See: src.data_loader.make_global_loader(TEMP_DIR, batch_size=32, sequence_length=16)
    # ------------------------------------------------------------------ #
    print("\n[3] GLOBAL ASSEMBLY")
    dfs = []
    for f in generated_files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"    [WARN] Corrupt shard {f}: {e}")

    if not dfs:
        return

    df_global = pd.concat(dfs).sort_index()
    print(f"    Global Tensor Assembled: {df_global.shape}")

    # ------------------------------------------------------------------ #
    # 4. TRAINING
    # ------------------------------------------------------------------ #
    # NOTE: MoE Ensemble uses HybridTrendExpert, which internally uses GraphVisionary.
    # The DataFrame X_train is automatically reshaped by TorchSklearnWrapper.fit():
    #   - Detects if samples are divisible by n_assets (Global Mode)
    #   - Reshapes (Batch*Assets, Seq*Features) -> (Batch, Seq, Assets, Features)
    #   - Passes 4D tensor to GraphVisionary for cross-asset attention
    # 
    # Current shape: (N_Samples, N_Features) where N_Samples = Time * N_Assets
    # GraphVisionary will reconstruct the temporal and asset dimensions internally.
    # ------------------------------------------------------------------ #
    print("\n[4] MIXED MODE TRAINING")
    
    y_global = _build_labels(df_global)
    valid = ~y_global.isna()
    
    # X is already filtered to (survivors + physics + asset_id)
    X = df_global.loc[valid].drop(columns=['close'], errors='ignore')
    y = y_global.loc[valid]
    
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    scheduler = TrainingScheduler()
    e_col = "entropy_200" if "entropy_200" in X.columns else X.columns[0]
    v_col = "fdi_200" if "fdi_200" in X.columns else X.columns[0]
    
    # Safe signal extraction
    if len(X_train) > 1000:
        e_sig = float(X_train[e_col].iloc[-1000:].mean())
        v_sig = float(X_train[v_col].iloc[-1000:].mean())
    else:
        e_sig, v_sig = 0.5, 1.5
        
    depth = scheduler.suggest_training_depth(e_sig, v_sig)
    print(f"    Training Config: {depth}")
    
    moe = MixtureOfExpertsEnsemble(
        physics_features=available_physics,
        random_state=42,
        trend_estimators=depth["n_estimators"],
        gating_epochs=depth["epochs"],
    )
    
    moe.fit(X_train, y_train)
    
    # ------------------------------------------------------------------ #
    # 5. VALIDATION
    # ------------------------------------------------------------------ #
    print("\n[5] VALIDATION & SNAPSHOT")
    probs = moe.predict_proba(X_test)[:, 1]
    
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    
    val_df = pd.DataFrame({"probability": probs, "target": y_test.values})
    val_path = artifacts / "money_machine_snapshot.parquet"
    val_df.to_parquet(val_path)
    print(f"    Snapshot saved.")
    
    run_tuning(validation_path=val_path, output_dir=artifacts)
    cleanup_temp_dir()

if __name__ == "__main__":
    run_pipeline()
