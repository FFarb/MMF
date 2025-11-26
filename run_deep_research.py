"""
Deep Quant Pipeline Orchestrator (Smart Horizon & Scout Assembly Edition).
Fixes MemoryError by reducing M5 history and selecting features BEFORE global merge.
"""
from __future__ import annotations

import argparse
import gc
import shutil
from pathlib import Path
from typing import List, Sequence, Optional

import numpy as np
import pandas as pd

# Import Config
from src.config import DAYS_BACK, SYMBOLS
import src.config as cfg
ASSET_LIST = SYMBOLS

from src.features import SignalFactory
from src.features.advanced_stats import apply_rolling_physics
from src.features.alpha_council import AlphaCouncil
from src.features.tensor_flex import TensorFlexFeatureRefiner
from src.models.moe_ensemble import MixtureOfExpertsEnsemble
from src.training.meta_controller import TrainingScheduler
from src.data_loader import MarketDataLoader
from src.models import SniperModelTrainer
# from src.analysis.threshold_tuner import run_tuning # Replaced by inline constrained tuner


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


def _load_or_fit_tensor_flex(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    force_retrain: bool = False,
) -> TensorFlexFeatureRefiner:
    """
    Load existing Tensor-Flex artifacts or fit a new refiner.
    """
    artifacts_dir = Path(cfg.TENSOR_FLEX_ARTIFACTS_DIR)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact_file = artifacts_dir / "tensor_flex.joblib"

    if cfg.TENSOR_FLEX_LOAD_IF_AVAILABLE and artifact_file.exists() and not force_retrain:
        try:
            refiner = TensorFlexFeatureRefiner.load(artifacts_dir)
            print(f"[Tensor-Flex] Loaded existing refiner from {artifact_file}")
            return refiner
        except Exception as exc:
            print(f"[Tensor-Flex] Failed to load existing Tensor-Flex artifacts ({exc}). Re-training...")

    refiner = TensorFlexFeatureRefiner(
        max_cluster_size=cfg.TENSOR_FLEX_MAX_CLUSTER_SIZE,
        max_pairs_per_cluster=cfg.TENSOR_FLEX_MAX_PAIRS_PER_CLUSTER,
        variance_threshold=cfg.TENSOR_FLEX_VARIANCE_THRESHOLD,
        n_splits_stability=cfg.TENSOR_FLEX_N_SPLITS_STABILITY,
        stability_threshold=cfg.TENSOR_FLEX_STABILITY_THRESHOLD,
        selector_coef_threshold=cfg.TENSOR_FLEX_SELECTOR_COEF_THRESHOLD,
        selector_c=cfg.TENSOR_FLEX_SELECTOR_C,
        random_state=cfg.TENSOR_FLEX_RANDOM_STATE,
        artifacts_dir=artifacts_dir,
    )
    refiner.fit(X_train, y_train)
    print(f"[Tensor-Flex] Trained refiner with {len(refiner.selected_feature_names_)} distilled features.")
    return refiner

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

def run_pipeline(
    use_tensor_flex: Optional[bool] = None,
    tensor_flex_train_only: bool = False,
    tensor_flex_force_retrain: bool = False,
) -> None:
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

    tensor_flex_enabled = cfg.USE_TENSOR_FLEX if use_tensor_flex is None else use_tensor_flex
    passthrough_cols: List[str] = []
    for col in ("asset_id",):
        if col in X.columns:
            passthrough_cols.append(col)
    for col in available_physics:
        if col in X.columns and col not in passthrough_cols:
            passthrough_cols.append(col)
    tensor_feature_cols = [col for col in X.columns if col not in passthrough_cols]
    tensor_flex_refiner: Optional[TensorFlexFeatureRefiner] = None

    if tensor_flex_enabled and tensor_feature_cols:
        print(f"[Tensor-Flex] Preparing distillation on {len(tensor_feature_cols)} raw features...")
        X_train_tensor = X_train[tensor_feature_cols].copy()
        X_test_tensor = X_test[tensor_feature_cols].copy()
        tensor_flex_refiner = _load_or_fit_tensor_flex(
            X_train_tensor,
            y_train,
            force_retrain=tensor_flex_force_retrain,
        )
        if tensor_flex_train_only:
            artifacts_dir = Path(cfg.TENSOR_FLEX_ARTIFACTS_DIR)
            print(f"[Tensor-Flex] Train-only mode complete. Artifacts stored at {artifacts_dir.resolve()}")
            cleanup_temp_dir()
            return
        X_train_tf = tensor_flex_refiner.transform(X_train_tensor, mode="selected")
        X_test_tf = tensor_flex_refiner.transform(X_test_tensor, mode="selected")
        cnn_train = None
        cnn_test = None
        if cfg.CNN_USE:
            X_train_tf_full = tensor_flex_refiner.transform(X_train_tensor, mode="full_latents")
            X_test_tf_full = tensor_flex_refiner.transform(X_test_tensor, mode="full_latents")
            cnn_train = X_train_tf_full.add_prefix(cfg.CNN_LATENT_PREFIX)
            cnn_test = X_test_tf_full.add_prefix(cfg.CNN_LATENT_PREFIX)
        passthrough_train = X_train[passthrough_cols].copy()
        passthrough_test = X_test[passthrough_cols].copy()
        X_train = pd.concat([X_train_tf, passthrough_train], axis=1)
        X_test = pd.concat([X_test_tf, passthrough_test], axis=1)
        if cnn_train is not None and cnn_test is not None:
            X_train = pd.concat([X_train, cnn_train], axis=1)
            X_test = pd.concat([X_test, cnn_test], axis=1)
            print(
                f"[Tensor-Flex] Routed {cnn_train.shape[1]} raw latents to the Temporal CNN expert "
                "alongside distilled controls."
            )
        elif cfg.CNN_USE:
            print("[Tensor-Flex] CNN expert requested but no Tensor-Flex latents are available. CNN disabled.")
        print(
            f"[Tensor-Flex] Distilled feature space: {X_train_tf.shape[1]} latents + "
            f"{len(passthrough_cols)} passthrough controls."
        )
    elif tensor_flex_enabled and not tensor_feature_cols:
        print("[Tensor-Flex] Skipping refinement because no eligible features remain after passthrough filtering.")
    elif tensor_flex_train_only:
        print("[Tensor-Flex] Train-only mode requested but Tensor-Flex is disabled in config. Exiting early.")
        cleanup_temp_dir()
        return
    elif cfg.CNN_USE:
        print("[CNN] Temporal CNN expert disabled because Tensor-Flex preprocessing is not active.")
    
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
    # ------------------------------------------------------------------ #
    # 5. VALIDATION & SNAPSHOT
    # ------------------------------------------------------------------ #
    print("\n[5] VALIDATION & SNAPSHOT")
    
    # If CV was used, we might have a model trained on last fold or full data.
    # For simplicity, we'll just use the returned model to predict on X_test (which is the last split).
    # In a rigorous CV setup, we'd aggregate OOF predictions.
    
    probs = moe.predict_proba(X_test)[:, 1]
    telemetry = moe.get_expert_telemetry(X_test)
    if telemetry:
        print(
            f"    [MoE] Gating Shares - Trend: {telemetry.get('share_trend', 0):.1%}, "
            f"Range: {telemetry.get('share_range', 0):.1%}, "
            f"Stress: {telemetry.get('share_stress', 0):.1%}, "
            f"CNN: {telemetry.get('share_cnn', 0):.1%} | "
            f"CNN weight mean: {telemetry.get('cnn_weight_mean', 0):.3f}"
        )
    
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    
    val_df = pd.DataFrame({"probability": probs, "target": y_test.values})
    val_path = artifacts / "money_machine_snapshot.parquet"
    val_df.to_parquet(val_path)
    print(f"    Snapshot saved.")
    
    # --- Constrained Threshold Optimization ---
    print("\n[6] CONSTRAINED THRESHOLD OPTIMIZATION")
    _run_constrained_tuning(val_df, artifacts)

    cleanup_temp_dir()

def _run_constrained_tuning(val_df: pd.DataFrame, output_dir: Path):
    """
    Finds optimal threshold with constraints on trade count and recall.
    """
    thresholds = np.arange(*cfg.THRESHOLD_GRID)
    results = []
    
    y_true = val_df["target"].values
    probs = val_df["probability"].values
    
    print(f"    Sweeping {len(thresholds)} thresholds ({thresholds[0]:.2f} - {thresholds[-1]:.2f})...")
    
    for t in thresholds:
        preds = (probs >= t).astype(int)
        n_trades = preds.sum()
        
        if n_trades == 0:
            results.append({
                "threshold": t, "precision": 0, "recall": 0, "f1": 0, 
                "trades": 0, "expectancy": 0, "sharpe_proxy": 0
            })
            continue
            
        prec = pd.Series(preds[preds==1] == y_true[preds==1]).mean() # Precision
        if np.isnan(prec): prec = 0.0
        
        rec = (preds * y_true).sum() / max(1, y_true.sum()) # Recall
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        
        # Expectancy
        exp = (prec * cfg.TP_PCT) - ((1 - prec) * cfg.SL_PCT)
        sharpe = exp * np.sqrt(n_trades)
        
        results.append({
            "threshold": t, 
            "precision": prec, 
            "recall": rec, 
            "f1": f1, 
            "trades": n_trades, 
            "expectancy": exp, 
            "sharpe_proxy": sharpe
        })
        
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "threshold_sweep_results.csv", index=False)
    
    # Apply Constraints
    mask = (results_df["trades"] >= cfg.THRESHOLD_MIN_TRADES) & (results_df["recall"] >= cfg.THRESHOLD_MIN_RECALL)
    candidates = results_df[mask]
    
    if candidates.empty:
        print("    [WARNING] No thresholds satisfied constraints! Falling back to global max Sharpe.")
        best_row = results_df.loc[results_df["sharpe_proxy"].idxmax()]
    else:
        best_row = candidates.loc[candidates["sharpe_proxy"].idxmax()]
        
    print(f"    Optimal Threshold: {best_row['threshold']:.2f}")
    print(f"    Trades: {best_row['trades']} | Precision: {best_row['precision']:.2%} | Recall: {best_row['recall']:.2%}")
    print(f"    Expectancy: {best_row['expectancy']:.4f} | Sharpe Proxy: {best_row['sharpe_proxy']:.4f}")
    
    # Simple HTML Report
    html = f"""
    <html><body>
    <h2>Threshold Optimization Report</h2>
    <p><b>Constraints:</b> Min Trades={cfg.THRESHOLD_MIN_TRADES}, Min Recall={cfg.THRESHOLD_MIN_RECALL:.1%}</p>
    <p><b>Selected:</b> Threshold={best_row['threshold']:.2f}, Trades={best_row['trades']}, Sharpe={best_row['sharpe_proxy']:.2f}</p>
    <table border="1">
    <tr><th>Threshold</th><th>Trades</th><th>Precision</th><th>Recall</th><th>Expectancy</th><th>Sharpe</th></tr>
    """
    for _, row in results_df.iterrows():
        style = "background-color: #e0ffe0;" if row['threshold'] == best_row['threshold'] else ""
        html += f"<tr style='{style}'><td>{row['threshold']:.2f}</td><td>{row['trades']}</td><td>{row['precision']:.2%}</td><td>{row['recall']:.2%}</td><td>{row['expectancy']:.4f}</td><td>{row['sharpe_proxy']:.2f}</td></tr>"
    html += "</table></body></html>"
    
    with open(output_dir / "threshold_optimization.html", "w") as f:
        f.write(html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the QFC deep research pipeline.")
    parser.add_argument("--use-tensor-flex", action="store_true", help="Force Tensor-Flex preprocessing.")
    parser.add_argument("--no-tensor-flex", action="store_true", help="Disable Tensor-Flex preprocessing.")
    parser.add_argument("--tensor-flex-train-only", action="store_true", help="Fit Tensor-Flex artifacts only.")
    parser.add_argument("--tensor-flex-force-retrain", action="store_true", help="Ignore cached Tensor-Flex artifacts.")
    
    # New Args
    parser.add_argument("--cv-folds", type=int, default=cfg.CV_NUM_FOLDS, help="Number of CV folds.")
    parser.add_argument("--bootstrap-trials", type=int, default=cfg.BOOTSTRAP_TRIALS, help="Number of bootstrap trials.")
    parser.add_argument("--threshold-min-trades", type=int, default=cfg.THRESHOLD_MIN_TRADES, help="Min trades for threshold tuning.")
    
    args = parser.parse_args()

    # Apply Overrides
    if args.cv_folds is not None: cfg.CV_NUM_FOLDS = args.cv_folds
    if args.bootstrap_trials is not None: cfg.BOOTSTRAP_TRIALS = args.bootstrap_trials
    if args.threshold_min_trades is not None: cfg.THRESHOLD_MIN_TRADES = args.threshold_min_trades

    use_tensor_flex_arg: Optional[bool] = None
    if args.use_tensor_flex:
        use_tensor_flex_arg = True
    elif args.no_tensor_flex:
        use_tensor_flex_arg = False

    run_pipeline(
        use_tensor_flex=use_tensor_flex_arg,
        tensor_flex_train_only=args.tensor_flex_train_only,
        tensor_flex_force_retrain=args.tensor_flex_force_retrain,
    )
