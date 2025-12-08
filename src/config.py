"""
Central configuration for the Quanta Futures research package.
Multi-Asset Sparse-Activated System Configuration.
"""

from pathlib import Path

# --- Multi-Asset Market Parameters -------------------------------------------
SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "XRPUSDT",
    "LTCUSDT",
    "DOGEUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "MATICUSDT",
    "LINKUSDT",
]
INTERVAL = "5"  # 5-minute candles for high-frequency multi-asset analysis
DAYS_BACK = 730  # Approx 2 years of data for deep history
HOURS_BACK = 0   # Additional hours (added to days)
MINUTES_BACK = 0 # Additional minutes (added to days + hours)
CACHE_DIR = Path(".")
MAX_FETCH_BATCHES = 10  # Safety net for paginated API calls


def get_lookback_timedelta(days=None, hours=None, minutes=None):
    """
    Compute total lookback timedelta from days/hours/minutes.
    
    Parameters
    ----------
    days : int, optional
        Number of days (default: DAYS_BACK)
    hours : int, optional
        Number of hours (default: HOURS_BACK)
    minutes : int, optional
        Number of minutes (default: MINUTES_BACK)
    
    Returns
    -------
    timedelta
        Total lookback period
    
    Examples
    --------
    >>> get_lookback_timedelta(days=7)
    timedelta(days=7)
    >>> get_lookback_timedelta(days=1, hours=12, minutes=30)
    timedelta(days=1, seconds=45000)
    """
    from datetime import timedelta
    
    d = days if days is not None else DAYS_BACK
    h = hours if hours is not None else HOURS_BACK
    m = minutes if minutes is not None else MINUTES_BACK
    
    return timedelta(days=d, hours=h, minutes=m)

# --- Multi-Asset Storage ------------------------------------------------------
MULTI_ASSET_CACHE = Path("multi_asset_cache.parquet")
TRAINING_SET = Path("multi_asset_training_data.parquet")

# --- Strategy parameters ------------------------------------------------------
LEVERAGE = 3
TP_PCT = 0.02  # +2% take-profit
SL_PCT = 0.01  # -1% stop-loss
BARRIER_HORIZON = 36  # bars evaluated by the triple-barrier logic

# --- Dynamic Strategy Settings ---
USE_DYNAMIC_TARGETS = True  # Set to False to use static fixed %
VOLATILITY_LOOKBACK = 14    # Period for ATR calculation (if not using pre-calculated)
TP_ATR_MULT = 2.5           # Take Profit = 2.5x ATR
SL_ATR_MULT = 1.0           # Stop Loss = 1.0x ATR

# --- Modeling ----------------------------------------------------------------
FEATURE_STORE = Path("btc_1000_features.parquet")
TOP_FEATURES = 25
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42
META_PROB_THRESHOLD = 0.55  # Sniper mode: Higher precision, lower recall
PRIMARY_RECALL_TARGET = 0.7

# --- Alpha Council Feature Budget -------------------------------------------
ALPHA_COUNCIL_MIN_FEATURES = 20
ALPHA_COUNCIL_MAX_FEATURES = 80
ALPHA_COUNCIL_FEATURE_PENALTY = 0.002  # expected Sharpe / expectancy gain needed per extra feature
ALPHA_COUNCIL_ENABLE_DYNAMIC_BUDGET = True

# --- Tensor-Flex Feature Refinement -----------------------------------------
USE_TENSOR_FLEX = False
TENSOR_FLEX_MAX_CLUSTER_SIZE = 64
TENSOR_FLEX_MAX_PAIRS_PER_CLUSTER = 5
TENSOR_FLEX_VARIANCE_THRESHOLD = 0.95
TENSOR_FLEX_N_SPLITS_STABILITY = 5
TENSOR_FLEX_RANDOM_STATE = 42
TENSOR_FLEX_STABILITY_THRESHOLD = 0.6
TENSOR_FLEX_SELECTOR_COEF_THRESHOLD = 1e-4
TENSOR_FLEX_SELECTOR_C = 0.1
TENSOR_FLEX_ARTIFACTS_DIR = Path("artifacts/tensor_flex")
TENSOR_FLEX_LOAD_IF_AVAILABLE = True

# Tensor-Flex latent selection
TENSOR_FLEX_MODE = "v2"  # "v1" or "v2"
TENSOR_FLEX_CORR_THRESHOLD = 0.85
TENSOR_FLEX_SUPERVISED_WEIGHT = 0.2
TENSOR_FLEX_MIN_LATENTS = 3
TENSOR_FLEX_MAX_LATENTS = 8
TENSOR_FLEX_VAR_EXPLAINED_MIN = 0.85  # cumulative variance threshold
TENSOR_FLEX_SHARPE_DELTA_MIN = 0.02   # minimum Sharpe/expectancy delta needed per extra latent
TENSOR_FLEX_ENABLE_DYNAMIC_LATENTS = True

# --- Temporal CNN Expert -----------------------------------------------------
CNN_USE = True
CNN_WINDOW_L = 64
CNN_C_MID = 128
CNN_HIDDEN = 64
CNN_DROPOUT = 0.2
CNN_LR = 1e-3
CNN_EPOCHS = 30
CNN_BATCH_SIZE = 64
CNN_RANDOM_STATE = 42
CNN_ARTIFACTS_DIR = Path("artifacts/cnn_expert")
CNN_FILL_EARLY = "pad_first_valid"  # or "nan"
CNN_LATENT_PREFIX = "cnn_latent__"

# --- Neural Architecture (Sparse-Activated System) ---------------------------
NUM_ASSETS = len(SYMBOLS)  # Number of assets for embedding layer
N_ASSETS = NUM_ASSETS      # Alias for consistency with new code
EMBEDDING_DIM = 16         # Dimension of asset embeddings
DROPOUT_RATE = 0.2         # Sparse activation dropout rate
MC_ITERATIONS = 10         # Monte Carlo inference iterations for uncertainty

# --- Training / Evaluation Protocol ------------------------------------------
CV_NUM_FOLDS = 1        # 1 means "no CV, single split" (current behavior)
CV_SCHEME = "expanding" # "expanding" or "rolling"
BOOTSTRAP_TRIALS = 0    # 0 disables bootstrap, >0 enables
BOOTSTRAP_SAMPLE_FRACTION = 0.7  # fraction of trades to sample with replacement
MIN_TRADES_FOR_EVAL = 200         # minimum trades to consider a fold valid

# --- Threshold Optimization Constraints --------------------------------------
THRESHOLD_MIN_TRADES = 300   # minimum trades to consider a threshold viable
THRESHOLD_MIN_RECALL = 0.03  # 3% recall minimum
THRESHOLD_GRID = (0.20, 0.70, 0.02) # start, end, step

# --- Visualization -----------------------------------------------------------
PLOT_TEMPLATE = "plotly_dark"

# --- Diffusion Denoiser Configuration ----------------------------------------
DIFFUSION_DENOISER = {
    'enabled': False,                    # Global on/off (start disabled to establish baseline)
    'target': 'frac_diff',               # 'frac_diff', 'log_returns', or 'full_features'
    'seq_len': 128,                      # Sequence length for model (128 H1 bars ≈ 5 days)
    'beta_schedule': 'cosine',           # 'linear', 'cosine', or 'quadratic'
    'num_timesteps': 1000,               # Training diffusion steps
    'inference_steps': 20,               # Reduced steps for fast inference
    'model_channels': 64,                # Base model dimension
    'num_res_blocks': 2,                 # Residual blocks per level
    'channel_mult': (1, 2, 4),           # Channel multipliers
    'time_dim': 128,                     # Time embedding dimension
    'dropout': 0.1,                      # Dropout rate
    'noise_level': 0.3,                  # Noise level for denoising (0-1)
    'checkpoint_path': 'artifacts/diffusion_denoiser/latest.ckpt',
    'artifacts_dir': Path('artifacts/diffusion_denoiser'),
}

# --- Diffusion Scenario Engine Configuration ---------------------------------
DIFFUSION_SCENARIO = {
    'enabled': False,                    # Start disabled
    'L_past': 96,                        # Past window length (conditioning)
    'H_future': 12,                      # Forecast horizon (12 H1 bars = 12 hours)
    'C_target': 1,                       # Target channels (1 = returns only)
    'beta_schedule': 'cosine',
    'num_timesteps': 1000,
    'inference_steps': 50,               # More steps for quality scenarios
    'model_channels': 64,
    'cond_dim': 128,                     # Conditioning embedding dimension
    'checkpoint_path': 'artifacts/diffusion_scenario/latest.ckpt',
    'artifacts_dir': Path('artifacts/diffusion_scenario'),
}

# --- Diffusion Expert (6th MoE Expert) Configuration ------------------------
DIFFUSION_EXPERT = {
    'enabled': False,                    # Add as 6th expert when True
    'num_scenarios': 32,                 # K scenarios per prediction
    'horizon': 12,                       # Forecast horizon in bars
    'use_calibration_head': True,        # Calibrate scenario-based P_up
    'calibration_hidden_dim': 32,        # Calibration MLP hidden size
    'use_denoised_features': False,      # Use denoised features as input
    'tail_risk_quantile': 0.05,          # For computing tail risk (5th percentile)
}

# --- Dual-Manifold Fusion Configuration --------------------------------------
DUAL_MANIFOLD = {
    'enabled': False,                    # Optional advanced gating
    'd_model': 64,                       # Embedding dimension
    'market_encoder': {
        'use_diffusion_denoiser': True,  # Use denoised features
        'use_sde': True,                 # Include LaP-SDE latents
        'use_tensor_flex': True,         # Include Tensor-Flex latents
        'num_layers': 1,                 # Self-attention layers
        'num_heads': 4,
        'use_gqa': False,                # Grouped-Query Attention
    },
    'cog_encoder': {
        'use_risk_token': True,          # Include risk engine state
        'use_meta_token': True,          # Include regime/meta state
        'num_layers': 1,
        'num_heads': 4,
        'use_gqa': False,
    },
    'fusion': {
        'num_layers': 2,                 # Cross-attention layers
        'num_heads': 4,
        'use_gqa': True,
        'num_kv_groups': 2,              # KV groups for GQA
        'dropout': 0.1,
    },
}

# --- MoE Gating Configuration ------------------------------------------------
MOE = {
    'gating_mode': 'classic',            # 'classic' or 'dual_manifold'
    'temperature': 1.0,                  # Softmax temperature
    'num_experts': 5,                    # 5 base experts (6 with Diffusion)
}

# --- Tuning & Production Configuration ---------------------------------------
TUNING = {
    'MODE': 'research',                  # 'research' or 'production'
    'METRICS_TARGETS': {
        'target_sharpe': 1.5,
        'max_allowed_drawdown': 0.25,
        'min_hit_rate': 0.45,
        'max_trades_per_day': 50,
        'max_leverage': 5.0,
        'max_position_duration_minutes': 1440,
    },
    'BACKTEST': {
        'transaction_costs': {
            'taker_fee_bps': 7,
            'maker_fee_bps': 2,
        },
        'slippage_model': 'fixed_bps',   # or 'volume_based'
        'slippage_bps': 3,
        'include_funding_fees': True,
    },
    'SAFETY': {
        'daily_loss_limit': 0.05,        # 5% daily loss triggers kill-switch
        'weekly_loss_limit': 0.10,       # 10% weekly loss
        'max_drawdown_since_start': 0.20,# 20% total drawdown
        'min_margin_buffer': 0.30,       # 30% margin buffer
    },
    'SWEEP': {
        'parameters_to_optimize': [
            'risk_engine.target_vol',
            'risk_engine.max_lev',
            'moe.temperature',
            'diffusion_denoiser.enabled',
            'diffusion_expert.enabled',
            'diffusion_expert.num_scenarios',
        ],
        'max_runs_per_asset': 50,
    },
}

