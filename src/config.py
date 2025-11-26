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
CACHE_DIR = Path(".")
MAX_FETCH_BATCHES = 10  # Safety net for paginated API calls

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
META_PROB_THRESHOLD = 0.65
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
