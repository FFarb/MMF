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
DAYS_BACK = 200  # Approx 6 months of 5m data (~6M data points total across assets)
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

# --- Neural Architecture (Sparse-Activated System) ---------------------------
NUM_ASSETS = len(SYMBOLS)  # Number of assets for embedding layer
EMBEDDING_DIM = 16         # Dimension of asset embeddings
DROPOUT_RATE = 0.2         # Sparse activation dropout rate
MC_ITERATIONS = 10         # Monte Carlo inference iterations for uncertainty

# --- Visualization -----------------------------------------------------------
PLOT_TEMPLATE = "plotly_dark"
