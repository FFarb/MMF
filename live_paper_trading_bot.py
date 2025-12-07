'''Live Paper Trading Bot (Dry Run) for Triple HMM Strategy

This script runs indefinitely, fetching live market data every 5 minutes, training three separate
Hidden Markov Models (HMM) on 1‑hour, 15‑minute and 5‑minute candles for the BTCUSDT pair, and
generating BUY/SELL signals based on the confluence of the three models.

Dependencies:
    - pybit
    - pandas
    - numpy
    - hmmlearn
    - time, datetime, os, csv, logging

The script is safe to run – it only logs signals to `paper_trade_log.csv` and never places real
orders. API errors are caught so the bot continues running.
'''

import os
import time
import datetime as dt
import logging
import csv
from pathlib import Path

import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM

# pybit – Bybit HTTP client (works for BTCUSDT)
from pybit.unified_trading import HTTP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SYMBOL = "BTCUSDT"
INTERVALS = {"H1": "60", "M15": "15", "M5": "5"}  # minutes as strings for pybit
CANDLE_LIMIT = 500  # number of candles to fetch per interval
LOG_FILE = Path(__file__).parent / "paper_trade_log.csv"

# Initialise logger – prints a nice dashboard to console and also writes to file
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Initialise Bybit client (public endpoint, no auth needed for market data)
client = HTTP(testnet=False)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def wait_until_next_candle():
    """Sleep until the exact start of the next 5‑minute candle.
    
    The function calculates how many seconds remain until the next timestamp that is a multiple
    of 5 minutes (e.g., xx:00, xx:05, xx:10 …) and sleeps for that duration.
    """
    now = dt.datetime.utcnow()
    # Align to 5‑minute grid
    next_minute = (now.minute // 5 + 1) * 5
    # Handle hour overflow
    if next_minute >= 60:
        next_hour = now.replace(minute=0, second=0, microsecond=0) + dt.timedelta(hours=1)
    else:
        next_hour = now.replace(minute=next_minute, second=0, microsecond=0)
    seconds_to_wait = (next_hour - now).total_seconds()
    logger.info(f"Waiting {seconds_to_wait:.1f}s until next 5‑minute candle at {next_hour.isoformat()}Z")
    time.sleep(seconds_to_wait)

def fetch_candles(interval_minutes: str) -> pd.DataFrame:
    """Fetch the most recent `CANDLE_LIMIT` candles for the given interval.
    
    Returns a DataFrame with a datetime index and columns: open, high, low, close, volume.
    """
    try:
        # New pybit unified_trading API requires category parameter
        resp = client.get_kline(
            category="linear",  # for USDT perpetual futures
            symbol=SYMBOL,
            interval=interval_minutes,
            limit=CANDLE_LIMIT
        )
        
        # Check response status
        if resp.get("retCode") != 0:
            raise ValueError(f"API error: {resp.get('retMsg', 'Unknown error')}")
        
        # Data is now in result['list'] as arrays: [timestamp, open, high, low, close, volume, turnover]
        data_list = resp.get("result", {}).get("list", [])
        if not data_list:
            raise ValueError("Empty result from API")
        
        # Convert list of arrays to DataFrame
        df = pd.DataFrame(data_list, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        
        # Convert timestamp (ms) to datetime UTC
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        
        # Keep only the columns we need and ensure numeric types
        df = df[["open", "high", "low", "close", "volume"]]
        df = df.astype(float)
        
        # Remove possible duplicate timestamps
        df = df[~df.index.duplicated(keep="last")]
        
        # Sort by timestamp (API returns newest first, we want oldest first)
        df = df.sort_index()
        
        return df
    except Exception as e:
        logger.error(f"Error fetching {interval_minutes}‑min candles: {e}")
        # Return an empty DataFrame so the caller can decide what to do
        return pd.DataFrame()

def train_hmm(df: pd.DataFrame, n_states: int = 3) -> GaussianHMM:
    """Train a Gaussian HMM on log‑returns of the close price.
    
    The model is fitted on a single feature – the log return series.
    """
    # Compute log returns
    returns = np.log(df["close"].pct_change().dropna()).values.reshape(-1, 1)
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(returns)
    return model

def label_states(model: GaussianHMM, df: pd.DataFrame) -> dict:
    """Assign semantic labels (BULL, BEAR, NEUTRAL) to each hidden state.
    
    Returns a mapping from state index to label.
    """
    returns = np.log(df["close"].pct_change().dropna()).values.reshape(-1, 1)
    hidden_states = model.predict(returns)
    # Compute mean return per state
    state_means = []
    for i in range(model.n_components):
        state_means.append(returns[hidden_states == i].mean())
    # Identify max, min, middle
    max_idx = int(np.argmax(state_means))
    min_idx = int(np.argmin(state_means))
    # The remaining index is neutral
    neutral_idx = next(i for i in range(model.n_components) if i not in (max_idx, min_idx))
    mapping = {max_idx: "BULL", min_idx: "BEAR", neutral_idx: "NEUTRAL"}
    return mapping

def latest_state(model: GaussianHMM, df: pd.DataFrame) -> int:
    """Return the most recent hidden state index based on the latest candle."""
    returns = np.log(df["close"].pct_change().dropna()).values.reshape(-1, 1)
    hidden_states = model.predict(returns)
    return int(hidden_states[-1])

def ensure_log_file():
    if not LOG_FILE.exists():
        with LOG_FILE.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "price", "state_h1", "state_m15", "state_m5", "signal"])

def log_signal(timestamp: dt.datetime, price: float, s_h1: str, s_m15: str, s_m5: str, signal: str):
    ensure_log_file()
    with LOG_FILE.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp.isoformat(), f"{price:.2f}", s_h1, s_m15, s_m5, signal])

def dashboard(timestamp: dt.datetime, price: float, s_h1: str, s_m15: str, s_m5: str, signal: str):
    logger.info("""
[{ts}]
H1: {h1} | M15: {m15} | M5: {m5}
DECISION: {dec}
Price: {price:.2f}
""".format(
        ts=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        h1=s_h1, m15=s_m15, m5=s_m5,
        dec=signal, price=price
    ))

def main_loop():
    logger.info("Starting Live Paper Trading Bot (dry‑run)…")
    while True:
        try:
            wait_until_next_candle()
            # Fetch data for each timeframe
            df_h1 = fetch_candles(INTERVALS["H1"])
            df_m15 = fetch_candles(INTERVALS["M15"])
            df_m5 = fetch_candles(INTERVALS["M5"])
            if df_h1.empty or df_m15.empty or df_m5.empty:
                logger.warning("One or more data frames are empty – skipping this cycle.")
                continue
            # Train HMMs
            hmm_h1 = train_hmm(df_h1)
            hmm_m15 = train_hmm(df_m15)
            hmm_m5 = train_hmm(df_m5)
            # Label states
            label_h1 = label_states(hmm_h1, df_h1)
            label_m15 = label_states(hmm_m15, df_m15)
            label_m5 = label_states(hmm_m5, df_m5)
            # Get latest state indices
            state_idx_h1 = latest_state(hmm_h1, df_h1)
            state_idx_m15 = latest_state(hmm_m15, df_m15)
            state_idx_m5 = latest_state(hmm_m5, df_m5)
            # Translate to semantic labels
            state_h1 = label_h1[state_idx_h1]
            state_m15 = label_m15[state_idx_m15]
            state_m5 = label_m5[state_idx_m5]
            # Determine signal
            if state_h1 == "BULL" and state_m15 == "BULL" and state_m5 == "BULL":
                signal = "LONG"
            elif state_h1 == "BEAR" and state_m15 == "BEAR" and state_m5 == "BEAR":
                signal = "SHORT"
            else:
                signal = "NEUTRAL"
            # Current price – use the latest close from the 5‑min frame
            current_price = df_m5["close"].iloc[-1]
            now_utc = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
            dashboard(now_utc, current_price, state_h1, state_m15, state_m5, signal)
            log_signal(now_utc, current_price, state_h1, state_m15, state_m5, signal)
        except KeyboardInterrupt:
            logger.info("Interrupted by user – shutting down bot.")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            # Continue to next iteration after a short pause to avoid tight error loops
            time.sleep(10)

if __name__ == "__main__":
    main_loop()
