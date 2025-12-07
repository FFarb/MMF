# micro_structure_analysis.py
"""
Micro‑Structure Analysis
=======================
This script evaluates whether adding a 1‑minute (M1) confirmation layer improves trade entry
prices compared to the existing 5‑minute (M5) signal.

Steps performed:
1. Fetch 90 days of H1, M15 and M5 data (macro layer) using ``BybitDataManager``.
2. Compute log‑returns and a rolling volatility feature for each interval.
3. Align the three data‑frames on the M5 timestamps and train a Gaussian HMM (3 hidden
   states = LONG / NEUTRAL / SHORT).
4. Derive a *consensus signal* from the HMM – the signal is the same for all three intervals
   because the model is trained on the combined feature set.
5. Fetch the last 2 days of M1 data (high‑frequency layer).
6. Simulate two entry strategies over the last 48 h:
   * **Scenario A (M5 only)** – enter at the open of the next M1 candle after a consensus
     signal appears.
   * **Scenario B (M1 confirmation)** – after the M5 signal, wait until the M1 momentum
     (simple up/down based on the previous close) matches the consensus direction, then enter
     at the open of the following M1 candle.
7. Record entry prices for both scenarios, compute the average price improvement and plot the
   M1 close series with markers for each scenario.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path
import matplotlib.pyplot as plt
from hmmlearn import hmm

from data_manager import BybitDataManager

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SYMBOL = "BTCUSDT"
# Macro intervals (in minutes) – used for the consensus HMM
MACRO_INTERVALS = {"H1": 60, "M15": 15, "M5": 5}
MACRO_DAYS_BACK = 90
# High‑frequency interval
M1_INTERVAL = 1
M1_DAYS_BACK = 2  # only last 2 days for the HF layer
# Rolling volatility window (in number of periods – 24 h equivalent)
ROLLING_WINDOW_HOURS = 24
# ---------------------------------------------------------------------------

def fetch_interval_data(interval_minutes: int, days_back: int) -> pd.DataFrame:
    """Fetch OHLCV data for the given interval and number of days.

    Returns a DataFrame indexed by UTC timestamps (datetime) with columns:
    ``open, high, low, close, volume``.
    """
    manager = BybitDataManager(symbol=SYMBOL, interval=str(interval_minutes))
    df = manager.get_data(days_back=days_back)
    # Ensure the index is timezone-aware UTC
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    else:
        df = df.tz_convert("UTC")
    # Deduplicate any duplicate timestamps, keeping the last occurrence
    df = df[~df.index.duplicated(keep="last")]
    return df


def add_features(df: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    """Add log-returns and rolling volatility to a DataFrame.

    ``ROLLING_WINDOW_HOURS`` is converted to a number of periods appropriate for the
    interval (e.g. for a 5-minute interval 24 h = 24*60/5 = 288 periods).
    """
    df = df.copy()
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    window = int((ROLLING_WINDOW_HOURS * 60) / interval_minutes)
    df["vol"] = df["log_ret"].rolling(window=window).std()
    return df


def train_macro_hmm(df_macro: pd.DataFrame) -> pd.Series:
    """Train a 3 state Gaussian HMM on the macro feature matrix.

    Returns a Series ``signal`` aligned with ``df_macro`` where 1 = LONG, -1 = SHORT,
    0 = NEUTRAL.
    """
    # Identify feature columns (log_ret_* and vol_*)
    feature_cols = [c for c in df_macro.columns if c.startswith("log_ret_") or c.startswith("vol_")]
    # Drop rows with any NaNs – required for HMM training
    df_clean = df_macro[feature_cols].dropna()
    X = df_clean.values
    model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100, init_params="mcs", params="tmc", random_state=42)
    model.fit(X)
    hidden_states = model.predict(X)
    df_clean = df_clean.copy()
    df_clean["state"] = hidden_states
    # Use the H1 log return to decide which state is LONG / SHORT
    state_means = df_clean.groupby("state")["log_ret_H1"].mean()
    long_state = int(state_means.idxmax())
    short_state = int(state_means.idxmin())
    neutral_state = ({0, 1, 2} - {long_state, short_state}).pop()
    mapping = {long_state: 1, short_state: -1, neutral_state: 0}
    signal = pd.Series(df_clean["state"].map(mapping), index=df_clean.index)
    return signal


def simple_m1_momentum(df_m1: pd.DataFrame) -> pd.Series:
    """Very lightweight M1 direction indicator.

    Returns 1 for upward momentum (close > previous close), -1 for downward, NaN for no change.
    """
    momentum = np.sign(df_m1["close"].diff())
    momentum.replace(0, np.nan, inplace=True)
    return momentum


def run_simulation():
    # -------------------------------------------------------------------
    # 1. Load macro data and build combined feature DataFrame (aligned on M5)
    # -------------------------------------------------------------------
    macro_frames = {}
    for name, minutes in MACRO_INTERVALS.items():
        df = fetch_interval_data(minutes, MACRO_DAYS_BACK)
        df = add_features(df, minutes)
        df.rename(columns={"log_ret": f"log_ret_{name}", "vol": f"vol_{name}"}, inplace=True)
        macro_frames[name] = df[[f"log_ret_{name}", f"vol_{name}"]]

    # Merge on timestamps – outer join then forward‑fill to align lower‑freq data
    df_macro = macro_frames["M5"].copy()
    for name in ["H1", "M15"]:
        df_macro = df_macro.join(macro_frames[name], how="outer")
    df_macro = df_macro.sort_index().ffill().dropna()

    # -------------------------------------------------------------------
    # 2. Train macro HMM and obtain consensus signal (aligned with M5 timestamps)
    # -------------------------------------------------------------------
    consensus_signal = train_macro_hmm(df_macro)
    df_macro["signal"] = consensus_signal

    # -------------------------------------------------------------------
    # 3. Load high‑frequency M1 data (last 2 days) and keep only last 48 h
    # -------------------------------------------------------------------
    df_m1 = fetch_interval_data(M1_INTERVAL, M1_DAYS_BACK)
    now_utc = pd.Timestamp.now(tz="UTC")
    start_48h = now_utc - timedelta(hours=48)
    df_m1 = df_m1.loc[start_48h:]
    # Deduplicate M1 timestamps just in case
    df_m1 = df_m1[~df_m1.index.duplicated(keep="last")]

    # Pre‑compute M1 momentum for confirmation logic
    m1_momentum = simple_m1_momentum(df_m1)

    # -------------------------------------------------------------------
    # 4. Simulate the two scenarios
    # -------------------------------------------------------------------
    scenario_a_prices = []
    scenario_b_prices = []
    # Identify timestamps where a consensus signal appears (non‑zero)
    signal_changes = df_macro[df_macro["signal"] != 0]
    for ts, row in signal_changes.iterrows():
        direction = int(row["signal"])
        # First M1 candle after the M5 timestamp
        m1_after = df_m1[df_m1.index > ts]
        if m1_after.empty:
            continue
        # Scenario A: entry at open of next M1 candle
        entry_a = m1_after.iloc[0]["open"]
        scenario_a_prices.append(entry_a)

        # Scenario B: wait until M1 momentum matches direction
        match_idx = None
        for i in range(1, len(m1_after)):
            current_mom = m1_momentum.loc[m1_after.index[i]]
            # If duplicates still exist or it returns a series, take the last value
            if isinstance(current_mom, pd.Series):
                current_mom = current_mom.iloc[-1]
            if current_mom == direction:
                match_idx = i
                break
        if match_idx is None:
            # No matching momentum – fall back to Scenario A price
            scenario_b_prices.append(entry_a)
            continue
        # Entry price is open of the candle after the matching momentum candle
        if match_idx + 1 < len(m1_after):
            entry_b = m1_after.iloc[match_idx + 1]["open"]
        else:
            entry_b = m1_after.iloc[-1]["open"]
        scenario_b_prices.append(entry_b)

    # -------------------------------------------------------------------
    # 5. Compute average improvement
    # -------------------------------------------------------------------
    improvements = []
    for a, b, direction in zip(scenario_a_prices, scenario_b_prices, signal_changes["signal"]):
        if direction == 1:  # LONG – lower entry price is better
            improvements.append(a - b)
        elif direction == -1:  # SHORT – higher entry price is better
            improvements.append(b - a)
    avg_improvement = np.mean(improvements) if improvements else np.nan

    # -------------------------------------------------------------------
    # 6. Plot results
    # -------------------------------------------------------------------
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_m1.index, df_m1["close"], label="M1 Close", color="cyan")
    # Scenario A markers (blue triangles)
    a_times = signal_changes.index[: len(scenario_a_prices)]
    ax.scatter(a_times, scenario_a_prices, marker="^", color="deepskyblue", s=80, label="Scenario A (M5)")
    # Scenario B markers (gold stars)
    b_times = signal_changes.index[: len(scenario_b_prices)]
    ax.scatter(b_times, scenario_b_prices, marker="*", color="gold", s=120, label="Scenario B (M1 Confirm)")
    ax.set_title("M1 Close with Entry Points (Last 48h)")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Price (USDT)")
    ax.legend()
    plot_path = Path("micro_structure_plot.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------
    # 7. Print report
    # -------------------------------------------------------------------
    print("=== Micro‑Structure Analysis Report ===")
    print(f"Total signals evaluated: {len(scenario_a_prices)}")
    print(f"Average entry‑price improvement per trade: {avg_improvement:.4f} USDT")
    print(f"Plot saved to: {plot_path.absolute()}")

if __name__ == "__main__":
    run_simulation()
