import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure src is in the path
sys.path.append(str(Path(__file__).parent))

from src.data_loader import MarketDataLoader
from src.features import build_feature_dataset
from src.models import SniperModelTrainer
from src.regimes import MarketRegimeModel
from src.policy import TradePolicy
from src.config import SYMBOL, INTERVAL, DAYS_BACK

def main():
    print("=" * 60)
    print(f"Starting Regime-Adaptive Sniper Experiment: {SYMBOL} ({INTERVAL}m)")
    print("=" * 60)

    # 1. Initialize Loader
    loader = MarketDataLoader()

    # 2. Build/Load Features
    print("\n[1/4] Building Feature Set...")
    features = build_feature_dataset(loader=loader, days_back=DAYS_BACK)

    # 3. Train HMM & Add Regime Feature
    print("\n[2/4] Training Market Regime Model (HMM)...")
    hmm_model = MarketRegimeModel(n_components=3)
    features_with_regime = hmm_model.fit_predict(features)
    
    # 4. Train Sniper Model
    print("\n[3/4] Training Sniper Model (with Regime features)...")
    trainer = SniperModelTrainer()
    # Pass the dataframe directly to avoid reloading from disk without regime cols
    results = trainer.run(input_df=features_with_regime)
    
    model = results["model"]
    top_features = results["top_features"]
    
    # 5. Run Backtest Simulation
    print("\n[4/4] Running Regime-Adaptive Backtest Simulation...")
    run_simulation(features_with_regime, model, top_features)

def run_simulation(df: pd.DataFrame, model, feature_cols: list):
    """
    Simulate trading with dynamic policy.
    """
    policy = TradePolicy()
    
    # Prepare data for simulation
    # We need to predict on the whole dataset (or test set)
    # For simplicity, let's simulate on the whole dataset to see regime effects, 
    # but strictly we should respect train/test split. 
    # The trainer split is 0.8. Let's use the last 20% for valid backtest.
    
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    
    print(f"       Simulating on Test Set ({len(test_df)} candles)...")
    
    # Prepare X for model
    X_test = test_df[feature_cols]
    
    # Get Model Signals
    # 1 = Long, 0 = No Trade (Model was trained on 1=Win, 0=Loss/NoTrade)
    # Wait, the model predicts "Will this trade be a win?". 
    # But we need to know IF we enter. 
    # The current labeling logic in models.py labels "Successful Long" as 1.
    # So if model predicts 1, it means "High probability of hitting TP before SL".
    # So Signal = 1 means ENTER LONG.
    
    signals = model.predict(X_test)
    test_df["signal"] = signals
    
    # Simulation Loop
    balance = 10000.0
    equity_curve = [balance]
    trades = []
    
    regime_stats = {
        "Trend": {"wins": 0, "losses": 0, "leverage_sum": 0.0, "count": 0},
        "Range": {"wins": 0, "losses": 0, "leverage_sum": 0.0, "count": 0},
        "Stress": {"wins": 0, "losses": 0, "leverage_sum": 0.0, "count": 0},
    }
    
    # Iterate
    # Note: Vectorization is faster, but loop is easier for dynamic logic
    
    close_prices = test_df["close"].values
    high_prices = test_df["high"].values
    low_prices = test_df["low"].values
    atrs = test_df["ATR_14"].values
    regimes = test_df["regime"].values
    sigs = test_df["signal"].values
    
    n = len(test_df)
    
    for i in range(n):
        if i + 1 >= n: break # Need at least next candle
        
        if sigs[i] == 1:
            # Entry
            entry_price = close_prices[i]
            current_atr = atrs[i]
            current_regime = regimes[i]
            
            # Get Policy
            params = policy.get_parameters(current_regime, current_atr, entry_price)
            
            # Skip if leverage is 0 or Stress policy says no
            if params.leverage <= 0:
                continue
                
            tp_price = entry_price + (current_atr * params.tp_mult)
            sl_price = entry_price - (current_atr * params.sl_mult)
            
            # Check Outcome (Simplified: check next 36 bars or until hit)
            # We need to look ahead
            outcome = 0 # 0=Running, 1=Win, -1=Loss
            pnl_pct = 0.0
            
            # Look ahead up to 36 bars (Horizon)
            horizon = 36
            for j in range(1, horizon + 1):
                if i + j >= n: break
                
                c_high = high_prices[i+j]
                c_low = low_prices[i+j]
                
                if c_low <= sl_price:
                    outcome = -1
                    pnl_pct = (sl_price - entry_price) / entry_price
                    break
                if c_high >= tp_price:
                    outcome = 1
                    pnl_pct = (tp_price - entry_price) / entry_price
                    break
            
            if outcome != 0:
                # Realized PnL
                # Fee approx 0.06% per trade (0.03 entry + 0.03 exit)
                fees = 0.0006
                realized_roe = (pnl_pct * params.leverage) - fees
                
                balance_change = balance * realized_roe
                balance += balance_change
                
                # Stats
                stats = regime_stats.get(current_regime, None)
                if stats:
                    stats["count"] += 1
                    stats["leverage_sum"] += params.leverage
                    if outcome == 1:
                        stats["wins"] += 1
                    else:
                        stats["losses"] += 1
                
                trades.append({
                    "regime": current_regime,
                    "leverage": params.leverage,
                    "roe": realized_roe,
                    "outcome": "Win" if outcome == 1 else "Loss"
                })
        
        equity_curve.append(balance)

    # Reporting
    print("\n" + "=" * 60)
    print("REGIME-ADAPTIVE STRATEGY RESULTS")
    print("=" * 60)
    print(f"Final Balance: ${balance:.2f} (Start: $10000.00)")
    print(f"Total Return:  {(balance - 10000) / 10000:.2%}")
    print("-" * 60)
    print(f"{'Regime':<10} | {'Win Rate':<10} | {'Avg Lev':<10} | {'Trades':<10}")
    print("-" * 60)
    
    for regime, stats in regime_stats.items():
        count = stats["count"]
        if count > 0:
            win_rate = stats["wins"] / count
            avg_lev = stats["leverage_sum"] / count
            print(f"{regime:<10} | {win_rate:.2%}    | {avg_lev:.2f}x      | {count:<10}")
        else:
            print(f"{regime:<10} | N/A        | N/A        | 0")
            
    print("=" * 60)
    print("Experiment completed.")

if __name__ == "__main__":
    main()
