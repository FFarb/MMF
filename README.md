# Money Machine Research Stack

Next-generation quant research environment for chaotic crypto futures markets. The stack fuses physics-inspired feature engineering, a democratic feature council, and a Mixture-of-Experts trader governed by an adaptive scheduler.

## Architecture

1. **Data & Physics Engine** (`src/features/advanced_stats.py` / `src/features/__init__.py`)  
   - Builds OHLCV features with `pandas_ta` plus Numba-accelerated chaos metrics (Hurst, Shannon entropy, FDI) across multiple windows.  
   - Rolling volatility columns (`volatility_20/100/200`) are emitted for downstream gating logic.

2. **Alpha Council** (`src/features/alpha_council.py`)  
   - Three voting experts (Lasso/ANOVA, RandomForest, Mutual Information) decide which features survive.  
   - A feature must place in the top 50% for at least two experts to pass.

3. **Money Machine Trader** (`src/models/moe_ensemble.py`)  
   - Mixture-of-Experts: Trend (Gradient Boosting), Range (k-NN), Stress (Logistic Regression).  
   - Gating MLP consumes `[hurst_200, entropy_200, fdi_200]` to weight each expert per sample.  
   - `get_system_complexity()` reports the effective parameter count (gating weights + tree nodes + k-NN memory + logistic coefs).

4. **Meta Controller** (`src/training/meta_controller.py`)  
   - Observes market entropy and volatility to adjust estimator counts / epochs.  
   - Monitors validation stability and rolling Sharpe decay to trigger deeper training or retraining.

## Running the Deep Research Pipeline

```bash
python run_deep_research.py
```

`run_deep_research.py` orchestrates the full flow:

1. Fetch live data (Bybit BTCUSDT 60m) via `MarketDataLoader`.  
2. Build features through the SignalFactory and `apply_rolling_physics`.  
3. Create quick forward-looking labels (0.5% / 36-bar hurdle).  
4. Let the Alpha Council select elite features (while forcing the physics trio).  
5. Split chronologically (80/20) and train the MoE with hyperparameters suggested by the `TrainingScheduler`.  
6. Print precision / recall metrics, the AI parameter count, and store a validation snapshot at `artifacts/money_machine_snapshot.parquet`.

Example console snippet:

```
[3] MIXED MODE TRAINING (Mixture-of-Experts)
    Meta-Controller recommends: {'n_estimators': 270, 'epochs': 38}
[4] SYSTEM VITAL SIGNS
    Gating Network Params : 41
    Trend Expert Nodes    : 3,668
    Range Memory Units    : 902,016
    Stress Coefficients   : 649
    TOTAL AI PARAMETERS   : 906,374
```

## Repository Layout

```
run_deep_research.py            # Master orchestrator
src/
  config.py                     # Global constants
  data_loader.py                # Bybit / cache utilities
  features/
    __init__.py                 # SignalFactory + feature helpers
    advanced_stats.py           # Numba chaos metrics
    alpha_council.py            # Feature voting system
  models/
    __init__.py                 # SniperModelTrainer (primary flow)
    moe_ensemble.py             # Mixture-of-Experts implementation
  training/
    meta_controller.py          # Adaptive scheduler
```

Legacy one-off scripts (e.g., `run_experiment.py`, visualization demos) remain for backward compatibility but the canonical entry point is `run_deep_research.py`.

## Requirements

- Python 3.10+  
- Core libraries: `pandas`, `numpy`, `scikit-learn`, `pandas-ta`, `numba`, `plotly`, `pyarrow`  
- Exchange connectivity: Bybit public REST (no API key required for historical klines, but non-US IP may be necessary).

Install dependencies:

```bash
pip install -r requirements.txt
```

## Notes & Tips

- The SignalFactory lags every engineered feature across `[1,2,3,5,8,13]` bars; expect ~1,300 columns before council pruning.  
- `AlphaCouncil.top_ratio` controls how aggressive the voting cutoff is.  
- To integrate with other assets or intervals, adjust `src/config.py` and rerun `run_deep_research.py`.  
- The MoE gating network requires the physics trio; do not remove `hurst_200`, `entropy_200`, or `fdi_200` columns.  
- Parameter count is a diagnostic metric; use it to track how model capacity changes across runs.

---
Created by the Quant Futures Research Team (2025). All components are intended for research use only.
