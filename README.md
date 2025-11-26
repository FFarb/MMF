# Money Machine Framework (MMF)

Next-generation quant research environment for chaotic crypto futures markets. The stack fuses physics-inspired feature engineering, a democratic feature council, a Tensor-Flex latent refinery, a temporal CNN expert, and a Mixture-of-Experts trader governed by an adaptive scheduler.  
The canonical Git remote lives at [`https://github.com/FFarb/MMF`](https://github.com/FFarb/MMF) — clone or add it as the primary origin before submitting changes.

## Architecture

1. **Data & Physics Engine** (`src/features/advanced_stats.py` / `src/features/__init__.py`)  
   - Builds OHLCV features with internal TA indicators plus Numba-accelerated chaos metrics (Hurst, Shannon entropy, FDI) across multiple windows.  
   - Rolling volatility columns (`volatility_20/100/200`) are emitted for downstream gating logic.

2. **Alpha Council** (`src/features/alpha_council.py`)  
   - Three voting experts (Lasso/ANOVA, RandomForest, Mutual Information) decide which features survive.  
   - A feature must place in the top 50% for at least two experts to pass.

3. **Tensor-Flex Latent Refinery** (`src/features/tensor_flex.py`)  
   - Clusters correlated features, runs Tensor Talk for cross-cluster deconfounding, then per-cluster PCA with stability-aware selection.  
   - Exposes two modes: `mode="selected"` (distilled latents for classic experts) and `mode="full_latents"` (complete latent stack for neural experts).

4. **Money Machine Trader** (`src/models/moe_ensemble.py`)  
   - Mixture-of-Experts: Trend (Gradient Boosting + GraphVisionary hybrid), Range (k-NN), Stress (Logistic Regression), Temporal CNN (`src/models/cnn_temporal.py`).  
   - Gating MLP consumes `[hurst_200, entropy_200, fdi_200]` and outputs four softmax weights. Telemetry reports the average share for each expert and the marginal uplift contributed by the CNN.  
   - CNNExpert operates on `[C × L]` Tensor-Flex latent windows and saves artifacts under `artifacts/cnn_expert`.

5. **Meta Controller** (`src/training/meta_controller.py`)  
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
    tensor_flex.py              # Tensor-Flex distillation & latent routing
  models/
    __init__.py                 # SniperModelTrainer (primary flow)
    moe_ensemble.py             # Mixture-of-Experts implementation
    cnn_temporal.py             # Temporal ConvNet + CNNExpert wrapper
  training/
    meta_controller.py          # Adaptive scheduler
```

Legacy one-off scripts (e.g., `run_experiment.py`, visualization demos) remain for backward compatibility but the canonical entry point is `run_deep_research.py`.

## Requirements

- Python 3.10+  
- Core libraries: `pandas`, `numpy`, `scikit-learn`, `numba`, `plotly`, `pyarrow`  
- Exchange connectivity: Bybit public REST (no API key required for historical klines, but non-US IP may be necessary).

Install dependencies:

```bash
pip install -r requirements.txt
```

## Git & Repository Initialization

```bash
# Clone (or add as origin) the official repo
git clone https://github.com/FFarb/MMF.git
cd MMF
# If this tree already exists locally, repoint origin:
git remote set-url origin https://github.com/FFarb/MMF.git
git pull origin main
```

Use feature branches for changes, commit locally, then push back to `FFarb/MMF`.

## Vast AI Deployment Cheat Sheet

1. **Provision** a GPU/CPU instance via Vast AI (Ubuntu 22.04+ recommended).  
2. **Bootstrap packages**:

```bash
sudo apt update
sudo apt install -y git python3.10 python3.10-venv python3-pip
```

3. **Clone + setup**:

```bash
cd /workspace  # or preferred mount
git clone https://github.com/FFarb/MMF.git
cd MMF
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Prime artifacts (optional)**:

```bash
# Fit Tensor-Flex only (saves to artifacts/tensor_flex)
python run_deep_research.py --use-tensor-flex --tensor-flex-train-only
```

5. **Full training passes**:

```bash
# Standard run with Tensor-Flex + CNN expert enabled via config
python run_deep_research.py --use-tensor-flex

# Force Tensor-Flex refit and retrain everything
python run_deep_research.py --use-tensor-flex --tensor-flex-force-retrain

# Disable Tensor-Flex/CNN for debugging
python run_deep_research.py --no-tensor-flex
```

All snapshots land in `artifacts/`, and CNN artifacts are persisted under `artifacts/cnn_expert`.

## Notes & Tips

- The SignalFactory lags every engineered feature across `[1,2,3,5,8,13]` bars; expect ~1,300 columns before council pruning.  
- `AlphaCouncil.top_ratio` controls how aggressive the voting cutoff is.  
- To integrate with other assets or intervals, adjust `src/config.py` and rerun `run_deep_research.py`.  
- The MoE gating network requires the physics trio; do not remove `hurst_200`, `entropy_200`, or `fdi_200` columns.  
- Parameter count is a diagnostic metric; use it to track how model capacity changes across runs.  
- Monitor `[MoE] Gating Shares` in the console: a healthy CNN integration shows a non-zero `share_cnn` with stable `cnn_delta_mean`.

---
Created by the Quant Futures Research Team (2025). All components are intended for research use only.
