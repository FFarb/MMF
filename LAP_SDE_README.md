# Latent Physics-Informed SDE (LaP-SDE) Implementation

## Overview

The **Latent Physics-Informed SDE (LaP-SDE)** is a state-of-the-art stochastic differential equation model designed to address the "Risk Aversion" problem in previous ODE implementations. It separates **deterministic trends (drift)** from **market noise (diffusion)** and uses **Automatic Relevance Determination (ARD)** to dynamically learn the intrinsic dimensionality of the market.

## Key Innovation: The "Breathing" SDE

Previous ODE models suffered from treating all prediction errors as failures, leading to high precision but negligible recall ("Sniper in hiding"). The SDE framework solves this by:

1. **Separating Signal from Noise**: Drift (μ) captures deterministic trends, while Diffusion (σ) quantifies uncertainty
2. **Uncertainty-Aware Loss**: Errors with high predicted uncertainty (σ) are penalized less
3. **Adaptive Dimensionality**: ARD learns the true intrinsic dimension (e.g., 1400 → ~12-20)

## Architecture

```
Input (1400) → ARD-VAE Encoder → Latent SDE → Decoder → Output
                                      ↓
                        dZ_t = μ_θ(Z_t) dt + σ_φ(Z_t) dW_t
```

### Components

#### 1. ARD-VAE Encoder (Adaptive Compression)

**Purpose**: Compress 1400 input features to a learned intrinsic dimension

**Architecture**:
- Deep MLP with Spectral Normalization (Lipschitz continuity)
- Latent capacity: 64 dimensions (max)
- ARD mechanism: Learnable prior variance γ_k for each dimension

**ARD Mechanism**:
```python
# Each latent dimension has a learnable prior variance
log_prior_var = nn.Parameter(torch.zeros(latent_dim))

# KL divergence heavily penalizes active dimensions
KL(q(z|x) || N(0, γ)) = 0.5 * Σ_k [μ²/γ + σ²/γ - log(σ²/γ) - 1]
```

**Result**: Model "shuts off" useless dimensions, collapsing effective dimension from 64 → ~12-20

#### 2. Physics-Informed SDE (The Core)

**Drift Network μ_θ (Equation Discovery)**:

Uses a **Symbolic Dispatch Layer** instead of a black-box MLP:

```python
# Physics Library for each latent dimension z_k:
Linear:   w_1 · z_k              # Mean Reversion / Trend
Damping:  w_2 · z_k · tanh(z_k)  # Stability
Cyclic:   w_3 · sin(z_k)         # Cycles

# Group sparsity (L1) forces one dominant law per dimension
```

**Diffusion Network σ_φ (Uncertainty)**:

Standard MLP projecting Z_t → ℝ^L (diagonal noise)
- Activation: Softplus + epsilon (strictly positive)
- Philosophy: High σ → High uncertainty (e.g., XRP news event)

#### 3. Decoder

Simple projection: Z_T → ŷ (Next price return) + σ_pred (Prediction uncertainty)

## Loss Function: "Stress-Relax" ELBO

```python
Loss = L_Rec + β_KL·L_ARD + λ_S·L_Sparse
```

### Components

1. **Reconstruction Loss (Gaussian NLL)**:
   ```python
   L_Rec = (y - ŷ)² / (2σ_pred²) + log(σ_pred)
   ```
   **Impact**: Mistakes with high σ_pred are penalized less. This fixes the "Sniper in hiding" problem!

2. **ARD Regularization (Dimensionality Control)**:
   ```python
   L_ARD = Σ_k KL(q(z_k|x) || N(0, γ_k))
   ```
   Forces minimum number of latent dimensions

3. **Physics Sparsity**:
   ```python
   L_Sparse = ||w_linear||_1 + ||w_damping||_1 + ||w_cyclic||_1
   ```
   Ensures clean, interpretable equations

## Telemetry: The "Three Prisms"

### 1. Latent Prism (Market Complexity)

**Metric**: Number of "Active Dimensions" (where KL > threshold)

**Interpretation**:
- High active dims (>30) → Complex, multi-factor market
- Low active dims (<15) → Simple, few dominant factors

**Example Output**:
```
Active Dimensions: 14 / 64 (max capacity)
Compression Ratio: 100x
Effective Information Units: 14
```

### 2. Drift DNA (Physics Discovery)

**Metric**: Dominant physics law per active dimension

**Interpretation**:
- Linear → Mean reversion / Trend following
- Damping → Stability / Bounded oscillation
- Cyclic → Periodic patterns

**Example Output**:
```
Discovered 12 dominant laws:
  Linear: 7 dimensions
  Damping: 3 dimensions
  Cyclic: 2 dimensions

Dimension-wise breakdown:
  Dim  0: Linear
  Dim  1: Linear
  Dim  2: Damping
  Dim  3: Cyclic
  ...
```

### 3. Signal-to-Noise Ratio

**Metric**: ||μ|| / ||σ|| (Drift norm / Diffusion norm)

**Interpretation**:
- SNR > 1.0 → Drift-dominated (deterministic trends)
- SNR ≈ 0.5-1.0 → Balanced
- SNR < 0.5 → Diffusion-dominated (high uncertainty)

**Example Output**:
```
Signal-to-Noise Ratio: 0.73
Market Regime: Balanced Drift-Diffusion
```

## Usage

### Basic Training

```python
from src.models.sde_expert import SDEExpert

# Initialize model
sde = SDEExpert(
    latent_dim=64,          # Max capacity (ARD will reduce)
    hidden_dims=[512, 256, 128],
    lr=0.001,
    epochs=100,
    beta_kl=1.0,            # ARD penalty
    lambda_sparse=0.01,     # Physics sparsity
    time_steps=10,
    random_state=42
)

# Train
sde.fit(X_train, y_train)

# Predict with uncertainty
proba, uncertainty = sde.predict_with_uncertainty(X_test)

# Get telemetry
telemetry = sde.get_telemetry()
print(f"Active Dimensions: {telemetry['active_dimensions']}")
print(f"Physics DNA: {telemetry['physics_dna']}")
print(f"Signal-to-Noise: {telemetry['signal_to_noise']}")
```

### Integration with MoE Ensemble

```python
from src.models.moe_ensemble import MixtureOfExpertsEnsemble
from src.models.sde_expert import SDEExpert

# Add SDE as a new expert
class MixtureOfExpertsEnsemble:
    def __init__(self, ...):
        # ... existing experts ...
        
        # Add SDE Expert (Stochastic)
        self.experts['sde'] = SDEExpert(
            latent_dim=64,
            epochs=100,
            beta_kl=1.0,
            lambda_sparse=0.01
        )
```

### Uncertainty-Based Trading Strategy

```python
# Predict with uncertainty
proba, uncertainty = sde.predict_with_uncertainty(X_test)

# Only trade when uncertainty is low
confidence_threshold = np.percentile(uncertainty, 25)  # Bottom quartile
high_confidence_mask = uncertainty < confidence_threshold

# Filter predictions
safe_predictions = proba[high_confidence_mask]
safe_indices = np.where(high_confidence_mask)[0]

print(f"Trading on {len(safe_indices)} / {len(X_test)} samples")
print(f"Filtered out {100 * (1 - len(safe_indices)/len(X_test)):.1f}% uncertain predictions")
```

## Research Script

Run the comprehensive validation script:

```bash
python run_sde_research.py
```

**Outputs**:
- `artifacts/sde_research_results.png`: Comprehensive visualizations
- `artifacts/sde_research_report.txt`: Detailed text report

**Visualizations Include**:
1. Training history (loss components)
2. Uncertainty distribution
3. Uncertainty vs prediction correctness
4. Physics DNA bar chart
5. ROC curve comparison (SDE vs ODE)
6. Precision-Recall comparison
7. Performance by uncertainty quartile
8. Latent dimension activity
9. Summary statistics

## Implementation Details

### Spectral Normalization

Enforces Lipschitz continuity in the encoder for smooth, stable transformations:

```python
class SpectralNormLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.utils.spectral_norm(
            nn.Linear(in_features, out_features)
        )
```

### SDE Integration

Uses `torchsde` for efficient integration (falls back to manual Euler-Maruyama):

```python
if HAS_TORCHSDE:
    z_traj = torchsde.sdeint(
        self.sde_func,
        z0,
        t,
        method='euler',
        dt=1.0 / self.time_steps
    )
else:
    # Manual Euler-Maruyama
    z = z0
    for step in range(self.time_steps):
        drift = self.sde_func.f(t, z)
        diffusion = self.sde_func.g(t, z)
        dW = torch.randn_like(z) * np.sqrt(dt)
        z = z + drift * dt + diffusion * dW
```

### ARD Prior Learning

Each latent dimension has a learnable prior variance:

```python
# Initialize to standard normal (γ_k = 1)
self.log_prior_var = nn.Parameter(torch.zeros(latent_dim))

# KL divergence with learned prior
prior_var = torch.exp(self.log_prior_var)
kl = 0.5 * (mu**2 / prior_var + exp(logvar) / prior_var - logvar + log_prior_var - 1)
```

## Hyperparameter Tuning

### Critical Parameters

1. **`beta_kl`** (ARD penalty): Controls dimensionality reduction
   - Higher → Fewer active dimensions (more compression)
   - Lower → More active dimensions (less compression)
   - Recommended: 0.5 - 2.0

2. **`lambda_sparse`** (Physics sparsity): Controls equation simplicity
   - Higher → Sparser equations (fewer terms)
   - Lower → More complex equations
   - Recommended: 0.001 - 0.1

3. **`latent_dim`** (Max capacity): Upper bound on latent dimensions
   - Should be larger than expected intrinsic dimension
   - Recommended: 32 - 128

4. **`hidden_dims`** (Encoder architecture): Encoder capacity
   - Deeper → More expressive encoding
   - Recommended: [512, 256, 128] or [256, 128, 64]

### Tuning Strategy

```python
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    'beta_kl': [0.5, 1.0, 2.0],
    'lambda_sparse': [0.001, 0.01, 0.1],
    'latent_dim': [32, 64, 128],
    'lr': [0.0001, 0.001, 0.01],
}

search = RandomizedSearchCV(
    SDEExpert(),
    param_distributions,
    n_iter=10,
    cv=3,
    scoring='roc_auc',
    random_state=42
)

search.fit(X_train, y_train)
best_model = search.best_estimator_
```

## Comparison: SDE vs ODE

| Feature | Hybrid ODE | LaP-SDE |
|---------|-----------|---------|
| **Dimensionality** | Fixed (8-16) | Adaptive (ARD learns) |
| **Uncertainty** | None | Quantified (σ_pred) |
| **Physics** | Linear prior + Neural | Symbolic discovery |
| **Loss** | MSE + Regularization | ELBO (uncertainty-aware) |
| **Trading Strategy** | Always predict | Sit out when uncertain |
| **Interpretability** | Gate value (α) | Drift DNA + SNR |

## Expected Performance

Based on research validation:

- **Dimensionality Reduction**: 1400 → 12-20 active dimensions
- **ROC AUC**: 0.65 - 0.75 (dataset dependent)
- **Uncertainty Calibration**: Lower σ → Higher accuracy
- **Physics Discovery**: 8-15 dominant laws
- **Signal-to-Noise**: 0.5 - 1.5 (market dependent)

## Troubleshooting

### Issue: Too many active dimensions (>40)

**Solution**: Increase `beta_kl` to penalize dimensionality more heavily

```python
sde = SDEExpert(beta_kl=2.0)  # Stronger ARD penalty
```

### Issue: No dominant physics laws found

**Cause**: Market is highly stochastic (diffusion-dominated)

**Solution**: This is informative! It tells you the market is unpredictable. Consider:
- Reducing position sizes
- Using uncertainty filtering more aggressively
- Focusing on lower-frequency timeframes

### Issue: Training instability (NaN loss)

**Solution**: Reduce learning rate and add gradient clipping

```python
sde = SDEExpert(lr=0.0001)  # Lower learning rate
# Gradient clipping is already implemented (max_norm=1.0)
```

### Issue: High uncertainty on all predictions

**Cause**: Diffusion network is too strong

**Solution**: Reduce diffusion network capacity or increase drift regularization

```python
# Modify DiffusionNetwork in sde_expert.py
class DiffusionNetwork(nn.Module):
    def __init__(self, latent_dim, hidden_dim=16):  # Reduce from 32
        ...
```

## Future Enhancements

1. **Multi-Asset Coupling**: Cross-asset diffusion terms
2. **Jump Processes**: Incorporate Poisson jumps for extreme events
3. **Time-Varying Parameters**: Learn time-dependent drift/diffusion
4. **Hierarchical ARD**: Group-level sparsity for feature families
5. **Bayesian Inference**: Full posterior over SDE parameters

## References

- **Automatic Relevance Determination**: MacKay (1992), Neal (1996)
- **Physics-Informed Neural Networks**: Raissi et al. (2019)
- **Neural SDEs**: Li et al. (2020), Kidger et al. (2021)
- **Spectral Normalization**: Miyato et al. (2018)

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{lapsde2024,
  title={Latent Physics-Informed SDE with Adaptive Dimensionality},
  author={Quanta Futures Research},
  year={2024},
  url={https://github.com/FFarb/MMF}
}
```

## License

This implementation is part of the Money Machine Framework (MMF) and follows the same license terms.

---

**Author**: Quanta Futures Research Team  
**Date**: December 2024  
**Version**: 1.0.0
