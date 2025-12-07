"""
Integration Guide: MoE Diagnostic Visualization System

This guide shows how to integrate the VisualReporter into your training loop
to generate comprehensive diagnostics after each fold.
"""

# ============================================================================
# STEP 1: Import the visualization system
# ============================================================================

from src.visualization import VisualReporter, create_diagnostic_report

# ============================================================================
# STEP 2: Initialize the reporter (once, before training loop)
# ============================================================================

# Create reporter with custom output directory
reporter = VisualReporter(
    output_dir=Path("artifacts/diagnostics"),
    expert_colors={
        'Trend': '#2E7D32',      # Dark Green
        'Range': '#1976D2',      # Blue
        'Stress': '#C62828',     # Red
        'Elastic': '#F57C00',    # Orange
        'Pattern': '#7B1FA2',    # Purple
    }
)

# ============================================================================
# STEP 3: Integrate into training loop (after each fold)
# ============================================================================

# Example integration in run_enriched_fleet.py or similar:

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
    print(f"\nCLUSTER {cluster_id} - FOLD {fold_idx}/{n_folds}")
    
    # ... existing training code ...
    
    # Train MoE
    moe = MixtureOfExpertsEnsemble(
        physics_features=available_physics,
        random_state=RANDOM_SEED,
        use_cnn=True,
        use_ou=True,
        use_asset_embedding=(len(members) > 1),
        cnn_params=cnn_params,
        cnn_epochs=15,
    )
    
    moe.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Get predictions
    y_pred_proba = moe.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # ========================================================================
    # NEW: Get expert weights for diagnostics
    # ========================================================================
    
    # Extract expert weights from gating network
    expert_telemetry = moe.get_expert_telemetry(X_val)
    expert_weights = expert_telemetry['weights']  # Shape: (n_samples, n_experts)
    expert_names = expert_telemetry['expert_names']
    
    # Get timestamps and prices
    timestamps = timestamp_col.iloc[val_idx]
    
    # Get prices (from close column or X_val)
    if 'close' in cluster_df.columns:
        prices = cluster_df['close'].iloc[val_idx].values
    elif 'close' in X_val.columns:
        prices = X_val['close'].values
    else:
        # Fallback: use index as proxy
        prices = np.arange(len(val_idx))
    
    # ========================================================================
    # NEW: Create diagnostic report
    # ========================================================================
    
    # For single-asset clusters
    if len(members) == 1:
        asset = members[0]
        
        report = create_diagnostic_report(
            fold_id=fold_idx,
            asset=asset,
            timestamps=timestamps,
            prices=prices,
            X_val=X_val,
            y_val=y_val,
            y_pred=y_pred,
            expert_weights=expert_weights,
            expert_names=expert_names,
        )
        
        # Generate all diagnostic plots
        plot_paths = reporter.generate_full_report(report, show_plots=False)
        
        print(f"\n[DIAGNOSTICS] Generated {len(plot_paths)} plots for {asset}")
    
    # For multi-asset clusters, generate per-asset reports
    else:
        for asset in members:
            asset_mask = X_val['asset_id'] == asset
            
            if asset_mask.sum() < 10:  # Skip if too few samples
                continue
            
            report = create_diagnostic_report(
                fold_id=fold_idx,
                asset=asset,
                timestamps=timestamps[asset_mask],
                prices=prices[asset_mask],
                X_val=X_val[asset_mask],
                y_val=y_val[asset_mask],
                y_pred=y_pred[asset_mask],
                expert_weights=expert_weights[asset_mask],
                expert_names=expert_names,
            )
            
            # Generate all diagnostic plots
            plot_paths = reporter.generate_full_report(report, show_plots=False)
            
            print(f"\n[DIAGNOSTICS] Generated {len(plot_paths)} plots for {asset}")
    
    # ... continue with existing code ...

# ============================================================================
# STEP 4: Review the generated plots
# ============================================================================

# After training completes, check:
# artifacts/diagnostics/
#   ├── trade_execution_BTCUSDT_fold1.png
#   ├── regime_heatmap_BTCUSDT_fold1.png
#   ├── phase_portrait_BTCUSDT_fold1.png
#   ├── expert_performance_BTCUSDT_fold1.png
#   ├── trade_execution_BTCUSDT_fold2.png
#   └── ...

# ============================================================================
# ALTERNATIVE: Minimal Integration (just regime heatmap)
# ============================================================================

# If you only want the regime switching visualization:

from src.visualization import VisualReporter, DiagnosticReport

reporter = VisualReporter()

# After getting predictions and expert weights:
report = DiagnosticReport(
    fold_id=fold_idx,
    asset=asset,
    timestamps=timestamps,
    prices=prices,
    predictions=y_pred,
    true_labels=y_val,
    expert_weights=expert_weights,
    expert_names=['Trend', 'Range', 'Stress', 'Elastic', 'Pattern'],
)

# Generate only regime heatmap
path = reporter.plot_regime_heatmap(report)
print(f"Regime heatmap saved to: {path}")

# ============================================================================
# WHAT YOU'LL SEE IN THE PLOTS
# ============================================================================

# Plot 1: Trade Execution
# - Price line chart
# - Green/Red markers for correct/incorrect predictions
# - Background color showing which expert is dominant
# - Reveals: "When Elastic expert is active, trades are more accurate"

# Plot 2: Regime Heatmap
# - Stacked area chart of expert probabilities over time
# - Timeline showing dominant expert
# - Reveals: "System switches from Trend to Elastic during volatility spikes"

# Plot 3: Phase Portrait
# - Scatter plot of FracDiff vs Volatility
# - Colored by dominant expert
# - Reveals: "Elastic expert dominates in high-volatility, mean-reverting regions"
# - Prep for Neural ODE: Shows state space structure

# Plot 4: Expert Performance
# - Activation frequency per expert
# - Accuracy when each expert is dominant
# - Average confidence and weight distribution
# - Reveals: "Elastic has 78% accuracy but only 15% activation"

# ============================================================================
# KEY INSIGHTS YOU'LL GAIN
# ============================================================================

# 1. Regime Identification:
#    - See exactly when market switches from trending to mean-reverting
#    - Identify which expert performs best in each regime

# 2. Expert Specialization:
#    - Verify that experts are actually specializing
#    - Find underutilized experts (candidates for removal or tuning)

# 3. Gating Network Quality:
#    - Check if gating network switches at appropriate times
#    - Identify regime misclassification

# 4. ODE Preparation:
#    - Phase portrait shows state space structure
#    - Reveals natural boundaries between regimes
#    - Guides Neural ODE architecture design

# 5. Debugging:
#    - Find why Fold 3 hit 78% but Fold 5 only 40%
#    - Identify data distribution shifts
#    - Spot overfitting or underfitting patterns

# ============================================================================
# NEXT STEPS: Transition to Neural ODEs
# ============================================================================

# After analyzing these plots, you'll be ready to implement Neural ODEs:

# 1. Use phase portrait to design state space:
#    State = [FracDiff, Volatility, Momentum, ...]

# 2. Use regime heatmap to identify regime transitions:
#    dState/dt = f_neural(State, θ)

# 3. Use expert performance to set initial θ:
#    θ_elastic = [mean_reversion_speed, equilibrium_level, ...]

# 4. Train Neural ODE to learn dynamic θ:
#    dθ/dt = g_neural(State, θ)

# This gives you adaptive physics parameters instead of fixed ones!
