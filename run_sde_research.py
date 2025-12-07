"""
Research Script: Latent Physics-Informed SDE (LaP-SDE) Validation.

This script validates the LaP-SDE model and compares it to the Hybrid ODE.

Objectives:
1. Train LaP-SDE on multi-asset data
2. Validate ARD dimensionality reduction (1400 → ~12-20)
3. Analyze discovered physics laws (Drift DNA)
4. Compare uncertainty quantification vs ODE
5. Generate comprehensive telemetry report

Expected Outcomes:
- Active dimensions: 12-20 (from 64 max capacity)
- Physics DNA: Interpretable laws per dimension
- Signal-to-Noise: Quantified drift/diffusion balance
- Prediction uncertainty: High σ for chaotic periods, low σ for stable trends
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    auc
)
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_loader import load_multi_asset_data
from src.features.alpha_council import AlphaCouncil
from src.models.sde_expert import SDEExpert
from src.models.hybrid_ode import HybridNeuralODEExpert
from src.config import (
    SYMBOLS,
    TRAIN_SPLIT,
    RANDOM_SEED,
    MULTI_ASSET_CACHE
)

# Plotting style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)


def prepare_data():
    """Load and prepare multi-asset data."""
    print("=" * 80)
    print("STEP 1: DATA PREPARATION")
    print("=" * 80)
    
    # Load multi-asset data
    print(f"\n[1/3] Loading multi-asset data from {MULTI_ASSET_CACHE}...")
    df = load_multi_asset_data()
    
    print(f"  ✓ Loaded {len(df):,} samples across {df['asset_id'].nunique()} assets")
    print(f"  ✓ Time range: {df.index.min()} to {df.index.max()}")
    
    # Check for target column
    if 'target' not in df.columns:
        print("\n  ⚠ No 'target' column found. Creating binary target from returns...")
        if 'return_1h' in df.columns:
            df['target'] = (df['return_1h'] > 0).astype(int)
        elif 'close' in df.columns:
            df['target'] = (df['close'].pct_change() > 0).astype(int)
        else:
            raise ValueError("Cannot create target: no 'return_1h' or 'close' column")
    
    # Remove NaN targets
    df = df.dropna(subset=['target'])
    
    print(f"\n[2/3] Running Alpha Council feature selection...")
    council = AlphaCouncil()
    df_selected = council.fit_transform(df, df['target'].values)
    
    selected_features = [col for col in df_selected.columns if col not in ['target', 'asset_id']]
    print(f"  ✓ Selected {len(selected_features)} features")
    
    # Prepare X and y
    print(f"\n[3/3] Preparing feature matrix and target...")
    
    # Keep asset_id for multi-asset awareness
    feature_cols = selected_features.copy()
    if 'asset_id' in df_selected.columns and 'asset_id' not in feature_cols:
        feature_cols.append('asset_id')
    
    X = df_selected[feature_cols].copy()
    y = df_selected['target'].values
    
    # Handle asset_id encoding
    if 'asset_id' in X.columns:
        # Convert to categorical codes
        X['asset_id'] = pd.Categorical(X['asset_id']).codes
    
    print(f"  ✓ Feature matrix: {X.shape}")
    print(f"  ✓ Target distribution: {np.bincount(y)}")
    
    return X, y, selected_features


def train_models(X_train, y_train, X_test, y_test):
    """Train both SDE and ODE models for comparison."""
    print("\n" + "=" * 80)
    print("STEP 2: MODEL TRAINING")
    print("=" * 80)
    
    results = {}
    
    # 1. Train LaP-SDE
    print("\n[1/2] Training Latent Physics-Informed SDE...")
    print("-" * 80)
    
    sde_expert = SDEExpert(
        latent_dim=64,  # Max capacity (ARD will reduce this)
        hidden_dims=[512, 256, 128],
        lr=0.001,
        epochs=100,
        beta_kl=1.0,  # ARD penalty
        lambda_sparse=0.01,  # Physics sparsity
        time_steps=10,
        random_state=RANDOM_SEED
    )
    
    sde_expert.fit(X_train, y_train)
    
    # Get predictions with uncertainty
    y_pred_sde, uncertainty_sde = sde_expert.predict_with_uncertainty(X_test)
    y_pred_sde_class = (y_pred_sde[:, 1] >= 0.5).astype(int)
    
    # Metrics
    auc_sde = roc_auc_score(y_test, y_pred_sde[:, 1])
    precision, recall, _ = precision_recall_curve(y_test, y_pred_sde[:, 1])
    pr_auc_sde = auc(recall, precision)
    
    results['sde'] = {
        'model': sde_expert,
        'y_pred': y_pred_sde,
        'y_pred_class': y_pred_sde_class,
        'uncertainty': uncertainty_sde,
        'auc': auc_sde,
        'pr_auc': pr_auc_sde,
        'telemetry': sde_expert.get_telemetry()
    }
    
    print(f"\n  ✓ LaP-SDE Performance:")
    print(f"    ROC AUC: {auc_sde:.4f}")
    print(f"    PR AUC: {pr_auc_sde:.4f}")
    print(f"    Mean Uncertainty: {uncertainty_sde.mean():.4f} ± {uncertainty_sde.std():.4f}")
    
    # 2. Train Hybrid ODE (for comparison)
    print("\n[2/2] Training Hybrid Neural ODE (baseline)...")
    print("-" * 80)
    
    ode_expert = HybridNeuralODEExpert(
        latent_dim=16,  # Fixed dimension
        hidden_dim=16,
        lr=0.001,
        epochs=100,
        lambda_l1=0.01,
        lambda_gate=0.1,
        lambda_jac=0.001,
        time_steps=10,
        random_state=RANDOM_SEED
    )
    
    ode_expert.fit(X_train, y_train)
    
    # Get predictions (no uncertainty)
    y_pred_ode = ode_expert.predict_proba(X_test)
    y_pred_ode_class = (y_pred_ode[:, 1] >= 0.5).astype(int)
    
    # Metrics
    auc_ode = roc_auc_score(y_test, y_pred_ode[:, 1])
    precision, recall, _ = precision_recall_curve(y_test, y_pred_ode[:, 1])
    pr_auc_ode = auc(recall, precision)
    
    results['ode'] = {
        'model': ode_expert,
        'y_pred': y_pred_ode,
        'y_pred_class': y_pred_ode_class,
        'auc': auc_ode,
        'pr_auc': pr_auc_ode,
        'diagnostics': ode_expert.get_diagnostics()
    }
    
    print(f"\n  ✓ Hybrid ODE Performance:")
    print(f"    ROC AUC: {auc_ode:.4f}")
    print(f"    PR AUC: {pr_auc_ode:.4f}")
    
    return results


def analyze_telemetry(results):
    """Analyze and visualize model telemetry."""
    print("\n" + "=" * 80)
    print("STEP 3: TELEMETRY ANALYSIS")
    print("=" * 80)
    
    sde_telemetry = results['sde']['telemetry']
    
    # 1. Latent Prism (Active Dimensions)
    print("\n[1/4] Latent Prism: Intrinsic Dimensionality")
    print("-" * 80)
    
    active_dims = sde_telemetry['active_dimensions']
    print(f"  Active Dimensions: {active_dims} / 64 (max capacity)")
    print(f"  Compression Ratio: {1400 / active_dims:.1f}x")
    print(f"  Effective Information Units: {active_dims}")
    
    # 2. Drift DNA (Physics Laws)
    print("\n[2/4] Drift DNA: Discovered Physics Laws")
    print("-" * 80)
    
    physics_dna = sde_telemetry['physics_dna']
    
    if physics_dna:
        law_counts = {}
        for dim, law in physics_dna.items():
            law_counts[law] = law_counts.get(law, 0) + 1
        
        print(f"  Discovered {len(physics_dna)} dominant laws:")
        for law, count in sorted(law_counts.items(), key=lambda x: -x[1]):
            print(f"    {law}: {count} dimensions")
        
        print(f"\n  Dimension-wise breakdown:")
        for dim, law in sorted(physics_dna.items()):
            print(f"    Dim {dim:2d}: {law}")
    else:
        print("  ⚠ No dominant physics laws (highly stochastic market)")
    
    # 3. Signal-to-Noise Ratio
    print("\n[3/4] Signal-to-Noise: Drift vs Diffusion")
    print("-" * 80)
    
    snr = sde_telemetry['signal_to_noise']
    print(f"  Signal-to-Noise Ratio: {snr:.4f}")
    
    if snr > 1.0:
        regime = "Drift-Dominated (Deterministic Trends)"
    elif snr > 0.5:
        regime = "Balanced Drift-Diffusion"
    else:
        regime = "Diffusion-Dominated (High Uncertainty)"
    
    print(f"  Market Regime: {regime}")
    
    # 4. Training History
    print("\n[4/4] Training Convergence")
    print("-" * 80)
    
    history = sde_telemetry['training_history']
    
    final_loss = history[-1]
    print(f"  Final Loss Components:")
    print(f"    Total: {final_loss['total']:.4f}")
    print(f"    Reconstruction: {final_loss['reconstruction']:.4f}")
    print(f"    ARD KL: {final_loss['ard_kl']:.4f}")
    print(f"    Sparsity: {final_loss['sparsity']:.4f}")
    print(f"    Mean Uncertainty: {final_loss['mean_uncertainty']:.4f}")


def visualize_results(results, y_test):
    """Create comprehensive visualizations."""
    print("\n" + "=" * 80)
    print("STEP 4: VISUALIZATION")
    print("=" * 80)
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Training History
    print("\n[1/5] Plotting training history...")
    
    ax1 = plt.subplot(3, 3, 1)
    history = results['sde']['telemetry']['training_history']
    
    epochs = range(1, len(history) + 1)
    total_loss = [h['total'] for h in history]
    rec_loss = [h['reconstruction'] for h in history]
    kl_loss = [h['ard_kl'] for h in history]
    
    ax1.plot(epochs, total_loss, label='Total', linewidth=2)
    ax1.plot(epochs, rec_loss, label='Reconstruction', alpha=0.7)
    ax1.plot(epochs, kl_loss, label='ARD KL', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('LaP-SDE Training History')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Uncertainty Distribution
    print("[2/5] Plotting uncertainty distribution...")
    
    ax2 = plt.subplot(3, 3, 2)
    uncertainty = results['sde']['uncertainty']
    
    ax2.hist(uncertainty, bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(uncertainty.mean(), color='red', linestyle='--', 
                label=f'Mean: {uncertainty.mean():.3f}')
    ax2.set_xlabel('Prediction Uncertainty (σ)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Uncertainty Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Uncertainty vs Correctness
    print("[3/5] Plotting uncertainty vs correctness...")
    
    ax3 = plt.subplot(3, 3, 3)
    
    y_pred_class = results['sde']['y_pred_class']
    correct = (y_pred_class == y_test)
    
    ax3.boxplot([uncertainty[correct], uncertainty[~correct]], 
                labels=['Correct', 'Incorrect'])
    ax3.set_ylabel('Prediction Uncertainty (σ)')
    ax3.set_title('Uncertainty vs Prediction Correctness')
    ax3.grid(True, alpha=0.3)
    
    # 4. Physics DNA Bar Chart
    print("[4/5] Plotting physics DNA...")
    
    ax4 = plt.subplot(3, 3, 4)
    physics_dna = results['sde']['telemetry']['physics_dna']
    
    if physics_dna:
        law_counts = {}
        for dim, law in physics_dna.items():
            law_counts[law] = law_counts.get(law, 0) + 1
        
        laws = list(law_counts.keys())
        counts = list(law_counts.values())
        
        ax4.bar(laws, counts, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Physics Law')
        ax4.set_ylabel('Number of Dimensions')
        ax4.set_title('Drift DNA: Discovered Physics Laws')
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, 'No Dominant Laws\n(Highly Stochastic)', 
                ha='center', va='center', fontsize=12)
        ax4.set_title('Drift DNA: Discovered Physics Laws')
    
    # 5. ROC Comparison
    print("[5/5] Plotting ROC comparison...")
    
    ax5 = plt.subplot(3, 3, 5)
    
    from sklearn.metrics import roc_curve
    
    # SDE ROC
    fpr_sde, tpr_sde, _ = roc_curve(y_test, results['sde']['y_pred'][:, 1])
    ax5.plot(fpr_sde, tpr_sde, label=f'LaP-SDE (AUC={results["sde"]["auc"]:.3f})', 
            linewidth=2)
    
    # ODE ROC
    fpr_ode, tpr_ode, _ = roc_curve(y_test, results['ode']['y_pred'][:, 1])
    ax5.plot(fpr_ode, tpr_ode, label=f'Hybrid ODE (AUC={results["ode"]["auc"]:.3f})', 
            linewidth=2, alpha=0.7)
    
    ax5.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax5.set_xlabel('False Positive Rate')
    ax5.set_ylabel('True Positive Rate')
    ax5.set_title('ROC Curve Comparison')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Precision-Recall Comparison
    ax6 = plt.subplot(3, 3, 6)
    
    # SDE PR
    precision_sde, recall_sde, _ = precision_recall_curve(y_test, results['sde']['y_pred'][:, 1])
    ax6.plot(recall_sde, precision_sde, 
            label=f'LaP-SDE (PR-AUC={results["sde"]["pr_auc"]:.3f})', 
            linewidth=2)
    
    # ODE PR
    precision_ode, recall_ode, _ = precision_recall_curve(y_test, results['ode']['y_pred'][:, 1])
    ax6.plot(recall_ode, precision_ode, 
            label=f'Hybrid ODE (PR-AUC={results["ode"]["pr_auc"]:.3f})', 
            linewidth=2, alpha=0.7)
    
    ax6.set_xlabel('Recall')
    ax6.set_ylabel('Precision')
    ax6.set_title('Precision-Recall Curve Comparison')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Confidence-Stratified Performance
    ax7 = plt.subplot(3, 3, 7)
    
    # Stratify by uncertainty quartiles
    uncertainty = results['sde']['uncertainty']
    quartiles = np.percentile(uncertainty, [25, 50, 75])
    
    strata_labels = ['Q1 (Low σ)', 'Q2', 'Q3', 'Q4 (High σ)']
    strata_acc = []
    
    for i in range(4):
        if i == 0:
            mask = uncertainty <= quartiles[0]
        elif i == 3:
            mask = uncertainty > quartiles[2]
        else:
            mask = (uncertainty > quartiles[i-1]) & (uncertainty <= quartiles[i])
        
        if mask.sum() > 0:
            acc = (results['sde']['y_pred_class'][mask] == y_test[mask]).mean()
            strata_acc.append(acc)
        else:
            strata_acc.append(0)
    
    ax7.bar(strata_labels, strata_acc, alpha=0.7, edgecolor='black')
    ax7.set_ylabel('Accuracy')
    ax7.set_title('Performance by Uncertainty Quartile')
    ax7.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Latent Dimension Activity
    ax8 = plt.subplot(3, 3, 8)
    
    active_dims = results['sde']['telemetry']['active_dimensions']
    max_dims = 64
    
    ax8.bar(['Active', 'Inactive'], 
           [active_dims, max_dims - active_dims],
           alpha=0.7, edgecolor='black')
    ax8.set_ylabel('Number of Dimensions')
    ax8.set_title(f'Latent Prism: {active_dims}/{max_dims} Active')
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. Summary Statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    summary_text = f"""
    LaP-SDE Summary
    ═══════════════════════════════
    
    Dimensionality:
      Input: 1400 features
      Latent: {active_dims} / 64 active
      Compression: {1400/active_dims:.1f}x
    
    Performance:
      ROC AUC: {results['sde']['auc']:.4f}
      PR AUC: {results['sde']['pr_auc']:.4f}
    
    Physics:
      Signal/Noise: {results['sde']['telemetry']['signal_to_noise']:.4f}
      Laws Found: {len(physics_dna)}
    
    Uncertainty:
      Mean σ: {uncertainty.mean():.4f}
      Std σ: {uncertainty.std():.4f}
    """
    
    ax9.text(0.1, 0.9, summary_text, fontsize=10, family='monospace',
            verticalalignment='top')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path('artifacts') / 'sde_research_results.png'
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Saved visualization to {output_path}")
    
    plt.show()


def generate_report(results, y_test):
    """Generate comprehensive text report."""
    print("\n" + "=" * 80)
    print("STEP 5: FINAL REPORT")
    print("=" * 80)
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("LATENT PHYSICS-INFORMED SDE (LaP-SDE) RESEARCH REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 1. Model Architecture
    report_lines.append("1. MODEL ARCHITECTURE")
    report_lines.append("-" * 80)
    report_lines.append("  Pipeline: Input (1400) → ARD-VAE → Latent SDE → Decoder → Output")
    report_lines.append("  SDE: dZ_t = μ_θ(Z_t) dt + σ_φ(Z_t) dW_t")
    report_lines.append("")
    
    # 2. Dimensionality Analysis
    report_lines.append("2. DIMENSIONALITY ANALYSIS (Latent Prism)")
    report_lines.append("-" * 80)
    
    active_dims = results['sde']['telemetry']['active_dimensions']
    report_lines.append(f"  Input Dimension: 1400 features")
    report_lines.append(f"  Latent Capacity: 64 dimensions (max)")
    report_lines.append(f"  Active Dimensions: {active_dims}")
    report_lines.append(f"  Compression Ratio: {1400/active_dims:.1f}x")
    report_lines.append(f"  Intrinsic Market Complexity: {active_dims} effective units")
    report_lines.append("")
    
    # 3. Physics Discovery
    report_lines.append("3. PHYSICS DISCOVERY (Drift DNA)")
    report_lines.append("-" * 80)
    
    physics_dna = results['sde']['telemetry']['physics_dna']
    
    if physics_dna:
        law_counts = {}
        for dim, law in physics_dna.items():
            law_counts[law] = law_counts.get(law, 0) + 1
        
        report_lines.append(f"  Discovered {len(physics_dna)} dominant physics laws:")
        for law, count in sorted(law_counts.items(), key=lambda x: -x[1]):
            report_lines.append(f"    {law}: {count} dimensions")
        
        report_lines.append("")
        report_lines.append("  Dimension-wise breakdown:")
        for dim, law in sorted(physics_dna.items()):
            report_lines.append(f"    Dim {dim:2d}: {law}")
    else:
        report_lines.append("  ⚠ No dominant physics laws found")
        report_lines.append("  Market is highly stochastic (diffusion-dominated)")
    
    report_lines.append("")
    
    # 4. Signal-to-Noise
    report_lines.append("4. SIGNAL-TO-NOISE ANALYSIS")
    report_lines.append("-" * 80)
    
    snr = results['sde']['telemetry']['signal_to_noise']
    report_lines.append(f"  Signal-to-Noise Ratio: {snr:.4f}")
    
    if snr > 1.0:
        regime = "Drift-Dominated (Deterministic Trends)"
    elif snr > 0.5:
        regime = "Balanced Drift-Diffusion"
    else:
        regime = "Diffusion-Dominated (High Uncertainty)"
    
    report_lines.append(f"  Market Regime: {regime}")
    report_lines.append("")
    
    # 5. Uncertainty Quantification
    report_lines.append("5. UNCERTAINTY QUANTIFICATION")
    report_lines.append("-" * 80)
    
    uncertainty = results['sde']['uncertainty']
    report_lines.append(f"  Mean Uncertainty: {uncertainty.mean():.4f}")
    report_lines.append(f"  Std Uncertainty: {uncertainty.std():.4f}")
    report_lines.append(f"  Min Uncertainty: {uncertainty.min():.4f}")
    report_lines.append(f"  Max Uncertainty: {uncertainty.max():.4f}")
    
    # Uncertainty vs Correctness
    y_pred_class = results['sde']['y_pred_class']
    correct = (y_pred_class == y_test)
    
    report_lines.append(f"\n  Uncertainty by Prediction Correctness:")
    report_lines.append(f"    Correct Predictions: σ = {uncertainty[correct].mean():.4f}")
    report_lines.append(f"    Incorrect Predictions: σ = {uncertainty[~correct].mean():.4f}")
    report_lines.append("")
    
    # 6. Performance Comparison
    report_lines.append("6. PERFORMANCE COMPARISON")
    report_lines.append("-" * 80)
    
    report_lines.append("  LaP-SDE:")
    report_lines.append(f"    ROC AUC: {results['sde']['auc']:.4f}")
    report_lines.append(f"    PR AUC: {results['sde']['pr_auc']:.4f}")
    
    report_lines.append("\n  Hybrid ODE (Baseline):")
    report_lines.append(f"    ROC AUC: {results['ode']['auc']:.4f}")
    report_lines.append(f"    PR AUC: {results['ode']['pr_auc']:.4f}")
    
    improvement = (results['sde']['auc'] - results['ode']['auc']) / results['ode']['auc'] * 100
    report_lines.append(f"\n  Improvement: {improvement:+.2f}%")
    report_lines.append("")
    
    # 7. Classification Reports
    report_lines.append("7. DETAILED CLASSIFICATION METRICS")
    report_lines.append("-" * 80)
    
    report_lines.append("\n  LaP-SDE:")
    report_lines.append(classification_report(y_test, results['sde']['y_pred_class'], 
                                              target_names=['Down', 'Up']))
    
    report_lines.append("\n  Hybrid ODE:")
    report_lines.append(classification_report(y_test, results['ode']['y_pred_class'], 
                                              target_names=['Down', 'Up']))
    
    # 8. Conclusion
    report_lines.append("8. CONCLUSION")
    report_lines.append("-" * 80)
    
    if results['sde']['auc'] > results['ode']['auc']:
        report_lines.append("  ✓ LaP-SDE outperforms Hybrid ODE")
    else:
        report_lines.append("  ⚠ LaP-SDE underperforms Hybrid ODE")
    
    report_lines.append(f"  ✓ Discovered intrinsic dimensionality: {active_dims} (from 1400)")
    report_lines.append(f"  ✓ Uncertainty quantification enables risk-aware trading")
    
    if physics_dna:
        report_lines.append(f"  ✓ Interpretable physics laws discovered: {len(physics_dna)}")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # Print and save report
    report_text = "\n".join(report_lines)
    print(report_text)
    
    output_path = Path('artifacts') / 'sde_research_report.txt'
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(report_text)
    
    print(f"\n  ✓ Saved report to {output_path}")


def main():
    """Main research pipeline."""
    print("\n" + "=" * 80)
    print("LATENT PHYSICS-INFORMED SDE (LaP-SDE) RESEARCH")
    print("=" * 80)
    print("\nObjective: Validate SDE model with ARD dimensionality reduction")
    print("Expected: 1400 features → ~12-20 intrinsic dimensions")
    print("")
    
    # 1. Prepare data
    X, y, features = prepare_data()
    
    # 2. Train/test split
    print(f"\nSplitting data: {TRAIN_SPLIT*100:.0f}% train, {(1-TRAIN_SPLIT)*100:.0f}% test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=TRAIN_SPLIT, random_state=RANDOM_SEED, stratify=y
    )
    
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Test: {len(X_test):,} samples")
    
    # 3. Train models
    results = train_models(X_train, y_train, X_test, y_test)
    
    # 4. Analyze telemetry
    analyze_telemetry(results)
    
    # 5. Visualize results
    visualize_results(results, y_test)
    
    # 6. Generate report
    generate_report(results, y_test)
    
    print("\n" + "=" * 80)
    print("RESEARCH COMPLETE")
    print("=" * 80)
    print("\nArtifacts generated:")
    print("  - artifacts/sde_research_results.png")
    print("  - artifacts/sde_research_report.txt")
    print("")


if __name__ == "__main__":
    main()
