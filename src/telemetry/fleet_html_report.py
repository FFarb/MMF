"""
Individual Fleet HTML Telemetry Generator.

Generates comprehensive single-file HTML reports with:
- Overall fleet summary
- Per-asset performance tables
- Per-fold graphs (embedded as Base64)
- Interactive analysis tools

Author: QFC System v4.0 - Individual Fleet Telemetry
"""

import base64
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to Base64 string for embedding in HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#1e1e2e')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"


def generate_fold_chart(
    fold_data: Dict,
    asset: str,
    fold_idx: int,
) -> str:
    """Generate a fold analysis chart and return as Base64."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='#1e1e2e')
    plt.rcParams.update({'text.color': 'white', 'axes.labelcolor': 'white', 
                         'xtick.color': 'white', 'ytick.color': 'white'})
    
    timestamps = fold_data.get('timestamps', [])
    y_true = np.array(fold_data.get('y_true', []))
    y_pred = np.array(fold_data.get('y_pred', []))
    y_proba = np.array(fold_data.get('y_proba', []))
    close_prices = np.array(fold_data.get('close_prices', []))
    threshold = fold_data.get('threshold', 0.55)
    
    # Chart 1: Price with predictions
    ax1 = axes[0, 0]
    ax1.set_facecolor('#2d2d3d')
    
    if len(timestamps) > 0 and len(close_prices) > 0:
        ax1.plot(range(len(close_prices)), close_prices, 'w-', alpha=0.7, linewidth=1, label='Price')
        
        # Mark predictions
        tp_mask = (y_pred == 1) & (y_true == 1)
        fp_mask = (y_pred == 1) & (y_true == 0)
        fn_mask = (y_pred == 0) & (y_true == 1)
        
        ax1.scatter(np.where(tp_mask)[0], close_prices[tp_mask], c='#00ff88', marker='^', 
                   s=80, label=f'TP ({tp_mask.sum()})', zorder=5)
        ax1.scatter(np.where(fp_mask)[0], close_prices[fp_mask], c='#ff4444', marker='v', 
                   s=80, label=f'FP ({fp_mask.sum()})', zorder=5)
        ax1.scatter(np.where(fn_mask)[0], close_prices[fn_mask], c='#ffaa00', marker='x', 
                   s=80, label=f'FN ({fn_mask.sum()})', zorder=5)
    
    ax1.set_title(f'{asset} - Fold {fold_idx}: Price & Decisions', color='white', fontsize=12)
    ax1.legend(loc='upper left', facecolor='#2d2d3d', labelcolor='white', fontsize=9)
    ax1.grid(True, alpha=0.2)
    
    # Chart 2: Probability distribution
    ax2 = axes[0, 1]
    ax2.set_facecolor('#2d2d3d')
    
    if len(y_proba) > 0:
        ax2.plot(range(len(y_proba)), y_proba, 'c-', alpha=0.7, linewidth=1.5, label='Probability')
        ax2.axhline(y=threshold, color='#ff4444', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.2f})')
        ax2.fill_between(range(len(y_proba)), 0, y_proba, where=(y_proba >= threshold), 
                        color='#00ff88', alpha=0.3, label='Signal Zone')
    
    ax2.set_ylim([0, 1])
    ax2.set_title(f'Probability Over Time', color='white', fontsize=12)
    ax2.legend(loc='upper right', facecolor='#2d2d3d', labelcolor='white', fontsize=9)
    ax2.grid(True, alpha=0.2)
    
    # Chart 3: Confusion Matrix
    ax3 = axes[1, 0]
    ax3.set_facecolor('#2d2d3d')
    
    if len(y_true) > 0 and len(y_pred) > 0:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                   xticklabels=['Pred 0', 'Pred 1'],
                   yticklabels=['True 0', 'True 1'],
                   cbar_kws={'label': 'Count'})
    
    ax3.set_title('Confusion Matrix', color='white', fontsize=12)
    
    # Chart 4: Probability histogram
    ax4 = axes[1, 1]
    ax4.set_facecolor('#2d2d3d')
    
    if len(y_proba) > 0:
        ax4.hist(y_proba[y_true == 0], bins=20, alpha=0.6, color='#ff4444', label='True Negative')
        ax4.hist(y_proba[y_true == 1], bins=20, alpha=0.6, color='#00ff88', label='True Positive')
        ax4.axvline(x=threshold, color='white', linestyle='--', linewidth=2)
    
    ax4.set_xlabel('Probability', color='white')
    ax4.set_ylabel('Frequency', color='white')
    ax4.set_title('Probability Distribution by Class', color='white', fontsize=12)
    ax4.legend(loc='upper right', facecolor='#2d2d3d', labelcolor='white', fontsize=9)
    ax4.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    return fig_to_base64(fig)


def generate_asset_summary_chart(fold_metrics: List[Dict], asset: str) -> str:
    """Generate asset summary chart showing metrics across folds."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='#1e1e2e')
    
    folds = [f['fold'] for f in fold_metrics]
    precision = [f['precision'] for f in fold_metrics]
    recall = [f['recall'] for f in fold_metrics]
    f1 = [f['f1'] for f in fold_metrics]
    expectancy = [f['expectancy'] for f in fold_metrics]
    
    # Chart 1: Precision/Recall/F1 by fold
    ax1 = axes[0]
    ax1.set_facecolor('#2d2d3d')
    
    x = np.arange(len(folds))
    width = 0.25
    
    ax1.bar(x - width, precision, width, label='Precision', color='#00ff88', alpha=0.8)
    ax1.bar(x, recall, width, label='Recall', color='#00aaff', alpha=0.8)
    ax1.bar(x + width, f1, width, label='F1', color='#ff88ff', alpha=0.8)
    
    ax1.set_xlabel('Fold', color='white')
    ax1.set_ylabel('Score', color='white')
    ax1.set_title(f'{asset} - Metrics by Fold', color='white', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'F{f}' for f in folds])
    ax1.legend(facecolor='#2d2d3d', labelcolor='white')
    ax1.grid(True, alpha=0.2, axis='y')
    ax1.tick_params(colors='white')
    
    # Chart 2: Expectancy by fold
    ax2 = axes[1]
    ax2.set_facecolor('#2d2d3d')
    
    colors = ['#00ff88' if e > 0 else '#ff4444' for e in expectancy]
    ax2.bar(folds, expectancy, color=colors, alpha=0.8, edgecolor='white')
    ax2.axhline(y=0, color='white', linestyle='-', linewidth=1)
    
    ax2.set_xlabel('Fold', color='white')
    ax2.set_ylabel('Expectancy', color='white')
    ax2.set_title(f'{asset} - Expectancy by Fold', color='white', fontsize=12)
    ax2.grid(True, alpha=0.2, axis='y')
    ax2.tick_params(colors='white')
    
    plt.tight_layout()
    
    return fig_to_base64(fig)


def generate_fleet_summary_chart(all_results: List[Dict]) -> str:
    """Generate fleet-wide summary chart."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='#1e1e2e')
    
    assets = [r['asset'] for r in all_results]
    avg_precision = [r['avg_precision'] for r in all_results]
    avg_expectancy = [r['avg_expectancy'] for r in all_results]
    avg_f1 = [r['avg_f1'] for r in all_results]
    
    # Chart 1: Precision & F1 by asset
    ax1 = axes[0]
    ax1.set_facecolor('#2d2d3d')
    
    x = np.arange(len(assets))
    width = 0.35
    
    ax1.bar(x - width/2, avg_precision, width, label='Precision', color='#00ff88', alpha=0.8)
    ax1.bar(x + width/2, avg_f1, width, label='F1', color='#ff88ff', alpha=0.8)
    
    ax1.axhline(y=0.55, color='#ffaa00', linestyle='--', linewidth=2, label='55% Target')
    
    ax1.set_xlabel('Asset', color='white')
    ax1.set_ylabel('Score', color='white')
    ax1.set_title('Fleet Performance - Precision & F1', color='white', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(assets, rotation=45, ha='right')
    ax1.legend(facecolor='#2d2d3d', labelcolor='white')
    ax1.grid(True, alpha=0.2, axis='y')
    ax1.tick_params(colors='white')
    
    # Chart 2: Expectancy by asset
    ax2 = axes[1]
    ax2.set_facecolor('#2d2d3d')
    
    colors = ['#00ff88' if e > 0 else '#ff4444' for e in avg_expectancy]
    ax2.bar(assets, avg_expectancy, color=colors, alpha=0.8, edgecolor='white')
    ax2.axhline(y=0, color='white', linestyle='-', linewidth=1)
    
    ax2.set_xlabel('Asset', color='white')
    ax2.set_ylabel('Avg Expectancy', color='white')
    ax2.set_title('Fleet Performance - Expectancy', color='white', fontsize=12)
    ax2.tick_params(colors='white', labelrotation=45)
    for label in ax2.get_xticklabels():
        label.set_ha('right')
    ax2.grid(True, alpha=0.2, axis='y')
    
    plt.tight_layout()
    
    return fig_to_base64(fig)


def generate_individual_fleet_html(
    all_results: List[Dict],
    asset_fold_data: Dict[str, List[Dict]],
    output_path: Path,
    training_params: Dict = None,
):
    """
    Generate comprehensive HTML telemetry report.
    
    Parameters
    ----------
    all_results : list of dict
        Overall results per asset
    asset_fold_data : dict
        Fold-level data per asset: {asset: [fold1_data, fold2_data, ...]}
    output_path : Path
        Where to save the HTML file
    training_params : dict, optional
        Training configuration parameters
    """
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Generate charts
    fleet_summary_chart = generate_fleet_summary_chart(all_results) if all_results else ""
    
    asset_sections = []
    
    for asset, fold_data_list in asset_fold_data.items():
        # Generate asset summary chart
        fold_metrics = [
            {
                'fold': fd['fold'],
                'precision': fd['precision'],
                'recall': fd['recall'],
                'f1': fd['f1'],
                'expectancy': fd['expectancy'],
            }
            for fd in fold_data_list
        ]
        
        asset_summary_chart = generate_asset_summary_chart(fold_metrics, asset)
        
        # Generate per-fold charts
        fold_charts_html = ""
        fold_tables_html = ""
        
        for fd in fold_data_list:
            fold_idx = fd['fold']
            fold_chart = generate_fold_chart(fd, asset, fold_idx)
            
            fold_charts_html += f"""
            <div class="fold-section">
                <h4>Fold {fold_idx}</h4>
                <div class="fold-metrics">
                    <span class="metric"><b>Precision:</b> {fd['precision']:.2%}</span>
                    <span class="metric"><b>Recall:</b> {fd['recall']:.2%}</span>
                    <span class="metric"><b>F1:</b> {fd['f1']:.2%}</span>
                    <span class="metric"><b>Accuracy:</b> {fd.get('accuracy', 0):.2%}</span>
                    <span class="metric"><b>ROC AUC:</b> {fd.get('roc_auc', 0):.4f}</span>
                    <span class="metric expectancy {'positive' if fd['expectancy'] > 0 else 'negative'}">
                        <b>Expectancy:</b> {fd['expectancy']:.5f}
                    </span>
                </div>
                <div class="fold-chart">
                    <img src="{fold_chart}" alt="Fold {fold_idx} Analysis">
                </div>
                <div class="fold-analysis">
                    <h5>Human Analysis Notes:</h5>
                    <textarea class="analysis-notes" placeholder="Add your analysis for Fold {fold_idx} here... Why did it perform this way?"></textarea>
                </div>
            </div>
            """
        
        # Calculate asset summary
        asset_result = next((r for r in all_results if r['asset'] == asset), None)
        if asset_result:
            avg_precision = asset_result['avg_precision']
            avg_f1 = asset_result['avg_f1']
            avg_expectancy = asset_result['avg_expectancy']
            status_class = 'success' if avg_expectancy > 0 else 'warning'
        else:
            avg_precision = avg_f1 = avg_expectancy = 0
            status_class = 'warning'
        
        asset_sections.append(f"""
        <div class="asset-card">
            <div class="asset-header {status_class}">
                <h3>{asset}</h3>
                <div class="asset-summary">
                    <span>Precision: {avg_precision:.2%}</span>
                    <span>F1: {avg_f1:.2%}</span>
                    <span>Expectancy: {avg_expectancy:.5f}</span>
                </div>
            </div>
            <div class="asset-content">
                <div class="asset-summary-chart">
                    <img src="{asset_summary_chart}" alt="{asset} Summary">
                </div>
                <div class="folds-container">
                    {fold_charts_html}
                </div>
            </div>
        </div>
        """)
    
    # Build summary table
    summary_rows = ""
    for r in sorted(all_results, key=lambda x: x['avg_expectancy'], reverse=True):
        status = 'positive' if r['avg_expectancy'] > 0 else 'negative'
        summary_rows += f"""
        <tr class="{status}">
            <td>{r['asset']}</td>
            <td>{r['avg_precision']:.2%}</td>
            <td>{r.get('avg_recall', 0):.2%}</td>
            <td>{r['avg_f1']:.2%}</td>
            <td>{r.get('avg_roc_auc', 0):.4f}</td>
            <td class="{status}">{r['avg_expectancy']:.5f}</td>
            <td>{r.get('folds', 5)}</td>
        </tr>
        """
    
    # Training params section
    params_html = ""
    if training_params:
        params_html = "<h3>Training Parameters</h3><ul>"
        for k, v in training_params.items():
            params_html += f"<li><b>{k}:</b> {v}</li>"
        params_html += "</ul>"
    
    # Count statistics
    profitable_count = sum(1 for r in all_results if r['avg_expectancy'] > 0)
    total_count = len(all_results)
    avg_fleet_precision = np.mean([r['avg_precision'] for r in all_results]) if all_results else 0
    avg_fleet_expectancy = np.mean([r['avg_expectancy'] for r in all_results]) if all_results else 0
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Individual Fleet Training Report - {timestamp}</title>
    <style>
        :root {{
            --bg-primary: #1e1e2e;
            --bg-secondary: #2d2d3d;
            --bg-card: #353545;
            --text-primary: #ffffff;
            --text-secondary: #b0b0b0;
            --accent-green: #00ff88;
            --accent-red: #ff4444;
            --accent-blue: #00aaff;
            --accent-purple: #ff88ff;
            --accent-yellow: #ffaa00;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 20px;
        }}
        
        .header {{
            text-align: center;
            padding: 30px;
            background: linear-gradient(135deg, var(--bg-secondary), var(--bg-card));
            border-radius: 12px;
            margin-bottom: 30px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(90deg, var(--accent-green), var(--accent-blue));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .header .timestamp {{
            color: var(--text-secondary);
            font-size: 0.9em;
        }}
        
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: var(--bg-secondary);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        .stat-card h3 {{
            font-size: 2em;
            margin-bottom: 5px;
        }}
        
        .stat-card.positive h3 {{ color: var(--accent-green); }}
        .stat-card.negative h3 {{ color: var(--accent-red); }}
        .stat-card.neutral h3 {{ color: var(--accent-blue); }}
        
        .stat-card p {{
            color: var(--text-secondary);
            font-size: 0.9em;
        }}
        
        .fleet-chart {{
            background: var(--bg-secondary);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        
        .fleet-chart img {{
            max-width: 100%;
            border-radius: 8px;
        }}
        
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--bg-secondary);
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 30px;
        }}
        
        .summary-table th, .summary-table td {{
            padding: 12px 15px;
            text-align: center;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        
        .summary-table th {{
            background: var(--bg-card);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }}
        
        .summary-table tr:hover {{
            background: rgba(255,255,255,0.05);
        }}
        
        .summary-table tr.positive td:last-child {{ color: var(--accent-green); }}
        .summary-table tr.negative td:last-child {{ color: var(--accent-red); }}
        
        .asset-card {{
            background: var(--bg-secondary);
            border-radius: 12px;
            margin-bottom: 30px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        .asset-header {{
            padding: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        
        .asset-header.success {{
            background: linear-gradient(90deg, rgba(0,255,136,0.2), transparent);
        }}
        
        .asset-header.warning {{
            background: linear-gradient(90deg, rgba(255,68,68,0.2), transparent);
        }}
        
        .asset-header h3 {{
            font-size: 1.5em;
        }}
        
        .asset-summary span {{
            margin-left: 20px;
            padding: 5px 12px;
            background: var(--bg-card);
            border-radius: 20px;
            font-size: 0.9em;
        }}
        
        .asset-content {{
            padding: 20px;
        }}
        
        .asset-summary-chart {{
            text-align: center;
            margin-bottom: 20px;
        }}
        
        .asset-summary-chart img {{
            max-width: 100%;
            border-radius: 8px;
        }}
        
        .fold-section {{
            background: var(--bg-card);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        
        .fold-section h4 {{
            margin-bottom: 15px;
            color: var(--accent-blue);
        }}
        
        .fold-metrics {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-bottom: 15px;
        }}
        
        .fold-metrics .metric {{
            padding: 8px 15px;
            background: var(--bg-secondary);
            border-radius: 8px;
            font-size: 0.9em;
        }}
        
        .fold-metrics .metric.positive {{ color: var(--accent-green); }}
        .fold-metrics .metric.negative {{ color: var(--accent-red); }}
        
        .fold-chart {{
            text-align: center;
            margin-bottom: 15px;
        }}
        
        .fold-chart img {{
            max-width: 100%;
            border-radius: 8px;
        }}
        
        .fold-analysis {{
            margin-top: 15px;
        }}
        
        .fold-analysis h5 {{
            margin-bottom: 10px;
            color: var(--accent-yellow);
        }}
        
        .analysis-notes {{
            width: 100%;
            min-height: 80px;
            padding: 12px;
            background: var(--bg-secondary);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 8px;
            color: var(--text-primary);
            font-family: inherit;
            resize: vertical;
        }}
        
        .analysis-notes:focus {{
            outline: none;
            border-color: var(--accent-blue);
        }}
        
        .params-section {{
            background: var(--bg-secondary);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        
        .params-section h3 {{
            margin-bottom: 15px;
            color: var(--accent-purple);
        }}
        
        .params-section ul {{
            list-style: none;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 10px;
        }}
        
        .params-section li {{
            padding: 8px 12px;
            background: var(--bg-card);
            border-radius: 6px;
        }}
        
        .footer {{
            text-align: center;
            padding: 20px;
            color: var(--text-secondary);
            font-size: 0.85em;
        }}
        
        @media (max-width: 768px) {{
            .asset-header {{
                flex-direction: column;
                text-align: center;
            }}
            
            .asset-summary {{
                margin-top: 15px;
            }}
            
            .asset-summary span {{
                margin: 5px;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Individual Fleet Training Report</h1>
        <p class="timestamp">Generated: {timestamp}</p>
    </div>
    
    <div class="summary-stats">
        <div class="stat-card {'positive' if profitable_count == total_count else 'neutral'}">
            <h3>{profitable_count}/{total_count}</h3>
            <p>Profitable Assets</p>
        </div>
        <div class="stat-card {'positive' if avg_fleet_precision >= 0.55 else 'warning'}">
            <h3>{avg_fleet_precision:.1%}</h3>
            <p>Avg Fleet Precision</p>
        </div>
        <div class="stat-card {'positive' if avg_fleet_expectancy > 0 else 'negative'}">
            <h3>{avg_fleet_expectancy:.5f}</h3>
            <p>Avg Fleet Expectancy</p>
        </div>
        <div class="stat-card neutral">
            <h3>{len(asset_fold_data.get(list(asset_fold_data.keys())[0] if asset_fold_data else '', []))}</h3>
            <p>CV Folds</p>
        </div>
    </div>
    
    {f'<div class="fleet-chart"><h3>Fleet Overview</h3><img src="{fleet_summary_chart}" alt="Fleet Summary"></div>' if fleet_summary_chart else ''}
    
    <div class="params-section">
        {params_html}
    </div>
    
    <h2 style="margin-bottom: 20px;">Summary Table</h2>
    <table class="summary-table">
        <thead>
            <tr>
                <th>Asset</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1</th>
                <th>ROC AUC</th>
                <th>Expectancy</th>
                <th>Folds</th>
            </tr>
        </thead>
        <tbody>
            {summary_rows}
        </tbody>
    </table>
    
    <h2 style="margin-bottom: 20px;">Per-Asset Analysis</h2>
    {''.join(asset_sections)}
    
    <div class="footer">
        <p>QFC System v4.0 - Individual Fleet Telemetry</p>
        <p>Total Assets: {total_count} | Profitable: {profitable_count} ({profitable_count/total_count*100 if total_count > 0 else 0:.1f}%)</p>
    </div>
    
    <script>
        // Save notes to localStorage
        document.querySelectorAll('.analysis-notes').forEach((textarea, idx) => {{
            const key = `fleet_notes_${{idx}}`;
            textarea.value = localStorage.getItem(key) || '';
            textarea.addEventListener('input', () => {{
                localStorage.setItem(key, textarea.value);
            }});
        }});
    </script>
</body>
</html>"""
    
    # Save HTML
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n[HTML Report] Saved to: {output_path}")
    
    return output_path
