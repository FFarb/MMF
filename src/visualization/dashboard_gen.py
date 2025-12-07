"""
Dashboard Generator for Trading Simulation.

Generates a self-contained HTML dashboard with:
- Equity curve vs BTC benchmark
- Active positions grid
- Signal probability vs SDE uncertainty panel
- Embedded Chart.js for visualization

The dashboard is a single HTML file with all data embedded.
No external dependencies required (uses CDN for Chart.js).

Author: QFC System v3.1 - Autonomous Trading Layer
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class DashboardGenerator:
    """
    Generate self-contained HTML trading dashboard.
    
    Creates a single HTML file with embedded data and Chart.js visualizations.
    
    Parameters
    ----------
    output_path : Path or str
        Path to save dashboard HTML file
    """
    
    def __init__(self, output_path: Path | str):
        self.output_path = Path(output_path)
    
    def generate(
        self,
        equity_history: List[Dict],
        active_positions: List[Dict],
        signals: List[Dict],
        account_summary: Dict,
    ):
        """
        Generate complete dashboard HTML.
        
        Parameters
        ----------
        equity_history : list of dict
            Equity curve data: [{'timestamp': ..., 'equity': ..., 'btc_price': ...}, ...]
        active_positions : list of dict
            Active positions: [{'symbol': ..., 'side': ..., 'pnl': ..., ...}, ...]
        signals : list of dict
            Latest signals: [{'symbol': ..., 'probability': ..., 'sigma': ..., ...}, ...]
        account_summary : dict
            Account metrics: {'balance': ..., 'equity': ..., ...}
        """
        html = self._build_html(equity_history, active_positions, signals, account_summary)
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"[Dashboard] Generated: {self.output_path}")
    
    def _build_html(
        self,
        equity_history: List[Dict],
        active_positions: List[Dict],
        signals: List[Dict],
        account_summary: Dict,
    ) -> str:
        """Build complete HTML document."""
        
        # Prepare data for Chart.js
        equity_data = self._prepare_equity_data(equity_history)
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Simulation Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        h1 {{
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            border: 1px solid rgba(255, 255, 255, 0.18);
        }}
        
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.8;
            margin-bottom: 10px;
        }}
        
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
        }}
        
        .stat-value.positive {{
            color: #4ade80;
        }}
        
        .stat-value.negative {{
            color: #f87171;
        }}
        
        .chart-container {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }}
        
        .chart-container h2 {{
            color: #333;
            margin-bottom: 20px;
        }}
        
        table {{
            width: 100%;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            margin-bottom: 30px;
        }}
        
        th, td {{
            padding: 15px;
            text-align: left;
            color: #333;
        }}
        
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            font-weight: bold;
        }}
        
        tr:nth-child(even) {{
            background: rgba(0, 0, 0, 0.05);
        }}
        
        .long {{
            color: #10b981;
            font-weight: bold;
        }}
        
        .short {{
            color: #ef4444;
            font-weight: bold;
        }}
        
        .timestamp {{
            text-align: center;
            margin-top: 20px;
            opacity: 0.8;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Autonomous Trading Simulation</h1>
        
        <!-- Account Summary -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Equity</div>
                <div class="stat-value">${account_summary.get('equity', 0):,.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Balance</div>
                <div class="stat-value">${account_summary.get('balance', 0):,.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Unrealized PnL</div>
                <div class="stat-value {'positive' if account_summary.get('unrealized_pnl', 0) >= 0 else 'negative'}">
                    ${account_summary.get('unrealized_pnl', 0):+,.2f}
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total PnL</div>
                <div class="stat-value {'positive' if account_summary.get('total_pnl', 0) >= 0 else 'negative'}">
                    ${account_summary.get('total_pnl', 0):+,.2f}
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Win Rate</div>
                <div class="stat-value">{account_summary.get('win_rate', 0)*100:.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Trades</div>
                <div class="stat-value">{account_summary.get('total_trades', 0)}</div>
            </div>
        </div>
        
        <!-- Equity Curve -->
        <div class="chart-container">
            <h2>📈 Equity Curve</h2>
            <canvas id="equityChart"></canvas>
        </div>
        
        <!-- Active Positions -->
        <h2 style="margin-bottom: 15px;">💼 Active Positions ({len(active_positions)})</h2>
        {self._build_positions_table(active_positions)}
        
        <!-- Latest Signals -->
        <h2 style="margin-bottom: 15px;">🎯 Latest Signals</h2>
        {self._build_signals_table(signals)}
        
        <div class="timestamp">
            Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
    
    <script>
        // Equity Chart
        const ctx = document.getElementById('equityChart').getContext('2d');
        const equityChart = new Chart(ctx, {{
            type: 'line',
            data: {equity_data},
            options: {{
                responsive: true,
                maintainAspectRatio: true,
                interaction: {{
                    mode: 'index',
                    intersect: false,
                }},
                plugins: {{
                    legend: {{
                        display: true,
                        position: 'top',
                    }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                let label = context.dataset.label || '';
                                if (label) {{
                                    label += ': ';
                                }}
                                if (context.parsed.y !== null) {{
                                    label += '$' + context.parsed.y.toLocaleString('en-US', {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
                                }}
                                return label;
                            }}
                        }}
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: false,
                        ticks: {{
                            callback: function(value) {{
                                return '$' + value.toLocaleString();
                            }}
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
        
        return html
    
    def _prepare_equity_data(self, equity_history: List[Dict]) -> str:
        """Prepare equity data for Chart.js."""
        if not equity_history:
            return json.dumps({
                'labels': [],
                'datasets': []
            })
        
        labels = [entry['timestamp'] for entry in equity_history]
        equity_values = [entry['equity'] for entry in equity_history]
        
        data = {
            'labels': labels,
            'datasets': [
                {
                    'label': 'Account Equity',
                    'data': equity_values,
                    'borderColor': 'rgb(75, 192, 192)',
                    'backgroundColor': 'rgba(75, 192, 192, 0.1)',
                    'tension': 0.1,
                    'fill': True,
                }
            ]
        }
        
        # Add BTC benchmark if available
        if equity_history and 'btc_price' in equity_history[0]:
            btc_prices = [entry.get('btc_price', 0) for entry in equity_history]
            
            # Normalize BTC to start at same value as equity
            if btc_prices and btc_prices[0] > 0:
                initial_equity = equity_values[0]
                initial_btc = btc_prices[0]
                btc_normalized = [initial_equity * (price / initial_btc) for price in btc_prices]
                
                data['datasets'].append({
                    'label': 'BTC Benchmark',
                    'data': btc_normalized,
                    'borderColor': 'rgb(255, 159, 64)',
                    'backgroundColor': 'rgba(255, 159, 64, 0.1)',
                    'tension': 0.1,
                    'fill': False,
                    'borderDash': [5, 5],
                })
        
        return json.dumps(data)
    
    def _build_positions_table(self, positions: List[Dict]) -> str:
        """Build HTML table for active positions."""
        if not positions:
            return "<p style='text-align: center; opacity: 0.7;'>No active positions</p>"
        
        rows = []
        for pos in positions:
            side_class = 'long' if pos['side'] == 'LONG' else 'short'
            pnl_class = 'positive' if pos.get('unrealized_pnl', 0) >= 0 else 'negative'
            
            rows.append(f"""
            <tr>
                <td><strong>{pos['symbol']}</strong></td>
                <td class="{side_class}">{pos['side']}</td>
                <td>${pos.get('size_usd', 0):,.0f}</td>
                <td>{pos.get('leverage', 1):.1f}x</td>
                <td>${pos.get('entry_price', 0):,.2f}</td>
                <td>${pos.get('current_price', 0):,.2f}</td>
                <td class="{pnl_class}">${pos.get('unrealized_pnl', 0):+,.2f}</td>
                <td>${pos.get('take_profit', 0):,.2f}</td>
                <td>${pos.get('stop_loss', 0):,.2f}</td>
                <td>{pos.get('risk_pct', 0)*100:.1f}%</td>
            </tr>
            """)
        
        return f"""
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Side</th>
                    <th>Size</th>
                    <th>Leverage</th>
                    <th>Entry</th>
                    <th>Current</th>
                    <th>PnL</th>
                    <th>TP</th>
                    <th>SL</th>
                    <th>Risk %</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        """
    
    def _build_signals_table(self, signals: List[Dict]) -> str:
        """Build HTML table for latest signals."""
        if not signals:
            return "<p style='text-align: center; opacity: 0.7;'>No signals available</p>"
        
        rows = []
        for sig in signals:
            prob_pct = sig.get('probability', 0) * 100
            sigma = sig.get('sigma', 0)
            
            # Color code by probability
            if prob_pct >= 70:
                prob_class = 'positive'
            elif prob_pct >= 55:
                prob_class = ''
            else:
                prob_class = 'negative'
            
            rows.append(f"""
            <tr>
                <td><strong>{sig['symbol']}</strong></td>
                <td class="{prob_class}">{prob_pct:.1f}%</td>
                <td>{sigma:.4f}</td>
                <td>{sig.get('expectancy', 0):.6f}</td>
                <td>${sig.get('close_price', 0):,.2f}</td>
            </tr>
            """)
        
        return f"""
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Probability</th>
                    <th>SDE σ</th>
                    <th>Expectancy</th>
                    <th>Price</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        """
