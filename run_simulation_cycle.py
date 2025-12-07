"""
Master Simulation Cycle - Autonomous Trading with Train-Trade-Retrain.

This is the main orchestration script that implements the complete lifecycle:
1. Initial Training: Build SDE-MoE fleet models
2. Trading Loop: Fetch data → Inference → Trade → Report
3. Retraining: Every 20 hours, retrain models on new data
4. Dashboard: Real-time HTML dashboard generation

Usage:
    # REALTIME mode (1-hour sleep between candles)
    python run_simulation_cycle.py --mode REALTIME
    
    # FAST mode (backtest, no sleep)
    python run_simulation_cycle.py --mode FAST --hours 48

Author: QFC System v3.1 - Autonomous Trading Layer
"""

from __future__ import annotations

import sys
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_loader import MarketDataLoader
from src.trading.risk_engine import FuturesRiskEngine
from src.trading.portfolio import PortfolioManager
from src.trading.paper_exchange import PaperExchange, Position
from src.trading.strategy_logic import decide_entry, get_latest_predictions
from src.visualization.dashboard_gen import DashboardGenerator

# Configuration
RETRAIN_INTERVAL = 20  # Hours between retraining
INITIAL_CAPITAL = 10000.0  # USDT
TRADING_FEE = 0.0005  # 0.05%

# Fleet assets
FLEET_ASSETS = [
    'BTCUSDT',
    'ETHUSDT',
    'SOLUSDT',
    'BNBUSDT',
    'XRPUSDT',
    'ADAUSDT',
    'DOGEUSDT',
    'AVAXUSDT',
    'LINKUSDT',
    'LTCUSDT',
]


class SimulationCycle:
    """
    Master simulation cycle orchestrator.
    
    Parameters
    ----------
    mode : str
        'REALTIME' or 'FAST'
    initial_capital : float
        Starting capital in USDT
    retrain_interval : int
        Hours between retraining
    artifacts_dir : Path
        Directory containing fleet predictions
    """
    
    def __init__(
        self,
        mode: str = 'REALTIME',
        initial_capital: float = INITIAL_CAPITAL,
        retrain_interval: int = RETRAIN_INTERVAL,
        artifacts_dir: Path = Path('artifacts/individual_fleet'),
    ):
        self.mode = mode
        self.initial_capital = initial_capital
        self.retrain_interval = retrain_interval
        self.artifacts_dir = artifacts_dir
        
        # Initialize components
        self.risk_engine = FuturesRiskEngine()
        self.portfolio_manager = PortfolioManager(
            fleet_summary_path=artifacts_dir / 'fleet_summary.csv',
            predictions_dir=artifacts_dir,
        )
        self.exchange = PaperExchange(
            initial_balance=initial_capital,
            trading_fee_pct=TRADING_FEE,
            state_file='wallet_state.json',
        )
        self.dashboard = DashboardGenerator(output_path='simulation_dashboard.html')
        
        # Tracking
        self.hour_counter = 0
        self.equity_history = []
        self.last_btc_price = None
        
        print("=" * 72)
        print("AUTONOMOUS TRADING SIMULATION - TRAIN-TRADE-RETRAIN")
        print("=" * 72)
        print(f"  Mode: {mode}")
        print(f"  Initial Capital: ${initial_capital:,.2f} USDT")
        print(f"  Retrain Interval: {retrain_interval} hours")
        print(f"  Fleet Assets: {len(FLEET_ASSETS)}")
        
        # Load historical data once for backtest
        self.historical_data = {}
        self.current_bar_index = 0
        
        print(f"  Loading historical M5 data...")
        for symbol in FLEET_ASSETS:
            try:
                loader = MarketDataLoader(symbol=symbol, interval='5')
                df = loader.get_data(days_back=30)  # Load 30 days of M5 data
                
                if df is not None and len(df) > 100:
                    self.historical_data[symbol] = df
            except Exception as e:
                print(f"    {symbol}: Failed - {e}")
        
        if self.historical_data:
            min_length = min(len(df) for df in self.historical_data.values())
            print(f"  Loaded {len(self.historical_data)} assets, {min_length} M5 bars ({min_length/12:.1f} hours)")
        
        print("=" * 72)
    
    def initial_training(self):
        """Run initial fleet training."""
        print("\n" + "=" * 72)
        print("STEP 0: INITIAL TRAINING")
        print("=" * 72)
        
        fleet_summary = self.artifacts_dir / 'fleet_summary.csv'
        
        if fleet_summary.exists():
            print("[Training] Fleet summary exists, skipping initial training")
            print("  To retrain from scratch, delete artifacts/individual_fleet/")
            return
        
        print("[Training] Running initial fleet training...")
        print("  This may take 30-60 minutes depending on your hardware...")
        
        # Run fleet training script
        result = subprocess.run(
            [sys.executable, 'run_individual_fleet.py', '--folds', '5'],
            capture_output=False,
        )
        
        if result.returncode != 0:
            print("[Training] ERROR: Fleet training failed!")
            sys.exit(1)
        
        print("[Training] [OK] Initial training complete")
    
    def fetch_latest_candles(self) -> Dict[str, Dict]:
        """
        Get current M5 bar from historical data.
        
        Returns
        -------
        candles : dict
            Current candle data by symbol
        """
        if self.current_bar_index >= min(len(df) for df in self.historical_data.values()):
            return {}  # End of data
        
        candles = {}
        
        for symbol, df in self.historical_data.items():
            if self.current_bar_index < len(df):
                row = df.iloc[self.current_bar_index]
                candles[symbol] = {
                    'timestamp': row.name,
                    'close': row['close'],
                    'high': row['high'],
                    'low': row['low'],
                    'volume': row['volume'],
                }
        
        self.current_bar_index += 1
        return candles
    
    def execute_trading_cycle(self, candles: Dict[str, Dict]):
        """
        Execute one trading cycle: Inference → Risk → Trade.
        
        Parameters
        ----------
        candles : dict
            Latest candle data by symbol
        """
        print("\n" + "-" * 72)
        print(f"TRADING CYCLE - M5 Bar {self.hour_counter} ({self.hour_counter * 5} minutes)")
        print("-" * 72)
        
        # Step 1: Get current prices
        current_prices = {symbol: data['close'] for symbol, data in candles.items()}
        
        # Step 1.5: Check dynamic exit conditions for open positions (NEW)
        for symbol, position in list(self.exchange.positions.items()):
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            
            # Get current sigma (placeholder - should come from model)
            # TODO: Extract from SDE model predictions
            current_sigma = max(0.01, 1.0 - 0.6)  # Placeholder
            
            # Check if we should exit
            should_exit, exit_reason, exit_size_pct = self.risk_engine.check_exit_conditions(
                position=position,
                current_price=current_price,
                current_sigma=current_sigma,
                entry_sigma=position.entry_sigma,
            )
            
            if should_exit:
                # Close position (partial or full)
                if exit_size_pct == 1.0:
                    # Full close
                    self.exchange.close_position(symbol, current_price, exit_reason)
                else:
                    # Partial close (reduce position size)
                    # For simplicity, we'll just close the full position
                    # In production, you'd reduce the position size
                    print(f"  [EXIT] {symbol}: Partial close not implemented, closing full position")
                    self.exchange.close_position(symbol, current_price, exit_reason)
        
        # Step 2: Update exchange with current prices (check TP/SL/Liquidation)
        timestamp = list(candles.values())[0]['timestamp'].isoformat() if candles else None
        closed_trades = self.exchange.update_prices(current_prices, timestamp)
        
        if closed_trades:
            print(f"\n[Exchange] {len(closed_trades)} positions closed")
        
        # Step 3: Get portfolio allocations
        equity = self.exchange.equity
        allocations = self.portfolio_manager.calculate_allocations(equity)
        
        if not allocations:
            print("[Portfolio] No valid allocations, skipping trading")
            return
        
        # Step 4: Execute trades
        print(f"\n[Trading] Executing {len(allocations)} signals...")
        
        for alloc in allocations:
            symbol = alloc.symbol
            
            # Skip if position already open
            if symbol in self.exchange.positions:
                print(f"  [Trading] {symbol}: Position already open, skipping")
                continue
            
            # Get current price
            if symbol not in current_prices:
                print(f"  [Trading] {symbol}: No price data, skipping")
                continue
            
            entry_price = current_prices[symbol]
            
            # Calculate trade parameters with risk engine
            # Note: We need SDE uncertainty (sigma) here
            # For now, use a placeholder based on probability
            # TODO: Extract sigma from SDE model predictions
            sigma_sde = max(0.01, 1.0 - alloc.probability)  # Rough approximation
            
            params = self.risk_engine.calculate_trade_parameters(
                symbol=symbol,
                side=alloc.side,
                entry_price=entry_price,
                sigma_sde=sigma_sde,
                equity=equity,
                allocation_pct=alloc.allocation_pct,
            )
            
            if params is None:
                print(f"  [Trading] {symbol}: Risk check failed, skipping")
                continue
            
            # Open position
            success = self.exchange.open_position(
                symbol=params.symbol,
                side=params.side,
                size_usd=params.position_size_usd,
                leverage=params.leverage,
                entry_price=params.entry_price,
                take_profit=params.take_profit,
                stop_loss=params.stop_loss,
                liquidation_price=params.liquidation_price,
                entry_sigma=sigma_sde,  # Store for volatility stop
                timestamp=timestamp,
            )
            
            if success:
                diagnostics = self.risk_engine.get_diagnostics(params)
                print(f"    Leverage: {diagnostics['leverage']:.1f}x | "
                      f"Risk: {diagnostics['risk_pct']:.1f}% | "
                      f"R:R: {diagnostics['risk_reward_ratio']:.1f}")
    
    def generate_dashboard(self, candles: Dict[str, Dict]):
        """
        Generate HTML dashboard.
        
        Parameters
        ----------
        candles : dict
            Latest candle data
        """
        # Get account summary
        account_summary = self.exchange.get_account_summary()
        
        # Track equity history
        timestamp = list(candles.values())[0]['timestamp'].isoformat() if candles else datetime.now().isoformat()
        btc_price = candles.get('BTCUSDT', {}).get('close', self.last_btc_price)
        
        if btc_price:
            self.last_btc_price = btc_price
        
        self.equity_history.append({
            'timestamp': timestamp,
            'equity': account_summary['equity'],
            'btc_price': btc_price,
        })
        
        # Prepare active positions data
        active_positions = []
        for symbol, pos in self.exchange.positions.items():
            current_price = candles.get(symbol, {}).get('close', pos.entry_price)
            
            active_positions.append({
                'symbol': pos.symbol,
                'side': pos.side,
                'size_usd': pos.size_usd,
                'leverage': pos.leverage,
                'entry_price': pos.entry_price,
                'current_price': current_price,
                'unrealized_pnl': pos.unrealized_pnl,
                'take_profit': pos.take_profit,
                'stop_loss': pos.stop_loss,
                'risk_pct': pos.unrealized_pnl / account_summary['equity'] if account_summary['equity'] > 0 else 0,
            })
        
        # Prepare signals data
        signals = []
        for symbol in FLEET_ASSETS:
            signal = self.portfolio_manager.get_latest_signal(symbol)
            if signal:
                signals.append({
                    'symbol': signal.symbol,
                    'probability': signal.probability,
                    'sigma': max(0.01, 1.0 - signal.probability),  # Placeholder
                    'expectancy': signal.expectancy,
                    'close_price': signal.close_price,
                })
        
        # Generate dashboard
        self.dashboard.generate(
            equity_history=self.equity_history,
            active_positions=active_positions,
            signals=signals,
            account_summary=account_summary,
        )
    
    def check_retrain(self):
        """Check if retraining is needed (every 20 hours = 240 M5 bars)."""
        bars_per_hour = 12  # 12 M5 bars per hour
        retrain_bars = self.retrain_interval * bars_per_hour
        
        if self.hour_counter >= retrain_bars:
            hours_elapsed = self.hour_counter / bars_per_hour
            print("\n" + "=" * 72)
            print("🔄 INITIATING RETRAINING...")
            print("=" * 72)
            print(f"  M5 Bars since last training: {self.hour_counter}")
            print(f"  Hours elapsed: {hours_elapsed:.1f}")
            print(f"  Running fleet retraining on new data...")
            
            # Run fleet training
            result = subprocess.run(
                [sys.executable, 'run_individual_fleet.py', '--folds', '3'],
                capture_output=False,
            )
            
            if result.returncode == 0:
                print("[Retrain] [OK] Retraining complete")
                
                # Reload portfolio manager
                self.portfolio_manager.reload_fleet_summary()
                
                # Reset counter
                self.hour_counter = 0
            else:
                print("[Retrain] WARNING: Retraining failed, continuing with old models")
    
    def run(self, max_hours: Optional[int] = None):
        """
        Run the main simulation loop.
        
        Parameters
        ----------
        max_hours : int, optional
            Maximum hours to simulate (None = infinite)
        """
        # Initial training
        self.initial_training()
        
        print("\n" + "=" * 72)
        print("STARTING SIMULATION LOOP")
        print("=" * 72)
        
        hours_simulated = 0
        
        try:
            while True:
                # Check if we've reached max hours
                if max_hours and hours_simulated >= max_hours:
                    print(f"\n[Simulation] Reached max hours ({max_hours}), stopping")
                    break
                
                # Step A: Get next M5 bar from historical data
                candles = self.fetch_latest_candles()
                
                if not candles:
                    print("[Data] No candles fetched, skipping cycle")
                else:
                    # Step B: Execute trading cycle
                    self.execute_trading_cycle(candles)
                    
                    # Step C: Generate dashboard
                    self.generate_dashboard(candles)
                    
                    # Step D: Save state
                    self.exchange.save_state()
                
                # Step E: Check retraining
                self.check_retrain()
                
                # Increment counters
                self.hour_counter += 1
                hours_simulated += 1
                
                # Step F: Sleep (if REALTIME mode)
                if self.mode == 'REALTIME':
                    print(f"\n[Sleep] Waiting 5 minutes until next M5 candle...")
                    time.sleep(300)  # 5 minutes (300 seconds)
                else:
                    # FAST mode: small delay for visibility
                    time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\n\n[Simulation] Interrupted by user")
        
        finally:
            # Final summary
            self.print_final_summary()
    
    def print_final_summary(self):
        """Print final simulation summary."""
        summary = self.exchange.get_account_summary()
        hours_simulated = self.hour_counter / 12  # 12 M5 bars per hour
        
        print("\n" + "=" * 72)
        print("SIMULATION SUMMARY")
        print("=" * 72)
        print(f"  Initial Capital:  ${self.initial_capital:,.2f}")
        print(f"  Final Equity:     ${summary['equity']:,.2f}")
        print(f"  Total PnL:        ${summary['total_pnl']:+,.2f}")
        print(f"  ROI:              {(summary['equity'] / self.initial_capital - 1) * 100:+.2f}%")
        print(f"  Total Trades:     {summary['total_trades']}")
        print(f"  Win Rate:         {summary['win_rate'] * 100:.1f}%")
        print(f"  Wins / Losses:    {summary['num_wins']} / {summary['num_losses']}")
        print(f"  M5 Bars Simulated: {self.hour_counter} ({hours_simulated:.1f} hours)")
        print("=" * 72)
        
        print(f"\n[Dashboard] View results: simulation_dashboard.html")
        print(f"[State] Wallet state saved: wallet_state.json")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Autonomous Trading Simulation with Train-Trade-Retrain (M5 Bars)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # REALTIME mode (5-minute sleep between M5 bars)
  python run_simulation_cycle.py --mode REALTIME
  
  # FAST mode (backtest 24 hours = 288 M5 bars)
  python run_simulation_cycle.py --mode FAST --bars 288
  
  # Custom retrain interval (retrain every 20 hours = 240 M5 bars)
  python run_simulation_cycle.py --mode FAST --bars 500 --retrain-interval 20

Note: Trading happens on M5 (5-minute) bars for realistic execution.
      Models are trained on H1 + M5 microstructure features.
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['REALTIME', 'FAST'],
        default='FAST',
        help='Simulation mode (default: FAST)'
    )
    parser.add_argument(
        '--bars',
        type=int,
        default=None,
        help='Maximum M5 bars to simulate (default: infinite). 12 bars = 1 hour, 288 bars = 1 day'
    )
    parser.add_argument(
        '--hours',
        type=int,
        default=None,
        help='Maximum hours to simulate (converted to M5 bars). Overrides --bars if specified.'
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=INITIAL_CAPITAL,
        help=f'Initial capital in USDT (default: {INITIAL_CAPITAL})'
    )
    parser.add_argument(
        '--retrain-interval',
        type=int,
        default=RETRAIN_INTERVAL,
        help=f'Hours between retraining (default: {RETRAIN_INTERVAL})'
    )
    
    args = parser.parse_args()
    
    # Convert hours to bars if specified
    max_bars = args.bars
    if args.hours:
        max_bars = args.hours * 12  # 12 M5 bars per hour
        print(f"[Config] Converting {args.hours} hours to {max_bars} M5 bars")
    
    # Create and run simulation
    sim = SimulationCycle(
        mode=args.mode,
        initial_capital=args.capital,
        retrain_interval=args.retrain_interval,
    )
    
    sim.run(max_hours=max_bars)  # Note: max_hours is actually max_bars now


if __name__ == '__main__':
    main()
