"""
Sniper Backtest Engine: H1 Strategy + M5 Execution.

This engine validates the "Sniper" trading architecture:
- H1 Model sets the bias (Long/Short/Neutral)
- M5 Physics (OU Mean Reversion) executes precise entries on dips/spikes
- Realistic trading constraints (fees, slippage, leverage)

Strategy: "H1 tells us WHAT to trade, M5 tells us WHEN to enter"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class TradingConfig:
    """Configuration for sniper backtest."""
    initial_capital: float = 10000.0
    leverage: float = 5.0  # Conservative (max 10x)
    taker_fee: float = 0.0006  # 0.06%
    slippage: float = 0.0002  # 0.02%
    position_size_pct: float = 0.10  # 10% of equity per trade
    ou_window: int = 24  # M5 bars for OU calculation
    ou_entry_long: float = -1.5  # Z-score threshold for long entry
    ou_entry_short: float = 1.5  # Z-score threshold for short entry
    use_sniper: bool = True  # If False, enter at H1 open (naive)
    strict_mode: bool = True  # If True, cancel if no signal by min 55


@dataclass
class Trade:
    """Record of a single trade."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    size: float  # Position size in base currency
    pnl: float
    pnl_pct: float
    fees: float
    slippage_cost: float
    h1_signal_time: pd.Timestamp
    entry_method: str  # 'sniper' or 'naive'
    ou_zscore_at_entry: Optional[float] = None


class SniperBacktestEngine:
    """
    Backtest engine for Sniper trading strategy.
    
    H1 Strategy + M5 Execution:
    - H1 model provides directional bias
    - M5 OU mean reversion provides precise entry timing
    - Exit at H1 close to isolate entry improvement
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.equity = config.initial_capital
        self.equity_curve = []
        self.trades: List[Trade] = []
        self.current_position = None
    
    def calculate_ou_zscore(self, prices: pd.Series, window: int = 24) -> pd.Series:
        """
        Calculate OU mean reversion Z-score.
        
        Z = (Price - RollingMean) / RollingStd
        
        Negative Z = Oversold (good for long entry)
        Positive Z = Overbought (good for short entry)
        """
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        
        zscore = (prices - rolling_mean) / (rolling_std + 1e-9)
        return zscore
    
    def find_sniper_entry(
        self,
        m5_bars: pd.DataFrame,
        direction: str,
        h1_signal_time: pd.Timestamp,
    ) -> Optional[Tuple[pd.Timestamp, float, float]]:
        """
        Find optimal M5 entry within the H1 bar using OU mean reversion.
        
        Parameters
        ----------
        m5_bars : pd.DataFrame
            M5 bars within the H1 period (should be ~12 bars)
        direction : str
            'long' or 'short'
        h1_signal_time : pd.Timestamp
            H1 bar timestamp
        
        Returns
        -------
        tuple or None
            (entry_time, entry_price, ou_zscore) if found, else None
        """
        if len(m5_bars) < 2:
            return None
        
        # Calculate OU Z-score on M5 closes
        m5_bars = m5_bars.copy()
        m5_bars['ou_zscore'] = self.calculate_ou_zscore(
            m5_bars['close'],
            window=self.config.ou_window
        )
        
        # Look for entry signal
        for idx, row in m5_bars.iterrows():
            if pd.isna(row['ou_zscore']):
                continue
            
            # Check if we're past minute 55 (strict mode)
            if self.config.strict_mode:
                minute = row.name.minute if hasattr(row.name, 'minute') else 0
                if minute >= 55:
                    # Timeout - no entry found
                    return None
            
            # Long entry: Wait for oversold dip
            if direction == 'long' and row['ou_zscore'] < self.config.ou_entry_long:
                return (row.name, row['close'], row['ou_zscore'])
            
            # Short entry: Wait for overbought spike
            if direction == 'short' and row['ou_zscore'] > self.config.ou_entry_short:
                return (row.name, row['close'], row['ou_zscore'])
        
        return None
    
    def execute_trade(
        self,
        entry_time: pd.Timestamp,
        exit_time: pd.Timestamp,
        direction: str,
        entry_price: float,
        exit_price: float,
        h1_signal_time: pd.Timestamp,
        entry_method: str,
        ou_zscore: Optional[float] = None,
    ) -> Trade:
        """Execute a trade and update equity."""
        
        # Calculate position size
        position_value = self.equity * self.config.position_size_pct * self.config.leverage
        size = position_value / entry_price
        
        # Apply slippage to entry
        entry_price_adj = entry_price * (1 + self.config.slippage if direction == 'long' else 1 - self.config.slippage)
        
        # Apply slippage to exit
        exit_price_adj = exit_price * (1 - self.config.slippage if direction == 'long' else 1 + self.config.slippage)
        
        # Calculate PnL
        if direction == 'long':
            pnl_pct = (exit_price_adj - entry_price_adj) / entry_price_adj
        else:  # short
            pnl_pct = (entry_price_adj - exit_price_adj) / entry_price_adj
        
        pnl = position_value * pnl_pct
        
        # Calculate fees (entry + exit)
        entry_fee = position_value * self.config.taker_fee
        exit_fee = position_value * self.config.taker_fee
        total_fees = entry_fee + exit_fee
        
        # Calculate slippage cost
        slippage_cost = position_value * self.config.slippage * 2  # Entry + exit
        
        # Net PnL
        net_pnl = pnl - total_fees
        
        # Update equity
        self.equity += net_pnl
        
        # Record trade
        trade = Trade(
            entry_time=entry_time,
            exit_time=exit_time,
            direction=direction,
            entry_price=entry_price_adj,
            exit_price=exit_price_adj,
            size=size,
            pnl=net_pnl,
            pnl_pct=net_pnl / (self.equity - net_pnl),  # % of equity before trade
            fees=total_fees,
            slippage_cost=slippage_cost,
            h1_signal_time=h1_signal_time,
            entry_method=entry_method,
            ou_zscore_at_entry=ou_zscore,
        )
        
        self.trades.append(trade)
        self.equity_curve.append({
            'timestamp': exit_time,
            'equity': self.equity,
            'trade_pnl': net_pnl,
        })
        
        return trade
    
    def run(
        self,
        h1_signals: pd.DataFrame,
        m5_data: pd.DataFrame,
    ) -> Dict:
        """
        Run backtest simulation.
        
        Parameters
        ----------
        h1_signals : pd.DataFrame
            H1 bars with columns: ['timestamp', 'signal', 'close']
            signal: 1 (long), -1 (short), 0 (neutral)
        m5_data : pd.DataFrame
            M5 OHLCV data with datetime index
        
        Returns
        -------
        dict
            Backtest results and statistics
        """
        print(f"[SNIPER] Starting backtest...")
        print(f"  Mode: {'Sniper (M5 Entry)' if self.config.use_sniper else 'Naive (H1 Entry)'}")
        print(f"  Capital: ${self.config.initial_capital:,.0f}")
        print(f"  Leverage: {self.config.leverage}x")
        print(f"  Position Size: {self.config.position_size_pct:.0%} of equity")
        
        # Ensure m5_data has datetime index
        if not isinstance(m5_data.index, pd.DatetimeIndex):
            if 'timestamp' in m5_data.columns:
                m5_data = m5_data.set_index('timestamp')
        
        # Iterate through H1 signals
        n_signals = 0
        n_trades = 0
        n_skipped = 0
        
        for idx, h1_row in h1_signals.iterrows():
            signal = h1_row['signal']
            
            # Skip neutral signals
            if signal == 0:
                continue
            
            n_signals += 1
            
            # Determine direction
            direction = 'long' if signal == 1 else 'short'
            
            # Get H1 bar time range
            h1_time = h1_row['timestamp'] if 'timestamp' in h1_row else idx
            h1_close_price = h1_row['close']
            
            # Define H1 bar boundaries (assuming 1-hour bars)
            h1_start = h1_time
            h1_end = h1_time + pd.Timedelta(hours=1)
            
            # Get M5 bars within this H1 period
            # Use boolean indexing for flexible time range matching
            m5_mask = (m5_data.index >= h1_start) & (m5_data.index < h1_end)
            m5_bars = m5_data.loc[m5_mask]
            
            if len(m5_bars) == 0:
                n_skipped += 1
                if n_skipped == 1:  # Print debug info for first skip
                    print(f"  [DEBUG] First skip: H1 time={h1_time}, M5 range={m5_data.index.min()} to {m5_data.index.max()}")
                continue
            
            # Entry logic
            if self.config.use_sniper:
                # Sniper mode: Find optimal M5 entry
                entry_result = self.find_sniper_entry(m5_bars, direction, h1_time)
                
                if entry_result is None:
                    # No sniper entry found
                    n_skipped += 1
                    continue
                
                entry_time, entry_price, ou_zscore = entry_result
                entry_method = 'sniper'
            else:
                # Naive mode: Enter at H1 open (first M5 bar)
                entry_time = m5_bars.index[0]
                entry_price = m5_bars.iloc[0]['close']
                ou_zscore = None
                entry_method = 'naive'
            
            # Exit at H1 close (last M5 bar or H1 close price)
            exit_time = h1_end
            exit_price = h1_close_price
            
            # Execute trade
            trade = self.execute_trade(
                entry_time=entry_time,
                exit_time=exit_time,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                h1_signal_time=h1_time,
                entry_method=entry_method,
                ou_zscore=ou_zscore,
            )
            
            n_trades += 1
        
        print(f"\n[SNIPER] Backtest complete:")
        print(f"  H1 Signals: {n_signals}")
        print(f"  Trades Executed: {n_trades}")
        print(f"  Trades Skipped: {n_skipped}")
        
        # Calculate statistics
        stats = self.calculate_statistics()
        
        return {
            'config': self.config,
            'trades': self.trades,
            'equity_curve': pd.DataFrame(self.equity_curve),
            'statistics': stats,
            'final_equity': self.equity,
        }
    
    def calculate_statistics(self) -> Dict:
        """Calculate backtest statistics."""
        if len(self.trades) == 0:
            return {
                'total_return_pct': 0.0,
                'total_return_dollar': 0.0,
                'n_trades': 0,
                'win_rate': 0.0,
                'avg_profit_per_trade': 0.0,
                'max_drawdown_pct': 0.0,
                'sharpe_ratio': 0.0,
            }
        
        trades_df = pd.DataFrame([
            {
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'direction': t.direction,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
            }
            for t in self.trades
        ])
        
        # Total return
        total_return_dollar = self.equity - self.config.initial_capital
        total_return_pct = total_return_dollar / self.config.initial_capital
        
        # Win rate
        winning_trades = (trades_df['pnl'] > 0).sum()
        win_rate = winning_trades / len(trades_df)
        
        # Average profit per trade
        avg_profit = trades_df['pnl'].mean()
        
        # Max drawdown
        equity_curve_df = pd.DataFrame(self.equity_curve)
        if len(equity_curve_df) > 0:
            cummax = equity_curve_df['equity'].cummax()
            drawdown = (equity_curve_df['equity'] - cummax) / cummax
            max_drawdown_pct = drawdown.min()
        else:
            max_drawdown_pct = 0.0
        
        # Sharpe ratio (annualized)
        if trades_df['pnl_pct'].std() > 0:
            # Assume ~8760 hours per year, ~8760 H1 bars
            # If we trade ~10% of bars, that's ~876 trades/year
            # Sharpe = Mean / Std * sqrt(N)
            sharpe_ratio = (trades_df['pnl_pct'].mean() / trades_df['pnl_pct'].std()) * np.sqrt(len(trades_df))
        else:
            sharpe_ratio = 0.0
        
        return {
            'total_return_pct': total_return_pct,
            'total_return_dollar': total_return_dollar,
            'n_trades': len(trades_df),
            'win_rate': win_rate,
            'avg_profit_per_trade': avg_profit,
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio,
            'avg_pnl_pct': trades_df['pnl_pct'].mean(),
            'std_pnl_pct': trades_df['pnl_pct'].std(),
        }


def calculate_entry_improvement(sniper_trades: List[Trade], naive_trades: List[Trade]) -> Dict:
    """
    Calculate improvement in entry price from sniper vs naive.
    
    Returns
    -------
    dict
        Statistics on entry price improvement in basis points
    """
    if len(sniper_trades) == 0 or len(naive_trades) == 0:
        return {
            'avg_improvement_bps': 0.0,
            'median_improvement_bps': 0.0,
            'pct_better_entries': 0.0,
        }
    
    # Match trades by H1 signal time
    sniper_dict = {t.h1_signal_time: t for t in sniper_trades}
    naive_dict = {t.h1_signal_time: t for t in naive_trades}
    
    improvements = []
    
    for h1_time in sniper_dict.keys():
        if h1_time not in naive_dict:
            continue
        
        sniper_t = sniper_dict[h1_time]
        naive_t = naive_dict[h1_time]
        
        # Calculate improvement (basis points)
        # For long: Lower entry is better
        # For short: Higher entry is better
        if sniper_t.direction == 'long':
            improvement_pct = (naive_t.entry_price - sniper_t.entry_price) / naive_t.entry_price
        else:  # short
            improvement_pct = (sniper_t.entry_price - naive_t.entry_price) / naive_t.entry_price
        
        improvement_bps = improvement_pct * 10000  # Convert to basis points
        improvements.append(improvement_bps)
    
    if len(improvements) == 0:
        return {
            'avg_improvement_bps': 0.0,
            'median_improvement_bps': 0.0,
            'pct_better_entries': 0.0,
        }
    
    improvements = np.array(improvements)
    
    return {
        'avg_improvement_bps': improvements.mean(),
        'median_improvement_bps': np.median(improvements),
        'pct_better_entries': (improvements > 0).sum() / len(improvements),
        'n_compared': len(improvements),
    }


__all__ = [
    'SniperBacktestEngine',
    'TradingConfig',
    'Trade',
    'calculate_entry_improvement',
]
