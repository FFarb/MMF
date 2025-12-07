"""
Paper Exchange - Virtual Futures Trading Simulator.

This module implements a realistic futures exchange simulator:
- Account state tracking (balance, equity, margin)
- Position management (open/close with leverage)
- Mark-to-market PnL calculation
- TP/SL/Liquidation detection
- Trading fees (0.05%)
- State persistence to wallet_state.json

Key Features:
- Realistic margin calculations
- Automatic TP/SL execution
- Liquidation monitoring
- Fee deduction on every trade

Author: QFC System v3.1 - Autonomous Trading Layer
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class Position:
    """
    Container for an open futures position.
    
    Attributes
    ----------
    symbol : str
        Asset symbol (e.g., 'BTCUSDT')
    side : str
        'LONG' or 'SHORT'
    size_usd : float
        Position size in USD (notional value)
    leverage : float
        Position leverage
    entry_price : float
        Entry price
    entry_time : str
        Entry timestamp (ISO format)
    take_profit : float
        Take profit price
    stop_loss : float
        Stop loss price
    liquidation_price : float
        Liquidation price
    margin_used : float
        Margin locked for this position
    unrealized_pnl : float
        Current unrealized PnL
    """
    symbol: str
    side: str
    size_usd: float
    leverage: float
    entry_price: float
    entry_time: str
    take_profit: float
    stop_loss: float
    liquidation_price: float
    margin_used: float
    unrealized_pnl: float = 0.0
    entry_sigma: float = 0.01  # SDE uncertainty at entry (for volatility stop)


@dataclass
class Trade:
    """
    Container for a closed trade (for history tracking).
    
    Attributes
    ----------
    symbol : str
        Asset symbol
    side : str
        'LONG' or 'SHORT'
    entry_price : float
        Entry price
    exit_price : float
        Exit price
    size_usd : float
        Position size in USD
    leverage : float
        Position leverage
    entry_time : str
        Entry timestamp
    exit_time : str
        Exit timestamp
    pnl : float
        Realized PnL (after fees)
    exit_reason : str
        'TP', 'SL', 'LIQUIDATION', or 'MANUAL'
    """
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    size_usd: float
    leverage: float
    entry_time: str
    exit_time: str
    pnl: float
    exit_reason: str


class PaperExchange:
    """
    Virtual Futures Exchange Simulator.
    
    Simulates a futures exchange with realistic mechanics:
    - Margin-based trading
    - Automatic TP/SL/Liquidation
    - Fee deduction
    - State persistence
    
    Parameters
    ----------
    initial_balance : float, default=10000
        Starting balance in USDT
    trading_fee_pct : float, default=0.0005
        Trading fee (0.05% = 0.0005)
    state_file : Path or str, optional
        Path to wallet state file for persistence
    """
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        trading_fee_pct: float = 0.0005,
        state_file: Optional[Path | str] = None,
    ):
        self.initial_balance = initial_balance
        self.trading_fee_pct = trading_fee_pct
        self.state_file = Path(state_file) if state_file else None
        
        # Account state
        self.balance = initial_balance  # Available USDT
        self.positions: Dict[str, Position] = {}  # Open positions by symbol
        self.trade_history: List[Trade] = []  # Closed trades
        
        # Try to load existing state
        if self.state_file and self.state_file.exists():
            self.load_state()
            print(f"[Exchange] Loaded state from {self.state_file}")
        else:
            print(f"[Exchange] Initialized with ${initial_balance:,.2f} USDT")
    
    @property
    def margin_used(self) -> float:
        """Total margin locked in open positions."""
        return sum(pos.margin_used for pos in self.positions.values())
    
    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized PnL from open positions."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    @property
    def equity(self) -> float:
        """Total account equity (balance + unrealized PnL)."""
        return self.balance + self.unrealized_pnl
    
    @property
    def available_balance(self) -> float:
        """Available balance (balance - margin_used)."""
        return self.balance - self.margin_used
    
    def open_position(
        self,
        symbol: str,
        side: str,
        size_usd: float,
        leverage: float,
        entry_price: float,
        take_profit: float,
        stop_loss: float,
        liquidation_price: float,
        entry_sigma: float = 0.01,
        timestamp: Optional[str] = None,
    ) -> bool:
        """
        Open a new futures position.
        
        Parameters
        ----------
        symbol : str
            Asset symbol
        side : str
            'LONG' or 'SHORT'
        size_usd : float
            Position size in USD (notional value)
        leverage : float
            Position leverage
        entry_price : float
            Entry price
        take_profit : float
            Take profit price
        stop_loss : float
            Stop loss price
        liquidation_price : float
            Liquidation price
        timestamp : str, optional
            Entry timestamp (ISO format)
        
        Returns
        -------
        success : bool
            True if position opened successfully
        """
        # Check if position already exists
        if symbol in self.positions:
            print(f"  [Exchange] {symbol}: Position already open, skipping")
            return False
        
        # Calculate margin required
        margin_required = size_usd / leverage
        
        # Calculate entry fee
        entry_fee = size_usd * self.trading_fee_pct
        
        # Check if sufficient balance
        total_required = margin_required + entry_fee
        
        if total_required > self.available_balance:
            print(f"  [Exchange] {symbol}: Insufficient balance "
                  f"(need ${total_required:,.2f}, have ${self.available_balance:,.2f})")
            return False
        
        # Deduct margin and fee from balance
        self.balance -= total_required
        
        # Create position
        position = Position(
            symbol=symbol,
            side=side,
            size_usd=size_usd,
            leverage=leverage,
            entry_price=entry_price,
            entry_time=timestamp or datetime.now().isoformat(),
            take_profit=take_profit,
            stop_loss=stop_loss,
            liquidation_price=liquidation_price,
            margin_used=margin_required,
            unrealized_pnl=0.0,
            entry_sigma=entry_sigma,
        )
        
        self.positions[symbol] = position
        
        print(f"  [Exchange] {symbol}: Opened {side} position")
        print(f"    Size: ${size_usd:,.2f} @ {entry_price:.2f} ({leverage:.1f}x)")
        print(f"    Margin: ${margin_required:,.2f} | Fee: ${entry_fee:,.2f}")
        print(f"    TP: {take_profit:.2f} | SL: {stop_loss:.2f} | Liq: {liquidation_price:.2f}")
        
        return True
    
    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str = 'MANUAL',
        timestamp: Optional[str] = None,
    ) -> Optional[Trade]:
        """
        Close an open position.
        
        Parameters
        ----------
        symbol : str
            Asset symbol
        exit_price : float
            Exit price
        exit_reason : str, default='MANUAL'
            Reason for exit ('TP', 'SL', 'LIQUIDATION', 'MANUAL')
        timestamp : str, optional
            Exit timestamp (ISO format)
        
        Returns
        -------
        trade : Trade or None
            Closed trade record, or None if position not found
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Calculate PnL
        if position.side == 'LONG':
            # Long: profit when price rises
            price_change_pct = (exit_price - position.entry_price) / position.entry_price
        else:  # SHORT
            # Short: profit when price falls
            price_change_pct = (position.entry_price - exit_price) / position.entry_price
        
        # PnL = size * price_change% * leverage
        gross_pnl = position.size_usd * price_change_pct * position.leverage
        
        # Deduct exit fee
        exit_fee = position.size_usd * self.trading_fee_pct
        net_pnl = gross_pnl - exit_fee
        
        # Return margin to balance
        self.balance += position.margin_used
        
        # Add PnL to balance
        self.balance += net_pnl
        
        # Create trade record
        trade = Trade(
            symbol=symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            size_usd=position.size_usd,
            leverage=position.leverage,
            entry_time=position.entry_time,
            exit_time=timestamp or datetime.now().isoformat(),
            pnl=net_pnl,
            exit_reason=exit_reason,
        )
        
        self.trade_history.append(trade)
        
        # Remove position
        del self.positions[symbol]
        
        pnl_sign = '+' if net_pnl >= 0 else ''
        print(f"  [Exchange] {symbol}: Closed {position.side} position ({exit_reason})")
        print(f"    Entry: {position.entry_price:.2f} → Exit: {exit_price:.2f}")
        print(f"    PnL: {pnl_sign}${net_pnl:,.2f} (after fees)")
        
        return trade
    
    def update_prices(self, prices: Dict[str, float], timestamp: Optional[str] = None) -> List[Trade]:
        """
        Update current prices and check for TP/SL/Liquidation.
        
        This is the main method called every candle to:
        1. Update unrealized PnL
        2. Check TP/SL hits
        3. Check liquidations
        
        Parameters
        ----------
        prices : dict
            Current prices by symbol {symbol: price}
        timestamp : str, optional
            Current timestamp (ISO format)
        
        Returns
        -------
        closed_trades : list of Trade
            Trades that were closed due to TP/SL/Liquidation
        """
        closed_trades = []
        symbols_to_close = []
        
        for symbol, position in self.positions.items():
            if symbol not in prices:
                continue
            
            current_price = prices[symbol]
            
            # Update unrealized PnL
            if position.side == 'LONG':
                price_change_pct = (current_price - position.entry_price) / position.entry_price
            else:  # SHORT
                price_change_pct = (position.entry_price - current_price) / position.entry_price
            
            position.unrealized_pnl = position.size_usd * price_change_pct * position.leverage
            
            # Check liquidation first (highest priority)
            if position.side == 'LONG':
                if current_price <= position.liquidation_price:
                    symbols_to_close.append((symbol, position.liquidation_price, 'LIQUIDATION'))
                    continue
            else:  # SHORT
                if current_price >= position.liquidation_price:
                    symbols_to_close.append((symbol, position.liquidation_price, 'LIQUIDATION'))
                    continue
            
            # Check stop-loss
            if position.side == 'LONG':
                if current_price <= position.stop_loss:
                    symbols_to_close.append((symbol, position.stop_loss, 'SL'))
                    continue
            else:  # SHORT
                if current_price >= position.stop_loss:
                    symbols_to_close.append((symbol, position.stop_loss, 'SL'))
                    continue
            
            # Check take-profit
            if position.side == 'LONG':
                if current_price >= position.take_profit:
                    symbols_to_close.append((symbol, position.take_profit, 'TP'))
                    continue
            else:  # SHORT
                if current_price <= position.take_profit:
                    symbols_to_close.append((symbol, position.take_profit, 'TP'))
                    continue
        
        # Close positions that hit TP/SL/Liquidation
        for symbol, exit_price, exit_reason in symbols_to_close:
            trade = self.close_position(symbol, exit_price, exit_reason, timestamp)
            if trade:
                closed_trades.append(trade)
        
        return closed_trades
    
    def get_account_summary(self) -> Dict[str, float]:
        """
        Get current account summary.
        
        Returns
        -------
        summary : dict
            Account metrics
        """
        total_pnl = sum(t.pnl for t in self.trade_history)
        win_trades = [t for t in self.trade_history if t.pnl > 0]
        loss_trades = [t for t in self.trade_history if t.pnl <= 0]
        
        win_rate = len(win_trades) / len(self.trade_history) if self.trade_history else 0
        
        return {
            'balance': self.balance,
            'equity': self.equity,
            'margin_used': self.margin_used,
            'available_balance': self.available_balance,
            'unrealized_pnl': self.unrealized_pnl,
            'num_open_positions': len(self.positions),
            'total_trades': len(self.trade_history),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'num_wins': len(win_trades),
            'num_losses': len(loss_trades),
        }
    
    def save_state(self):
        """Save wallet state to JSON file."""
        if not self.state_file:
            return
        
        state = {
            'balance': self.balance,
            'positions': {
                symbol: asdict(pos) for symbol, pos in self.positions.items()
            },
            'trade_history': [asdict(t) for t in self.trade_history],
            'last_updated': datetime.now().isoformat(),
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"[Exchange] State saved to {self.state_file}")
    
    def load_state(self):
        """Load wallet state from JSON file."""
        if not self.state_file or not self.state_file.exists():
            return
        
        with open(self.state_file, 'r') as f:
            state = json.load(f)
        
        self.balance = state['balance']
        
        # Restore positions
        self.positions = {
            symbol: Position(**pos_data)
            for symbol, pos_data in state.get('positions', {}).items()
        }
        
        # Restore trade history
        self.trade_history = [
            Trade(**trade_data)
            for trade_data in state.get('trade_history', [])
        ]
        
        print(f"[Exchange] Loaded {len(self.positions)} positions, {len(self.trade_history)} trades")
