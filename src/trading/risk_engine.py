"""
Futures Risk Engine for Dynamic Leverage & Stop-Loss Calculation.

This module implements intelligent risk management for futures trading:
- Dynamic Leverage: Inversely proportional to SDE uncertainty
- Dynamic Stops: Wider stops when uncertainty is high
- Liquidation Safety: Ensures stops are safer than liquidation price

Key Formula:
    Leverage = min(5.0, TargetVol / σ_SDE)
    
Where:
    - σ_SDE: SDE model uncertainty (diffusion coefficient)
    - TargetVol: Desired volatility exposure (default 0.02 = 2%)
    - Max leverage capped at 5x for safety

Author: QFC System v3.1 - Autonomous Trading Layer
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TradeParameters:
    """
    Container for calculated trade parameters.
    
    Attributes
    ----------
    symbol : str
        Asset symbol (e.g., 'BTCUSDT')
    side : str
        'LONG' or 'SHORT'
    leverage : float
        Calculated leverage (1.0 to 5.0)
    position_size_usd : float
        Position size in USD
    entry_price : float
        Entry price
    take_profit : float
        Take profit price
    stop_loss : float
        Stop loss price
    liquidation_price : float
        Estimated liquidation price
    risk_pct : float
        Risk as % of equity
    """
    symbol: str
    side: str
    leverage: float
    position_size_usd: float
    entry_price: float
    take_profit: float
    stop_loss: float
    liquidation_price: float
    risk_pct: float


class FuturesRiskEngine:
    """
    Futures Risk Engine with Dynamic Leverage & Stop-Loss.
    
    Calculates optimal trade parameters based on:
    1. SDE model uncertainty (σ_SDE)
    2. Account equity and risk limits
    3. Liquidation price safety
    
    Parameters
    ----------
    target_volatility : float, default=0.02
        Target volatility exposure (2% = moderate risk)
    max_leverage : float, default=5.0
        Maximum allowed leverage
    min_leverage : float, default=1.0
        Minimum leverage (spot equivalent)
    stop_loss_sigma_mult : float, default=2.0
        Stop-loss distance in multiples of σ_SDE
    take_profit_mult : float, default=2.0
        Take-profit as multiple of stop-loss distance
    max_risk_pct : float, default=0.02
        Maximum risk per trade as % of equity (2%)
    maintenance_margin_rate : float, default=0.004
        Maintenance margin rate (0.4% for most perpetuals)
    """
    
    def __init__(
        self,
        target_volatility: float = 0.02,
        max_leverage: float = 5.0,
        min_leverage: float = 1.0,
        stop_loss_sigma_mult: float = 2.0,
        take_profit_mult: float = 2.0,
        max_risk_pct: float = 0.02,
        maintenance_margin_rate: float = 0.004,
    ):
        self.target_volatility = target_volatility
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage
        self.stop_loss_sigma_mult = stop_loss_sigma_mult
        self.take_profit_mult = take_profit_mult
        self.max_risk_pct = max_risk_pct
        self.maintenance_margin_rate = maintenance_margin_rate
    
    def calculate_dynamic_leverage(self, sigma_sde: float) -> float:
        """
        Calculate dynamic leverage based on SDE uncertainty.
        
        Formula: Lev = min(max_lev, target_vol / σ_SDE)
        
        Intuition:
        - Low σ_SDE (confident model) → High leverage
        - High σ_SDE (uncertain model) → Low leverage
        
        Parameters
        ----------
        sigma_sde : float
            SDE model uncertainty (prediction diffusion)
        
        Returns
        -------
        leverage : float
            Calculated leverage (clamped to [min_lev, max_lev])
        """
        if sigma_sde <= 0:
            # Fallback: use minimum leverage if sigma is invalid
            return self.min_leverage
        
        # Dynamic leverage: inversely proportional to uncertainty
        leverage = self.target_volatility / sigma_sde
        
        # Clamp to safe range
        leverage = np.clip(leverage, self.min_leverage, self.max_leverage)
        
        return float(leverage)
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        sigma_sde: float,
        side: str,
    ) -> float:
        """
        Calculate dynamic stop-loss based on SDE uncertainty.
        
        Formula: SL = Entry ± (k * σ_SDE * Entry)
        
        Where:
        - k: stop_loss_sigma_mult (default 2.0)
        - ±: minus for LONG, plus for SHORT
        
        Parameters
        ----------
        entry_price : float
            Entry price
        sigma_sde : float
            SDE model uncertainty
        side : str
            'LONG' or 'SHORT'
        
        Returns
        -------
        stop_loss : float
            Stop-loss price
        """
        # Stop distance as % of entry price
        stop_distance_pct = self.stop_loss_sigma_mult * sigma_sde
        
        if side == 'LONG':
            # Long: stop below entry
            stop_loss = entry_price * (1.0 - stop_distance_pct)
        else:  # SHORT
            # Short: stop above entry
            stop_loss = entry_price * (1.0 + stop_distance_pct)
        
        return float(stop_loss)
    
    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        side: str,
    ) -> float:
        """
        Calculate take-profit as multiple of stop-loss distance.
        
        Formula: TP = Entry + (take_profit_mult * |Entry - SL|)
        
        Parameters
        ----------
        entry_price : float
            Entry price
        stop_loss : float
            Stop-loss price
        side : str
            'LONG' or 'SHORT'
        
        Returns
        -------
        take_profit : float
            Take-profit price
        """
        # Stop distance (absolute)
        stop_distance = abs(entry_price - stop_loss)
        
        # TP distance is multiple of stop distance
        tp_distance = self.take_profit_mult * stop_distance
        
        if side == 'LONG':
            # Long: TP above entry
            take_profit = entry_price + tp_distance
        else:  # SHORT
            # Short: TP below entry
            take_profit = entry_price - tp_distance
        
        return float(take_profit)
    
    def calculate_liquidation_price(
        self,
        entry_price: float,
        leverage: float,
        side: str,
    ) -> float:
        """
        Estimate liquidation price for a leveraged position.
        
        Formula (LONG): Liq = Entry * (1 - 1/Lev + MMR)
        Formula (SHORT): Liq = Entry * (1 + 1/Lev - MMR)
        
        Where MMR = Maintenance Margin Rate
        
        Parameters
        ----------
        entry_price : float
            Entry price
        leverage : float
            Position leverage
        side : str
            'LONG' or 'SHORT'
        
        Returns
        -------
        liquidation_price : float
            Estimated liquidation price
        """
        # Inverse leverage (margin fraction)
        margin_fraction = 1.0 / leverage
        
        if side == 'LONG':
            # Long liquidation: price drops by (1/Lev - MMR)
            liq_price = entry_price * (1.0 - margin_fraction + self.maintenance_margin_rate)
        else:  # SHORT
            # Short liquidation: price rises by (1/Lev - MMR)
            liq_price = entry_price * (1.0 + margin_fraction - self.maintenance_margin_rate)
        
        return float(liq_price)
    
    def is_stop_safe(
        self,
        stop_loss: float,
        liquidation_price: float,
        side: str,
    ) -> bool:
        """
        Check if stop-loss is safer than liquidation price.
        
        For LONG: SL should be > Liq (stop triggers before liquidation)
        For SHORT: SL should be < Liq (stop triggers before liquidation)
        
        Parameters
        ----------
        stop_loss : float
            Stop-loss price
        liquidation_price : float
            Liquidation price
        side : str
            'LONG' or 'SHORT'
        
        Returns
        -------
        is_safe : bool
            True if stop is safer than liquidation
        """
        if side == 'LONG':
            # Long: stop should be above liquidation
            return stop_loss > liquidation_price
        else:  # SHORT
            # Short: stop should be below liquidation
            return stop_loss < liquidation_price
    
    def calculate_position_size(
        self,
        equity: float,
        entry_price: float,
        stop_loss: float,
        leverage: float,
        allocation_pct: float,
    ) -> float:
        """
        Calculate position size based on risk and allocation.
        
        Two constraints:
        1. Risk constraint: Max loss = max_risk_pct * equity
        2. Allocation constraint: Position size = allocation_pct * equity * leverage
        
        Takes the minimum of both to ensure safety.
        
        Parameters
        ----------
        equity : float
            Total account equity (USD)
        entry_price : float
            Entry price
        stop_loss : float
            Stop-loss price
        leverage : float
            Position leverage
        allocation_pct : float
            Target allocation as % of equity (e.g., 0.15 = 15%)
        
        Returns
        -------
        position_size_usd : float
            Position size in USD
        """
        # 1. Risk-based sizing
        # Max loss = equity * max_risk_pct
        # Position size = max_loss / (|entry - stop| / entry)
        stop_distance_pct = abs(entry_price - stop_loss) / entry_price
        
        if stop_distance_pct > 0:
            risk_based_size = (equity * self.max_risk_pct) / stop_distance_pct
        else:
            risk_based_size = equity * allocation_pct * leverage
        
        # 2. Allocation-based sizing
        allocation_based_size = equity * allocation_pct * leverage
        
        # Take minimum (most conservative)
        position_size = min(risk_based_size, allocation_based_size)
        
        return float(position_size)
    
    def calculate_trade_parameters(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        sigma_sde: float,
        equity: float,
        allocation_pct: float,
    ) -> Optional[TradeParameters]:
        """
        Calculate complete trade parameters with safety checks.
        
        This is the main entry point for the risk engine.
        
        Parameters
        ----------
        symbol : str
            Asset symbol (e.g., 'BTCUSDT')
        side : str
            'LONG' or 'SHORT'
        entry_price : float
            Entry price
        sigma_sde : float
            SDE model uncertainty
        equity : float
            Total account equity (USD)
        allocation_pct : float
            Target allocation as % of equity
        
        Returns
        -------
        params : TradeParameters or None
            Complete trade parameters, or None if safety checks fail
        """
        # Step 1: Calculate dynamic leverage
        leverage = self.calculate_dynamic_leverage(sigma_sde)
        
        # Step 2: Calculate stop-loss
        stop_loss = self.calculate_stop_loss(entry_price, sigma_sde, side)
        
        # Step 3: Calculate take-profit
        take_profit = self.calculate_take_profit(entry_price, stop_loss, side)
        
        # Step 4: Calculate liquidation price
        liquidation_price = self.calculate_liquidation_price(entry_price, leverage, side)
        
        # Step 5: Safety check - is stop safer than liquidation?
        if not self.is_stop_safe(stop_loss, liquidation_price, side):
            # Stop would trigger AFTER liquidation - UNSAFE!
            # Reduce leverage and recalculate
            print(f"  [RISK] {symbol} {side}: Stop unsafe, reducing leverage...")
            
            # Reduce leverage by 20% and retry
            leverage = max(self.min_leverage, leverage * 0.8)
            liquidation_price = self.calculate_liquidation_price(entry_price, leverage, side)
            
            # Check again
            if not self.is_stop_safe(stop_loss, liquidation_price, side):
                print(f"  [RISK] {symbol} {side}: Still unsafe after reduction, REJECTING trade")
                return None
        
        # Step 6: Calculate position size
        position_size_usd = self.calculate_position_size(
            equity=equity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            leverage=leverage,
            allocation_pct=allocation_pct,
        )
        
        # Step 7: Calculate risk %
        stop_distance_pct = abs(entry_price - stop_loss) / entry_price
        risk_pct = stop_distance_pct * (position_size_usd / (equity * leverage))
        
        # Step 8: Create trade parameters
        params = TradeParameters(
            symbol=symbol,
            side=side,
            leverage=leverage,
            position_size_usd=position_size_usd,
            entry_price=entry_price,
            take_profit=take_profit,
            stop_loss=stop_loss,
            liquidation_price=liquidation_price,
            risk_pct=risk_pct,
        )
        
        return params
    
    def check_exit_conditions(
        self,
        position,
        current_price: float,
        current_sigma: float,
        entry_sigma: float,
    ) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Check if position should be exited based on dynamic conditions.
        
        Exit Rules:
        -----------
        1. **Standard SL**: Fixed % stop-loss (already in position.stop_loss)
        2. **Volatility Stop** (NEW): If current_sigma > 1.5 * entry_sigma → Exit 50%
        3. **Trailing Stop**: If profit > 1% → Move SL to breakeven
        
        Parameters
        ----------
        position : Position
            Current position object
        current_price : float
            Current market price
        current_sigma : float
            Current SDE uncertainty
        entry_sigma : float
            SDE uncertainty at entry
        
        Returns
        -------
        should_exit : bool
            Whether to exit the position
        exit_reason : str or None
            Reason for exit ('VOLATILITY_STOP', 'TRAILING_STOP', 'STANDARD_SL', None)
        exit_size_pct : float or None
            Percentage of position to exit (0.5 = 50%, 1.0 = 100%)
        """
        # Calculate current profit %
        if position.side == 'LONG':
            profit_pct = (current_price - position.entry_price) / position.entry_price
        else:  # SHORT
            profit_pct = (position.entry_price - current_price) / position.entry_price
        
        # Rule 1: Volatility Stop (Rising Uncertainty = Market Structure Changed)
        if current_sigma > 1.5 * entry_sigma:
            print(f"  [EXIT] {position.symbol}: Volatility Stop triggered "
                  f"(σ: {entry_sigma:.4f} → {current_sigma:.4f})")
            return True, 'VOLATILITY_STOP', 0.5  # Close 50%
        
        # Rule 2: Trailing Stop (Lock in profits)
        if profit_pct > 0.01:  # 1% profit
            # Check if price has retraced to breakeven
            if position.side == 'LONG':
                if current_price <= position.entry_price:
                    print(f"  [EXIT] {position.symbol}: Trailing Stop at breakeven "
                          f"(profit was {profit_pct*100:.2f}%)")
                    return True, 'TRAILING_STOP', 1.0  # Close 100%
            else:  # SHORT
                if current_price >= position.entry_price:
                    print(f"  [EXIT] {position.symbol}: Trailing Stop at breakeven "
                          f"(profit was {profit_pct*100:.2f}%)")
                    return True, 'TRAILING_STOP', 1.0  # Close 100%
        
        # Rule 3: Standard SL (handled by PaperExchange.update_prices)
        # No action needed here - exchange checks TP/SL automatically
        
        return False, None, None
    
    def get_diagnostics(self, params: TradeParameters) -> Dict[str, float]:
        """
        Get diagnostic information for a trade.
        
        Parameters
        ----------
        params : TradeParameters
            Trade parameters
        
        Returns
        -------
        diagnostics : dict
            Diagnostic metrics
        """
        # Calculate distances
        stop_distance_pct = abs(params.entry_price - params.stop_loss) / params.entry_price
        tp_distance_pct = abs(params.take_profit - params.entry_price) / params.entry_price
        liq_distance_pct = abs(params.liquidation_price - params.entry_price) / params.entry_price
        
        # Risk-reward ratio
        risk_reward = tp_distance_pct / stop_distance_pct if stop_distance_pct > 0 else 0
        
        return {
            'leverage': params.leverage,
            'position_size_usd': params.position_size_usd,
            'stop_distance_pct': stop_distance_pct * 100,  # %
            'tp_distance_pct': tp_distance_pct * 100,  # %
            'liq_distance_pct': liq_distance_pct * 100,  # %
            'risk_pct': params.risk_pct * 100,  # %
            'risk_reward_ratio': risk_reward,
        }
