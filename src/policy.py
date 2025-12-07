"""
Trade Policy and Risk Management.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class TradeParams:
    tp_mult: float
    sl_mult: float
    leverage: float
    regime: str

class TradePolicy:
    """
    Determines trading parameters based on market regime and risk settings.
    """

    def __init__(self):
        # Configuration map for each regime
        self.policy_map = {
            "Trend": {"tp_mult": 4.0, "sl_mult": 1.5, "risk_factor": 1.0},
            "Range": {"tp_mult": 2.0, "sl_mult": 2.0, "risk_factor": 0.8},
            "Stress": {"tp_mult": 1.0, "sl_mult": 1.0, "risk_factor": 0.0}, # No trading or minimal
        }
        
        self.target_risk_equity = 0.01  # Risk 1% of equity per trade
        self.max_leverage = 10.0

    def get_parameters(self, regime: str, atr: float, price: float) -> TradeParams:
        """
        Get dynamic TP/SL and Leverage based on regime and volatility.
        """
        params = self.policy_map.get(regime, self.policy_map["Range"])
        
        # If Stress, we might want to skip, but for now let's return safe params
        if regime == "Stress":
             return TradeParams(
                tp_mult=params["tp_mult"],
                sl_mult=params["sl_mult"],
                leverage=1.0, # Minimal leverage in stress
                regime=regime
            )

        tp_dist = atr * params["tp_mult"]
        sl_dist = atr * params["sl_mult"]
        
        # Calculate Dynamic Leverage
        sl_pct = sl_dist / price
        if sl_pct == 0:
            leverage = 1.0
        else:
            # Risk Parity: Leverage = Target_Risk / Stop_Loss_Pct
            # e.g. 1% Risk / 1% SL = 1x Lev
            #      1% Risk / 0.5% SL = 2x Lev
            leverage = (self.target_risk_equity * params["risk_factor"]) / sl_pct
        
        # Cap leverage
        leverage = min(leverage, self.max_leverage)
        leverage = max(leverage, 1.0) # Min 1x
        
        return TradeParams(
            tp_mult=params["tp_mult"],
            sl_mult=params["sl_mult"],
            leverage=round(leverage, 2),
            regime=regime
        )
