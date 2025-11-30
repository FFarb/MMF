"""
Risk Engine - Physics-Based Position Sizing

This module implements risk management based on Early Warning Signals (EWS)
from the Stability Monitor. Position sizing is scaled based on market stability
to avoid catastrophic losses during critical transitions.
"""

from __future__ import annotations

from typing import Literal


class StabilityRiskManager:
    """
    Physics-based risk manager that scales position sizing based on EWS.
    
    Uses Critical Slowing Down indicators (theta, ACF, warnings) to
    dynamically adjust leverage and position size, reducing exposure
    before market crashes.
    """
    
    def __init__(self, mode: Literal['strict', 'moderate', 'continuous'] = 'strict'):
        """
        Initialize the risk manager.
        
        Parameters
        ----------
        mode : {'strict', 'moderate', 'continuous'}, default='strict'
            Risk management mode:
            - 'strict': Warning → 0.0 (go to cash)
            - 'moderate': Warning → 0.5 (half size)
            - 'continuous': Smooth scaling based on theta
        """
        self.mode = mode
    
    def get_leverage_multiplier(
        self,
        theta: float,
        acf: float = 0.0,
        warning: bool = False,
    ) -> float:
        """
        Calculate position size multiplier based on market stability.
        
        Parameters
        ----------
        theta : float
            Mean reversion speed from OU calibration
            - High theta (>0.05): Strong mean reversion, safe
            - Low theta (<0.01): Weak mean reversion, dangerous
            - Zero theta: No mean reversion, critical
        acf : float, default=0.0
            Lag-1 autocorrelation
            - High ACF (>0.95): Sluggish recovery, warning sign
        warning : bool, default=False
            Combined warning flag from StabilityMonitor
            
        Returns
        -------
        float
            Position size multiplier in [0.0, 1.0]
            - 1.0 = Full size (normal conditions)
            - 0.5 = Half size (moderate risk)
            - 0.0 = No position (critical risk)
            
        Notes
        -----
        The multiplier determines actual position size:
            actual_size = base_size * multiplier
        
        In 'strict' mode, any warning immediately forces cash (0.0).
        In 'moderate' mode, warnings reduce to half size (0.5).
        In 'continuous' mode, scaling is smooth based on theta.
        """
        # Mode-specific logic
        if self.mode == 'strict':
            # Strict: Warning → Cash
            if warning:
                return 0.0
            else:
                return 1.0
        
        elif self.mode == 'moderate':
            # Moderate: Warning → Half size
            if warning:
                return 0.5
            else:
                return 1.0
        
        elif self.mode == 'continuous':
            # Continuous: Smooth scaling based on theta
            # If warning, force to 0 or 0.5 depending on severity
            if warning:
                # Check severity: if theta is very low, go to cash
                if theta < 0.005:
                    return 0.0
                else:
                    return 0.5
            
            # Otherwise, scale smoothly based on theta
            # Threshold: theta < 0.01 is dangerous
            theta_threshold = 0.01
            
            if theta >= theta_threshold:
                # Safe: full size
                return 1.0
            elif theta > 0:
                # Weak mean reversion: scale down linearly
                multiplier = theta / theta_threshold
                return max(0.0, min(1.0, multiplier))
            else:
                # No mean reversion: cash
                return 0.0
        
        else:
            # Default: full size
            return 1.0
    
    def calculate_kelly_fraction(
        self,
        win_rate: float = 0.5,
        avg_win: float = 1.0,
        avg_loss: float = 1.0,
    ) -> float:
        """
        Calculate Kelly fraction for optimal position sizing.
        
        Parameters
        ----------
        win_rate : float, default=0.5
            Probability of winning trade
        avg_win : float, default=1.0
            Average win size (in units of risk)
        avg_loss : float, default=1.0
            Average loss size (in units of risk)
            
        Returns
        -------
        float
            Kelly fraction (currently returns 1.0 as placeholder)
            
        Notes
        -----
        Kelly Criterion formula:
            f* = (p*b - q) / b
        where:
            p = win_rate
            q = 1 - p
            b = avg_win / avg_loss
        
        This is a placeholder for future implementation.
        """
        # Placeholder: return full Kelly (1.0)
        # Future: implement actual Kelly calculation
        return 1.0
