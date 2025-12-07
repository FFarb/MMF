"""
Portfolio Manager for Expectancy-Based Capital Allocation.

This module implements intelligent capital allocation across the SDE-MoE fleet:
- Reads fleet performance from fleet_summary.csv
- Allocates capital proportional to expectancy
- Enforces diversification limits (max 20% per asset)
- Integrates signals from predictions_*.csv files

Key Principle:
    Allocation ∝ Expectancy (with diversification caps)
    
Example:
    BNB (Expectancy: 0.0068) → 20% allocation (capped)
    ETH (Expectancy: 0.0027) → 8% allocation

Author: QFC System v3.1 - Autonomous Trading Layer
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class AssetSignal:
    """
    Container for asset trading signal.
    
    Attributes
    ----------
    symbol : str
        Asset symbol (e.g., 'BTCUSDT')
    probability : float
        Model probability (0 to 1)
    signal : int
        Signal direction (1 = LONG, -1 = SHORT/NO TRADE)
    close_price : float
        Current close price
    expectancy : float
        Asset expectancy from fleet summary
    """
    symbol: str
    probability: float
    signal: int
    close_price: float
    expectancy: float


@dataclass
class PortfolioAllocation:
    """
    Container for portfolio allocation decision.
    
    Attributes
    ----------
    symbol : str
        Asset symbol
    side : str
        'LONG' or 'SHORT'
    allocation_pct : float
        Target allocation as % of equity (0 to 0.20)
    allocation_usd : float
        Target allocation in USD
    probability : float
        Model probability
    expectancy : float
        Asset expectancy
    """
    symbol: str
    side: str
    allocation_pct: float
    allocation_usd: float
    probability: float
    expectancy: float


class PortfolioManager:
    """
    Portfolio Manager for Expectancy-Based Capital Allocation.
    
    Manages capital allocation across the SDE-MoE fleet based on:
    1. Asset expectancy (from fleet_summary.csv)
    2. Model signals (from predictions_*.csv)
    3. Diversification limits
    
    Parameters
    ----------
    fleet_summary_path : Path or str
        Path to fleet_summary.csv
    predictions_dir : Path or str
        Directory containing predictions_*.csv files
    max_allocation_pct : float, default=0.20
        Maximum allocation per asset (20%)
    min_probability : float, default=0.55
        Minimum probability to consider a signal
    max_positions : int, default=5
        Maximum number of concurrent positions
    """
    
    def __init__(
        self,
        fleet_summary_path: Path | str,
        predictions_dir: Path | str,
        max_allocation_pct: float = 0.20,
        min_probability: float = 0.55,
        max_positions: int = 5,
    ):
        self.fleet_summary_path = Path(fleet_summary_path)
        self.predictions_dir = Path(predictions_dir)
        self.max_allocation_pct = max_allocation_pct
        self.min_probability = min_probability
        self.max_positions = max_positions
        
        # Load fleet summary
        self.fleet_summary = self._load_fleet_summary()
        
        # Cache for predictions
        self._predictions_cache = {}
    
    def _load_fleet_summary(self) -> pd.DataFrame:
        """
        Load fleet summary with expectancy data.
        
        Returns
        -------
        summary : pd.DataFrame
            Fleet summary with columns: asset, avg_expectancy, etc.
        """
        if not self.fleet_summary_path.exists():
            raise FileNotFoundError(f"Fleet summary not found: {self.fleet_summary_path}")
        
        summary = pd.read_csv(self.fleet_summary_path)
        
        # Validate required columns
        required_cols = ['asset', 'avg_expectancy']
        missing = [col for col in required_cols if col not in summary.columns]
        if missing:
            raise ValueError(f"Fleet summary missing columns: {missing}")
        
        print(f"[Portfolio] Loaded fleet summary: {len(summary)} assets")
        print(f"  Expectancy range: {summary['avg_expectancy'].min():.6f} to {summary['avg_expectancy'].max():.6f}")
        
        return summary
    
    def _load_latest_predictions(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load latest predictions for an asset.
        
        Parameters
        ----------
        symbol : str
            Asset symbol (e.g., 'BTCUSDT')
        
        Returns
        -------
        predictions : pd.DataFrame or None
            Latest predictions, or None if not found
        """
        pred_file = self.predictions_dir / f"predictions_{symbol}.csv"
        
        if not pred_file.exists():
            print(f"  [Portfolio] WARNING: No predictions found for {symbol}")
            return None
        
        # Load predictions
        df = pd.read_csv(pred_file, parse_dates=['timestamp'])
        
        # Cache for future use
        self._predictions_cache[symbol] = df
        
        return df
    
    def get_latest_signal(self, symbol: str) -> Optional[AssetSignal]:
        """
        Get latest trading signal for an asset.
        
        Parameters
        ----------
        symbol : str
            Asset symbol
        
        Returns
        -------
        signal : AssetSignal or None
            Latest signal, or None if not available
        """
        # Load predictions
        df = self._load_latest_predictions(symbol)
        if df is None or len(df) == 0:
            return None
        
        # Get latest row
        latest = df.iloc[-1]
        
        # Get expectancy from fleet summary
        asset_row = self.fleet_summary[self.fleet_summary['asset'] == symbol]
        if len(asset_row) == 0:
            print(f"  [Portfolio] WARNING: {symbol} not in fleet summary")
            return None
        
        expectancy = asset_row['avg_expectancy'].values[0]
        
        # Create signal
        signal = AssetSignal(
            symbol=symbol,
            probability=latest['probability'],
            signal=latest['signal'],
            close_price=latest['close'],
            expectancy=expectancy,
        )
        
        return signal
    
    def calculate_expectancy_weights(self) -> Dict[str, float]:
        """
        Calculate allocation weights based on expectancy.
        
        Formula:
            weight_i = expectancy_i / Σ(expectancy_j) for all j with expectancy > 0
        
        Returns
        -------
        weights : dict
            Mapping from symbol to weight (0 to 1)
        """
        # Filter positive expectancy only
        positive = self.fleet_summary[self.fleet_summary['avg_expectancy'] > 0].copy()
        
        if len(positive) == 0:
            print("[Portfolio] WARNING: No assets with positive expectancy!")
            return {}
        
        # Calculate weights (proportional to expectancy)
        total_expectancy = positive['avg_expectancy'].sum()
        positive['weight'] = positive['avg_expectancy'] / total_expectancy
        
        # Convert to dict
        weights = dict(zip(positive['asset'], positive['weight']))
        
        return weights
    
    def calculate_allocations(
        self,
        equity: float,
        current_timestamp: Optional[pd.Timestamp] = None,
    ) -> List[PortfolioAllocation]:
        """
        Calculate target allocations for all assets.
        
        This is the main entry point for portfolio management.
        
        Parameters
        ----------
        equity : float
            Total account equity (USD)
        current_timestamp : pd.Timestamp, optional
            Current timestamp (for filtering signals)
        
        Returns
        -------
        allocations : list of PortfolioAllocation
            Target allocations for each asset
        """
        # Step 1: Get expectancy-based weights
        weights = self.calculate_expectancy_weights()
        
        if not weights:
            print("[Portfolio] No valid weights, returning empty allocations")
            return []
        
        # Step 2: Get latest signals for all assets
        signals = []
        for symbol in weights.keys():
            signal = self.get_latest_signal(symbol)
            if signal is not None:
                signals.append(signal)
        
        if not signals:
            print("[Portfolio] No valid signals, returning empty allocations")
            return []
        
        # Step 3: Filter signals by probability threshold
        valid_signals = [
            s for s in signals
            if s.signal == 1 and s.probability >= self.min_probability
        ]
        
        if not valid_signals:
            print(f"[Portfolio] No signals above threshold ({self.min_probability:.2f})")
            return []
        
        print(f"[Portfolio] {len(valid_signals)} valid signals (prob >= {self.min_probability:.2f})")
        
        # Step 4: Sort by expectancy (descending)
        valid_signals.sort(key=lambda s: s.expectancy, reverse=True)
        
        # Step 5: Limit to max positions
        if len(valid_signals) > self.max_positions:
            print(f"[Portfolio] Limiting to top {self.max_positions} positions")
            valid_signals = valid_signals[:self.max_positions]
        
        # Step 6: Calculate allocations
        allocations = []
        
        for signal in valid_signals:
            # Base allocation from expectancy weight
            base_allocation_pct = weights.get(signal.symbol, 0)
            
            # Cap at max allocation
            allocation_pct = min(base_allocation_pct, self.max_allocation_pct)
            
            # Calculate USD amount
            allocation_usd = equity * allocation_pct
            
            # Determine side (always LONG for now, signals are binary)
            side = 'LONG' if signal.signal == 1 else 'SHORT'
            
            allocation = PortfolioAllocation(
                symbol=signal.symbol,
                side=side,
                allocation_pct=allocation_pct,
                allocation_usd=allocation_usd,
                probability=signal.probability,
                expectancy=signal.expectancy,
            )
            
            allocations.append(allocation)
        
        # Step 7: Normalize if total allocation > 100%
        total_allocation_pct = sum(a.allocation_pct for a in allocations)
        
        if total_allocation_pct > 1.0:
            print(f"[Portfolio] Total allocation {total_allocation_pct:.2%} > 100%, normalizing...")
            
            # Scale down proportionally
            scale_factor = 1.0 / total_allocation_pct
            
            for allocation in allocations:
                allocation.allocation_pct *= scale_factor
                allocation.allocation_usd = equity * allocation.allocation_pct
        
        # Print summary
        print(f"\n[Portfolio] Allocation Summary:")
        print(f"  Total Equity: ${equity:,.2f}")
        print(f"  Active Positions: {len(allocations)}")
        
        for alloc in allocations:
            print(f"    {alloc.symbol:10s} {alloc.side:5s} | "
                  f"Alloc: {alloc.allocation_pct:5.1%} (${alloc.allocation_usd:,.0f}) | "
                  f"Prob: {alloc.probability:.3f} | Exp: {alloc.expectancy:.6f}")
        
        return allocations
    
    def get_portfolio_stats(self, allocations: List[PortfolioAllocation]) -> Dict[str, float]:
        """
        Calculate portfolio-level statistics.
        
        Parameters
        ----------
        allocations : list of PortfolioAllocation
            Current allocations
        
        Returns
        -------
        stats : dict
            Portfolio statistics
        """
        if not allocations:
            return {
                'num_positions': 0,
                'total_allocation_pct': 0.0,
                'avg_probability': 0.0,
                'avg_expectancy': 0.0,
                'max_allocation_pct': 0.0,
            }
        
        total_alloc = sum(a.allocation_pct for a in allocations)
        avg_prob = np.mean([a.probability for a in allocations])
        avg_exp = np.mean([a.expectancy for a in allocations])
        max_alloc = max(a.allocation_pct for a in allocations)
        
        return {
            'num_positions': len(allocations),
            'total_allocation_pct': total_alloc,
            'avg_probability': avg_prob,
            'avg_expectancy': avg_exp,
            'max_allocation_pct': max_alloc,
        }
    
    def reload_fleet_summary(self):
        """Reload fleet summary (useful after retraining)."""
        print("[Portfolio] Reloading fleet summary...")
        self.fleet_summary = self._load_fleet_summary()
        self._predictions_cache.clear()
