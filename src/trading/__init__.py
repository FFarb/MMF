"""Trading module for risk management and strategy execution."""

from .risk_engine import FuturesRiskEngine
from .portfolio import PortfolioManager
from .paper_exchange import PaperExchange
from .sniper_engine import (
    SniperBacktestEngine,
    TradingConfig,
    Trade,
    calculate_entry_improvement,
)

__all__ = [
    'FuturesRiskEngine',
    'PortfolioManager',
    'PaperExchange',
    'SniperBacktestEngine',
    'TradingConfig',
    'Trade',
    'calculate_entry_improvement',
]
