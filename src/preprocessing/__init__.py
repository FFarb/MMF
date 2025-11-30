"""
Preprocessing utilities for the Quanta Futures framework.

This module provides advanced preprocessing techniques including:
- Fractional Differentiation for memory-preserving stationarity
"""

from .frac_diff import FractionalDifferentiator

__all__ = ["FractionalDifferentiator"]
