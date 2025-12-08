"""
Telemetry module for QFC System.

Provides comprehensive reporting and visualization tools.
"""

from .fleet_html_report import generate_individual_fleet_html

__all__ = [
    'generate_individual_fleet_html',
]
