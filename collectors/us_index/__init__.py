"""Collectors package for qstock-collector.

This package contains data collectors for various stock market indices and data sources.
"""

from .collector import USIndexCollector, collect_us_index

__all__ = [
    "USIndexCollector",
    "collect_us_index",
]
