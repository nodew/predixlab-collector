"""Collectors package for cn_index.

This package contains data collectors for Chinese stock index constituents (CSI 300 and CSI 500).
"""

from .collector import CNIndexCollector, collect_cn_index

__all__ = [
    "CNIndexCollector",
    "collect_cn_index",
]
