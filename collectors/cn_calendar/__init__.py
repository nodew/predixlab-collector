"""CN Calendar collector module.

This module collects Chinese stock trading calendar dates from 2015-01-01 to current date
and saves them to the configured cn_calendar_path file.
"""

from .collector import collect_cn_calendar

__all__ = ["collect_cn_calendar"]
