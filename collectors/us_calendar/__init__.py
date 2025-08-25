"""US Calendar collector module.

This module collects US stock trading calendar dates from 2015-01-01 to current date
and saves them to the configured us_calendar_path file.
"""

from .collector import collect_us_calendar

__all__ = ["collect_us_calendar"]
