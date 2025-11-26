"""CN Calendar collector module.

This module collects Chinese stock trading calendar dates from 2015-01-01 to current date
using Yahoo Finance data through yahooquery, and saves them to the configured file.
"""

import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path
from loguru import logger
from typing import List
import time
from yahooquery import Ticker

from config import settings
from utils import normalize_datetime_to_date


class CNCalendarCollector:
    """Collector for Chinese stock trading calendar dates."""

    # Constants
    DEFAULT_START_DATE = "2015-01-01"
    DEFAULT_WEEKLY_START_DATE = "2007-12-31"

    def __init__(self, start_date: str = None, interval: str = "1d"):
        """Initialize the CN calendar collector.

        Args:
            start_date: Start date for collecting calendar data. If None, uses 2015-01-01 for daily 
                       data and 2007-12-31 for weekly data
            interval: Data interval, default is "1d". Supported values: "1d", "1wk", "1mo"
        """
        # Set default start date based on interval
        if start_date:
            self.start_date = start_date
        elif interval == "1wk":
            self.start_date = self.DEFAULT_WEEKLY_START_DATE
        else:
            self.start_date = self.DEFAULT_START_DATE
        
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        self.interval = interval
        
        # Select calendar path based on interval
        if interval == "1wk":
            self.cn_calendar_path = Path(settings.cn_weekly_calendar_path).expanduser()
        else:
            # For "1d" and other intervals, use the default path
            self.cn_calendar_path = Path(settings.cn_calendar_path).expanduser()
        
        self.cn_calendar_path.parent.mkdir(parents=True, exist_ok=True)
        # Use Shanghai Composite Index (000001.SS) as reference for Chinese trading calendar
        self.reference_symbol = "000001.SS"

    def get_cn_trading_dates(self) -> List[date]:
        """Get Chinese stock trading dates from start_date to current date using Yahoo Finance.

        Uses retry mechanism with exponential backoff (max 3 retries) for yahooquery calls.

        Returns:
            List of trading dates as Python date objects
        """
        logger.info(f"Fetching CN trading calendar (interval={self.interval}) from {self.start_date} to {self.end_date}...")

        max_retries = 3
        base_delay = 1.0  # Base delay in seconds

        for attempt in range(max_retries + 1):
            try:
                # Use Yahoo Finance to get historical data for Shanghai Composite Index
                ticker = Ticker(self.reference_symbol)

                # Get historical data with maximum period to cover from start_date
                hist_data = ticker.history(
                    interval=self.interval,
                    start=self.start_date,
                    end=self.end_date
                )

                if hist_data is None or hist_data.empty:
                    raise ValueError(f"No data received for symbol {self.reference_symbol}")

                # Extract trading dates from the index
                if isinstance(hist_data.index, pd.MultiIndex):
                    # If MultiIndex (symbol, date), get level 1 (date)
                    trading_dates = hist_data.index.get_level_values(level="date").unique()
                else:
                    # If single index (date)
                    trading_dates = hist_data.index

                trading_dates = [normalize_datetime_to_date(d) for d in trading_dates]

                if self.interval == "1wk":
                    today = date.today()
                    days = today.weekday()
                    if days >= 5:  # Saturday or Sunday
                        reference_monday = today - timedelta(days=days)  # This week's Monday
                    else:  # Monday to Friday
                        reference_monday = today - timedelta(days=7 + days)  # Last week's Monday

                    trading_dates = [d for d in trading_dates if d <= reference_monday]

                # Sort the dates
                trading_dates = sorted(trading_dates)

                # Filter to ensure dates are within our specified range
                start_ts = datetime.strptime(self.start_date, "%Y-%m-%d").date()
                end_ts = datetime.strptime(self.end_date, "%Y-%m-%d").date()
                trading_dates = [d for d in trading_dates if start_ts <= d <= end_ts]

                logger.info(f"Successfully fetched {len(trading_dates)} CN trading dates")
                if trading_dates:
                    logger.info(f"Date range: {trading_dates[0]} to {trading_dates[-1]}")

                return trading_dates

            except Exception as e:
                if attempt < max_retries:
                    # Calculate delay with exponential backoff
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed to fetch CN trading calendar: {e}")
                    logger.info(f"Retrying in {delay:.1f} seconds... ({attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    logger.error(f"Failed to fetch CN trading calendar after {max_retries + 1} attempts: {e}")
                    raise

    def save_calendar(self, trading_dates: List[date]) -> None:
        """Save trading dates to the configured calendar file using merge strategy.

        If the file already exists, merge the new dates with existing dates
        and save the union of both sets.

        Args:
            trading_dates: List of trading dates to save
        """
        try:
            # Convert dates to string format (YYYY-MM-DD)
            new_date_strings = set(d.strftime("%Y-%m-%d") for d in trading_dates)

            # Check if file already exists and read existing dates
            existing_date_strings = set()
            if self.cn_calendar_path.exists():
                logger.info(f"Existing calendar file found: {self.cn_calendar_path}")
                with open(self.cn_calendar_path, 'r', encoding='utf-8') as f:
                    existing_date_strings = set(line.strip() for line in f if line.strip())
                logger.info(f"Loaded {len(existing_date_strings)} existing trading dates")

            # Merge new dates with existing dates (union)
            merged_date_strings = new_date_strings.union(existing_date_strings)

            # Sort the merged dates
            sorted_dates = sorted(merged_date_strings)

            # Save merged dates to file
            with open(self.cn_calendar_path, 'w', encoding='utf-8') as f:
                for date_str in sorted_dates:
                    f.write(f"{date_str}\n")

            logger.info(f"Merged and saved {len(sorted_dates)} CN trading dates to: {self.cn_calendar_path}")
            logger.info(f"  - Existing dates: {len(existing_date_strings)}")
            logger.info(f"  - New dates: {len(new_date_strings)}")
            logger.info(f"  - Total unique dates: {len(sorted_dates)}")

        except Exception as e:
            logger.error(f"Failed to save CN calendar to {self.cn_calendar_path}: {e}")
            raise

    def collect(self) -> None:
        """Collect CN trading calendar and save to configured file."""
        logger.info("Starting CN trading calendar collection...")

        try:
            # Get trading dates
            trading_dates = self.get_cn_trading_dates()

            if not trading_dates:
                raise ValueError("No trading dates were collected")

            # Save to file
            self.save_calendar(trading_dates)

            logger.info("CN trading calendar collection completed successfully!")

        except Exception as e:
            logger.error(f"CN trading calendar collection failed: {e}")
            raise


def collect_cn_calendar(start_date: str = None, interval: str = "1d"):
    """Main entry point for CN trading calendar collection.

    Args:
        start_date: Start date for collecting calendar data. If None, uses 2015-01-01 for daily 
                   data and 2007-12-31 for weekly data
        interval: Data interval, default is "1d". Supported values: "1d", "1wk", "1mo"
    """
    collector = CNCalendarCollector(start_date=start_date, interval=interval)
    collector.collect()


if __name__ == "__main__":
    collect_cn_calendar()
