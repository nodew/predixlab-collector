"""US Calendar collector module.

This module collects US stock trading calendar dates from 2015-01-01 to current date
using Yahoo Finance data through market-prices, and saves them to the configured file.
"""

from collections.abc import Sequence
from datetime import date, datetime, timedelta

import pandas as pd
from pathlib import Path
from loguru import logger
import time
from market_prices import PricesYahoo

from config import settings
from utils import normalize_datetime_to_date

class USCalendarCollector:
    """Collector for US stock trading calendar dates."""

    # Constants
    DEFAULT_START_DATE = "2015-01-01"
    DEFAULT_WEEKLY_START_DATE = "2007-12-31"

    def __init__(self, start_date: str | None = None, interval: str = "1d"):
        """Initialize the US calendar collector.

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
            self.us_calendar_path = Path(settings.us_weekly_calendar_path).expanduser()
        else:
            # For "1d" and other intervals, use the default path
            self.us_calendar_path = Path(settings.us_calendar_path).expanduser()

        self.us_calendar_path.parent.mkdir(parents=True, exist_ok=True)
        # Use S&P 500 index (^GSPC) as reference for US trading calendar
        self.reference_symbol = "^GSPC"

    def get_us_trading_dates(self) -> list[date]:
        """Get US stock trading dates from start_date to current date using Yahoo Finance.

        Uses retry mechanism with exponential backoff (max 3 retries) for market-prices calls.

        Returns:
            List of trading dates as date objects
        """
        logger.info(f"Fetching US trading calendar (interval={self.interval}) from {self.start_date} to {self.end_date}...")

        max_retries = 3
        base_delay = 1.0  # Base delay in seconds

        for attempt in range(max_retries + 1):
            try:
                prices = PricesYahoo(self.reference_symbol, delays=0)

                # market-prices uses "1D" for daily bars. For weekly/monthly calendars
                # we derive period markers from daily trading dates to keep the existing
                # downstream expectations stable.
                hist = prices.get("1D", start=self.start_date, end=self.end_date)
                if hist is None:
                    raise ValueError(f"No data received for symbol {self.reference_symbol}")

                hist_data = pd.DataFrame(hist)
                if hist_data.empty:
                    raise ValueError(f"No data received for symbol {self.reference_symbol}")

                # Extract trading dates from the index (handle IntervalIndex just in case)
                idx = hist_data.index.left if isinstance(hist_data.index, pd.IntervalIndex) else hist_data.index
                raw_dates = pd.to_datetime(idx, errors="coerce")
                daily_dates: list[date] = []
                for d in raw_dates:
                    d2 = normalize_datetime_to_date(d)
                    if d2 is None:
                        continue
                    daily_dates.append(d2)
                daily_dates = sorted(set(daily_dates))

                if self.interval == "1d":
                    trading_dates = daily_dates
                elif self.interval == "1wk":
                    # Monday-anchored weekly markers
                    week_starts = {d - timedelta(days=d.weekday()) for d in daily_dates}
                    trading_dates = sorted(week_starts)
                elif self.interval == "1mo":
                    # Last trading day of each completed month
                    last_by_month = {}
                    for d in daily_dates:
                        key = (d.year, d.month)
                        last_by_month[key] = max(last_by_month.get(key, d), d)

                    # Exclude current (incomplete) month
                    today = date.today()
                    current_key = (today.year, today.month)
                    last_by_month.pop(current_key, None)
                    trading_dates = sorted(last_by_month.values())
                else:
                    # Fallback: treat as daily
                    trading_dates = daily_dates

                if (self.interval == "1wk"):
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
                trading_dates = [date for date in trading_dates if start_ts <= date <= end_ts]

                logger.info(f"Successfully fetched {len(trading_dates)} US trading dates")
                if trading_dates:
                    logger.info(f"Date range: {trading_dates[0]} to {trading_dates[-1]}")

                return trading_dates

            except Exception as e:
                if attempt < max_retries:
                    # Calculate delay with exponential backoff
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed to fetch US trading calendar: {e}")
                    logger.info(f"Retrying in {delay:.1f} seconds... ({attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"Failed to fetch US trading calendar after {max_retries + 1} attempts: {e}")
                    raise

        raise RuntimeError("Failed to fetch US trading calendar")

    def save_calendar(self, trading_dates: Sequence[date]) -> None:
        """Save trading dates to the configured calendar file using merge strategy.

        If the file already exists, merge the new dates with existing dates
        and save the union of both sets.

        Args:
            trading_dates: List of trading dates to save
        """
        try:
            # Convert dates to string format (YYYY-MM-DD)
            new_date_strings = {d.strftime("%Y-%m-%d") for d in trading_dates}

            # Check if file already exists and read existing dates
            existing_date_strings = set()
            if self.us_calendar_path.exists():
                logger.info(f"Existing calendar file found: {self.us_calendar_path}")
                with open(self.us_calendar_path, "r", encoding="utf-8") as f:
                    existing_date_strings = set(line.strip() for line in f if line.strip())
                logger.info(f"Loaded {len(existing_date_strings)} existing trading dates")

            # Merge new dates with existing dates (union)
            merged_date_strings = new_date_strings.union(existing_date_strings)

            # Sort the merged dates
            sorted_dates = sorted(merged_date_strings)

            # Save merged dates to file
            with open(self.us_calendar_path, "w", encoding="utf-8") as f:
                for date_str in sorted_dates:
                    f.write(f"{date_str}\n")

            logger.info(f"Merged and saved {len(sorted_dates)} US trading dates to: {self.us_calendar_path}")
            logger.info(f"  - Existing dates: {len(existing_date_strings)}")
            logger.info(f"  - New dates: {len(new_date_strings)}")
            logger.info(f"  - Total unique dates: {len(sorted_dates)}")

        except Exception as e:
            logger.error(f"Failed to save US calendar to {self.us_calendar_path}: {e}")
            raise

    def collect(self) -> None:
        """Collect US trading calendar and save to configured file."""
        logger.info("Starting US trading calendar collection...")

        try:
            # Get trading dates
            trading_dates = self.get_us_trading_dates()

            if not trading_dates:
                raise ValueError("No trading dates were collected")

            # Save to file
            self.save_calendar(trading_dates)

            logger.info("US trading calendar collection completed successfully!")

        except Exception as e:
            logger.error(f"US trading calendar collection failed: {e}")
            raise

def collect_us_calendar(start_date: str | None = None, interval: str = "1d"):
    """Main entry point for US trading calendar collection.

    Args:
        start_date: Start date for collecting calendar data. If None, uses 2015-01-01 for daily
                   data and 2007-12-31 for weekly data
        interval: Data interval, default is "1d". Supported values: "1d", "1wk", "1mo"
    """
    collector = USCalendarCollector(start_date=start_date, interval=interval)
    collector.collect()

if __name__ == "__main__":
    collect_us_calendar()
