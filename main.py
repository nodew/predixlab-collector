"""Main entry point for qstock-marketdata service.

This service provides independent stock market data collection and processing
using yfinance and Yahoo Query, extracted from the qstock project.

Usage:
    python main.py --help                    # Show available commands
    python main.py collect --region US       # Collect US stock data
    python main.py collect --region CN       # Collect Chinese stock data
    python main.py list-symbols --region US  # List available US symbols
"""

from datetime import datetime
from pathlib import Path
import fire
from loguru import logger

from collectors.us_index import collect_us_index
from collectors.us_calendar import collect_us_calendar
from collectors.yahoo import collect_yahoo_data, normalize_yahoo_data
from config import settings

class QStockMarketDataService:
    """Main service class for stock market data operations."""

    def collect_us_index(self):
        """Collect US index constituents (SP500 + NASDAQ100).

        This method fetches the latest constituents of SP500 and NASDAQ100 indices
        from Wikipedia, merges them, and saves to the configured us_index_path.
        """
        logger.info("Starting US index collection...")
        try:
            collect_us_index()
            logger.info("✅ US index collection completed successfully!")
        except Exception as e:
            logger.error(f"❌ US index collection failed: {e}")

    def collect_us_calendar(self, start_date: str = "2015-01-01"):
        """Collect US stock trading calendar dates.

        This method fetches US stock trading calendar dates from start_date to current date
        using Yahoo Finance data and saves to the configured us_calendar_path.

        Parameters
        ----------
        start_date : str, optional
            Start date for collecting calendar data, by default "2015-01-01"
        """
        logger.info("Starting US trading calendar collection...")
        try:
            collect_us_calendar(start_date=start_date)
            logger.info("✅ US trading calendar collection completed successfully!")
        except Exception as e:
            logger.error(f"❌ US trading calendar collection failed: {e}")
            raise

    def collect_yahoo_data(
        self,
        start_date: str = None,
        end_date: str = None,
        interval: str = "1d",
        delay: float = 0.5,
        limit_nums: int = None
    ):
        """Collect US stock data from Yahoo Finance.

        This method implements Yahoo Finance data collection with the following features:
        1. Gets US stock list from config.us_index_path instead of downloading from network
        2. Saves raw data to config.us_stock_data_dir instead of parameter-specified directory
        3. Checks local files and downloads full data (from 2015-01-01) if missing,
           or downloads incremental data based on date range and merges with existing
        4. Marks abnormal data during merge and re-downloads full data if needed

        Parameters
        ----------
        start_date : str, optional
            Start date for data collection (YYYY-MM-DD), by default None (uses 2015-01-01)
        end_date : str, optional
            End date for data collection (YYYY-MM-DD), by default None (uses current date)
        interval : str, optional
            Data interval, by default "1d"
        delay : float, optional
            Delay between requests in seconds, by default 0.5
        limit_nums : int, optional
            Limit the number of symbols to process (for testing), by default None (process all)
        """
        logger.info("Starting Yahoo Finance stock data collection...")
        logger.info(f"Parameters: start={start_date}, end={end_date}, interval={interval}, limit_nums={limit_nums}")

        try:
            collect_yahoo_data(
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                delay=delay,
                limit_nums=limit_nums
            )
            logger.info("✅ Yahoo Finance stock data collection completed successfully!")
        except Exception as e:
            logger.error(f"❌ Yahoo Finance stock data collection failed: {e}")
            raise

    def normalize_yahoo_data(
        self,
        start_date: str = None,
        end_date: str = None,
        max_workers: int = None,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol"
    ):
        """Normalize US stock data from Yahoo Finance.

        This method processes raw stock data stored in config.us_stock_data_dir and
        applies standardization, anomaly detection, and adjustment calculations to
        prepare data for analysis. The normalized data is saved to config.us_normalized_data_dir.

        The normalization process includes:
        1. Basic data normalization (timezone handling, duplicate removal, anomaly correction)
        2. Price adjustment for splits and dividends using adjusted close prices
        3. Manual adjustment to normalize all fields relative to the first day's close price

        Parameters
        ----------
        start_date : str, optional
            Start date for normalization (YYYY-MM-DD), by default None (process all data)
        end_date : str, optional
            End date for normalization (YYYY-MM-DD), by default None (process all data)
        max_workers : int, optional
            Number of worker processes for parallel processing, by default None (auto-detect)
        date_field_name : str, optional
            Date field name in the data, by default "date"
        symbol_field_name : str, optional
            Symbol field name in the data, by default "symbol"
        """
        logger.info("Starting Yahoo Finance stock data normalization...")
        logger.info(f"Parameters: start={start_date}, end={end_date}, max_workers={max_workers}")

        try:
            normalize_yahoo_data(
                start_date=start_date,
                end_date=end_date,
                max_workers=max_workers,
                date_field_name=date_field_name,
                symbol_field_name=symbol_field_name
            )
            logger.info("✅ Yahoo Finance stock data normalization completed successfully!")
        except Exception as e:
            logger.error(f"❌ Yahoo Finance stock data normalization failed: {e}")
            raise

    def update_daily_data(self):
        """Update daily stock data."""
        try:
            logger.info("Starting the update of daily stock data...")

            current_date = datetime.now().strftime("%Y-%m-%d")
            calendar_path = Path(settings.us_calendar_path)
            if not calendar_path.exists():
                logger.warning(f"US calendar file not found: {calendar_path}, defaulting last_trading_date to 2015-01-01")
                last_trading_date = "2015-01-01"
            else:
                with calendar_path.open('r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]

                if not lines:
                    logger.warning(f"US calendar file is empty: {calendar_path}, defaulting last_trading_date to 2015-01-01")
                    last_trading_date = "2015-01-01"
                else:
                    last_trading_date = lines[-2] if len(lines) >= 2 else lines[0]
            # Validate date format
            datetime.strptime(last_trading_date, "%Y-%m-%d")
            logger.info(f"Last trading date from calendar: {last_trading_date}")

            # 1) Ensure calendar is up to date first
            self.collect_us_calendar(start_date=last_trading_date)

            # 2) Update index, collect and normalize only up to last trading date
            self.collect_us_index()

            # 3) Collect and normalize stock data from yahoo finance API
            self.collect_yahoo_data(interval="1d", start_date=last_trading_date, end_date=current_date)
            self.normalize_yahoo_data()

            logger.info("✅ Full update of 1-day interval stock data completed successfully!")
        except Exception as e:
            logger.error(f"❌ Full update of 1-day interval stock data failed: {e}")
            raise

def main():
    """Main entry point."""
    fire.Fire(QStockMarketDataService)

if __name__ == "__main__":
    main()
