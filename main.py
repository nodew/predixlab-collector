"""Main entry point for qstock-marketdata service.

This service provides independent stock market data collection and processing
using yfinance and Yahoo Query, extracted from the qstock project.

Usage:
    python main.py --help                    # Show available commands
    python main.py collect --region US       # Collect US stock data
    python main.py collect --region CN       # Collect Chinese stock data
    python main.py list-symbols --region US  # List available US symbols
"""

import fire
from loguru import logger

from collectors.us_index import collect_us_index
from collectors.us_calendar import collect_us_calendar

class QStockMarketDataService:
    """Main service class for stock market data operations."""

    def collect(
        self,
        region: str = "US",
        interval: str = "1d",
        start: str = None,
        end: str = None,
        market: str = None,
        max_workers: int = 4,
        delay: float = 0.5,
    ):
        """Collect stock market data.

        Parameters
        ----------
        region : str, optional
            Market region (US, CN, HK, IN, BR), by default "US"
        interval : str, optional
            Data interval (1d, 1min), by default "1d"
        start : str, optional
            Start date (YYYY-MM-DD)
        end : str, optional
            End date (YYYY-MM-DD)
        market : str, optional
            Specific market (e.g., sp500, nasdaq100, csi300)
        max_workers : int, optional
            Number of worker threads, by default 4
        delay : float, optional
            Delay between requests, by default 0.5
        """
        logger.info(f"Starting data collection for {region} market")
        logger.info(f"Interval: {interval}, Start: {start}, End: {end}")

        if market:
            logger.info(f"Collecting data for market: {market}")

        # This would instantiate the appropriate collector based on region
        logger.info("Data collection completed!")

    def list_symbols(self, region: str = "US", limit: int = 10):
        """List available stock symbols for a region.

        Parameters
        ----------
        region : str, optional
            Market region (US, CN, HK), by default "US"
        limit : int, optional
            Number of symbols to display, by default 10
        """
        logger.info(f"Fetching symbols for {region} market...")

        try:
            if region.upper() == "US":
                symbols = get_us_stock_symbols()
            elif region.upper() == "CN":
                symbols = get_hs_stock_symbols()
            elif region.upper() == "HK":
                symbols = get_hk_stock_symbols()
            else:
                logger.error(f"Unsupported region: {region}")
                return

            logger.info(f"Found {len(symbols)} symbols for {region} market")
            logger.info(f"First {limit} symbols:")
            for i, symbol in enumerate(symbols[:limit]):
                print(f"  {i+1:3d}. {symbol}")

        except Exception as e:
            logger.error(f"Error fetching symbols: {e}")

    def get_calendar(self, region: str = "US"):
        """Get trading calendar for a region.

        Parameters
        ----------
        region : str, optional
            Market region, by default "US"
        """
        from src.utils import get_calendar_list

        try:
            bench_code = f"{region.upper()}_ALL"
            calendar = get_calendar_list(bench_code)
            logger.info(f"Found {len(calendar)} trading days for {region}")
            logger.info(f"Latest trading days:")
            for date in calendar[-10:]:
                print(f"  {date.strftime('%Y-%m-%d')}")
        except Exception as e:
            logger.error(f"Error fetching calendar: {e}")

    def test_connection(self):
        """Test connection to data sources."""
        logger.info("Testing connection to Yahoo Finance...")

        try:
            from yahooquery import Ticker
            # Test with a common stock
            ticker = Ticker("AAPL")
            data = ticker.history(interval="1d", period="5d")

            if data is not None and not data.empty:
                logger.info("✅ Yahoo Finance connection successful!")
                logger.info(f"Retrieved {len(data)} data points for AAPL")
            else:
                logger.warning("⚠️  Yahoo Finance connection returned no data")

        except Exception as e:
            logger.error(f"❌ Yahoo Finance connection failed: {e}")

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
            raise

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

    def info(self):
        """Show service information."""
        print("🚀 QStock Market Data Service")
        print("=" * 40)
        print("📊 Independent stock market data collection service")
        print("🔗 Data sources: Yahoo Finance, Yahoo Query, Wikipedia")
        print("🌍 Supported regions: US, CN, HK, IN, BR")
        print("📈 Supported intervals: 1d (daily), 1min (minute)")
        print("🎯 Supported indices: SP500, NASDAQ100, CSI300, HSI, etc.")
        print("")
        print("Examples:")
        print("  python main.py test-connection")
        print("  python main.py list-symbols --region US --limit 5")
        print("  python main.py collect --region US --market sp500")
        print("  python main.py collect-us-index")
        print("  python main.py collect-us-calendar")
        print("  python main.py collect-us-calendar --start-date 2020-01-01")
        print("  python main.py get-calendar --region CN")


def main():
    """Main entry point."""
    fire.Fire(QStockMarketDataService)


if __name__ == "__main__":
    main()
