"""Main entry point for qstock-marketdata service.

This service provides independent stock market data collection and processing
using yfinance and Yahoo Query, extracted from the qstock project.

Usage:
    python main.py --help                    # Show available commands
    python main.py collect --region US       # Collect US stock data
    python main.py collect --region CN       # Collect Chinese stock data
    python main.py list-symbols --region US  # List available US symbols
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
import fire
from loguru import logger
from pymongo import MongoClient

from collectors.us_index import collect_us_index
from collectors.us_calendar import collect_us_calendar
from collectors.yahoo import collect_yahoo_data, normalize_yahoo_data
from config import settings
from notification import send_email_notification

class QStockMarketDataService:
    """Main service class for stock market data operations."""

    def _save_job_status_to_db(self, job_status: Dict[str, Any]) -> bool:
        """
        Save job execution status to MongoDB.

        Args:
            job_status: Dictionary containing job execution status and metadata

        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            client = MongoClient(settings.mongodb_url)
            db = client[settings.database_name]
            coll = db[settings.jobs_collection]

            # Add timestamp if not present
            if 'created_at' not in job_status:
                job_status['created_at'] = datetime.now(timezone.utc)

            # Insert job status document
            result = coll.insert_one(job_status)
            
            logger.info(f"Job status saved to database. Document ID: {result.inserted_id}")
            client.close()
            return True

        except Exception as e:
            logger.error(f"Failed to save job status to database: {e}")
            return False

    def _post_job(self, job_status: Dict[str, Any]) -> None:
        """
        Execute post-job tasks: save job status to database and send email notification.

        Args:
            job_status: Dictionary containing job execution status and metadata
                Expected keys:
                    - job_name: Internal job identifier (stable, used for DB persistence / queries)
                    - job_display_name: Human readable display name (used for notifications / UI)
                    - status: 'success' or 'failed'
                    - start_time: ISO format start time
                    - end_time: ISO format end time
                    - duration_seconds: Duration in seconds
                    - results: Optional dict with execution results
                    - error: Optional error message if failed
        """
        logger.info("Executing post-job tasks (DB save -> notification)")

        # 1. Save job status to database first so notification reflects persisted state
        db_saved = self._save_job_status_to_db(job_status)
        if db_saved:
            logger.info("Job status saved to database successfully")
        else:
            logger.warning("Job status was not saved to database")

        # 2. Send email notification (include note if DB failed)
        if not db_saved:
            job_status.setdefault('warning', 'DB save failed prior to notification')

        email_sent = send_email_notification(job_status)
        if email_sent:
            logger.info("Email notification sent successfully")
        else:
            logger.warning("Email notification was not sent")

        logger.info("Post-job tasks completed")

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

    def collect_us_calendar(self, start_date: str = "2015-01-01", interval: str = "1d"):
        """Collect US stock trading calendar dates.

        This method fetches US stock trading calendar dates from start_date to current date
        using Yahoo Finance data and saves to the configured us_calendar_path.

        Parameters
        ----------
        start_date : str, optional
            Start date for collecting calendar data, by default "2015-01-01"
        interval : str, optional
            Data interval, by default "1d". Supported values: "1d", "1wk", "1mo"
        """
        logger.info("Starting US trading calendar collection...")
        try:
            collect_us_calendar(start_date=start_date, interval=interval)
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

    def update_daily_data(self, no_upload: bool = False):
        """Update daily stock data with job status tracking and notifications.

        Parameters
        ----------
        no_upload : bool, optional
            If True, skip uploading job status to database and sending email notifications, by default False
        """
        start_time = datetime.now()
        job_status = {
            'job_name': 'predixlab_daily_marketdata_collector',  # machine-friendly identifier
            'job_display_name': 'PredixLab Daily Market Data Collector',  # human readable
            'start_time': start_time.isoformat(),
            'status': 'failed',  # Default to failed, will update on success
            'results': {}
        }

        try:
            logger.info("Starting the update of daily stock data...")

            current_date = datetime.now().strftime("%Y-%m-%d")
            calendar_path = Path(settings.us_calendar_path).expanduser()
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
            self.collect_us_calendar(start_date=last_trading_date, interval="1d")

            # 2) Update index, collect and normalize only up to last trading date
            self.collect_us_index()

            # Count symbols processed
            index_path = Path(settings.us_index_path).expanduser()
            symbols_count = 0
            if index_path.exists():
                with index_path.open('r', encoding='utf-8') as f:
                    symbols_count = sum(1 for line in f if line.strip())

            # 3) Collect and normalize stock data from yahoo finance API
            self.collect_yahoo_data(interval="1d", start_date=last_trading_date, end_date=current_date)
            self.normalize_yahoo_data()

            logger.info("✅ Full update of 1-day interval stock data completed successfully!")
            
            # Update job status with success
            job_status['status'] = 'success'
            job_status['results'] = {
                'last_trading_date': last_trading_date,
                'symbols_processed': symbols_count,
                'data_collected': True,
                'data_normalized': True
            }

        except Exception as e:
            logger.error(f"❌ Full update of 1-day interval stock data failed: {e}")
            job_status['error'] = str(e)
            raise
        finally:
            # Execute post-job tasks unless no_upload is specified
            end_time = datetime.now()
            job_status['end_time'] = end_time.isoformat()
            job_status['duration_seconds'] = (end_time - start_time).total_seconds()

            if no_upload:
                logger.info("Skipping post-job tasks (no_upload=True)")
            else:
                self._post_job(job_status)

    def update_weekly_data(self, no_upload: bool = False):
        """Update weekly stock data with job status tracking and notifications.

        Parameters
        ----------
        no_upload : bool, optional
            If True, skip uploading job status to database and sending email notifications, by default False
        """
        start_time = datetime.now()
        job_status = {
            'job_name': 'predixlab_weekly_marketdata_collector',  # machine-friendly identifier
            'job_display_name': 'PredixLab Weekly Market Data Collector',  # human readable
            'start_time': start_time.isoformat(),
            'status': 'failed',  # Default to failed, will update on success
            'results': {}
        }

        try:
            logger.info("Starting the update of weekly stock data...")

            current_date = datetime.now().strftime("%Y-%m-%d")
            calendar_path = Path(settings.us_weekly_calendar_path).expanduser()
            if not calendar_path.exists():
                logger.warning(f"US weekly calendar file not found: {calendar_path}, defaulting last_trading_date to 2007-12-31")
                last_trading_date = "2007-12-31"
            else:
                with calendar_path.open('r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]

                if not lines:
                    logger.warning(f"US weekly calendar file is empty: {calendar_path}, defaulting last_trading_date to 2007-12-31")
                    last_trading_date = "2007-12-31"
                else:
                    last_trading_date = lines[-2] if len(lines) >= 2 else lines[0]
            # Validate date format
            datetime.strptime(last_trading_date, "%Y-%m-%d")
            logger.info(f"Last trading date from weekly calendar: {last_trading_date}")

            # 1) Ensure calendar is up to date first
            self.collect_us_calendar(start_date=last_trading_date, interval="1wk")

            # Count symbols processed
            index_path = Path(settings.us_index_path).expanduser()
            symbols_count = 0
            if index_path.exists():
                with index_path.open('r', encoding='utf-8') as f:
                    symbols_count = sum(1 for line in f if line.strip())

            # 3) Collect and normalize weekly stock data from Yahoo Finance API
            logger.info("Collecting weekly data...")
            from collectors.yahoo import YahooCollector, YahooNormalizer
            
            # Collect weekly data
            collector = YahooCollector(
                start_date=last_trading_date,
                end_date=current_date,  # Far future date to ensure up to current
                interval="1wk",
                delay=0.5,
                limit_nums=None
            )
            collector.collect()

            # Normalize weekly data
            logger.info("Normalizing weekly data...")
            normalizer = YahooNormalizer(
                start_date=None,
                end_date=None,
                interval="1wk"
            )
            normalizer.normalize()

            logger.info("✅ Full update of weekly stock data completed successfully!")
            
            # Update job status with success
            job_status['status'] = 'success'
            job_status['results'] = {
                'last_trading_date': last_trading_date,
                'symbols_processed': symbols_count,
                'data_collected': True,
                'data_normalized': True,
                'interval': '1wk'
            }

        except Exception as e:
            logger.error(f"❌ Full update of weekly stock data failed: {e}")
            job_status['error'] = str(e)
            raise
        finally:
            # Execute post-job tasks unless no_upload is specified
            end_time = datetime.now()
            job_status['end_time'] = end_time.isoformat()
            job_status['duration_seconds'] = (end_time - start_time).total_seconds()

            if no_upload:
                logger.info("Skipping post-job tasks (no_upload=True)")
            else:
                self._post_job(job_status)

def main():
    """Main entry point."""
    fire.Fire(QStockMarketDataService)

if __name__ == "__main__":
    main()
