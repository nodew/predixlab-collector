"""Yahoo Finance data collector for US stocks.
This collector handles downloading, updating, and validating stock data
from Yahoo Finance using the yahooquery library. It supports both full history
and incremental updates, with robust error handling and anomaly detection.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Optional, List, Tuple
from datetime import datetime, timedelta
import time
import requests
from yahooquery import Ticker
from dateutil.tz import tzlocal

from config import settings

class YahooCollector:
    """Yahoo Finance data collector for US stocks."""

    # Constants
    DEFAULT_START_DATE = "2015-01-01"
    ABNORMAL_CHANGE_THRESHOLD = 0.5  # 50% change threshold for abnormal data detection
    RETRY_COUNT = 3
    DELAY_BETWEEN_REQUESTS = 0.5

    def __init__(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
        delay: float = 0.5,
        limit_nums: Optional[int] = None
    ):
        """Initialize Yahoo Finance collector.

        Parameters
        ----------
        start_date : Optional[str]
            Start date for data collection (YYYY-MM-DD)
        end_date : Optional[str]
            End date for data collection (YYYY-MM-DD)
        interval : str
            Data interval, default "1d"
        delay : float
            Delay between requests in seconds, default 0.5
        limit_nums : Optional[int]
            Limit the number of symbols to process (for testing), default None (process all)
        """
        self.start_date = start_date or self.DEFAULT_START_DATE
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.interval = interval
        self.delay = delay
        self.limit_nums = limit_nums

        # Get paths from config
        self.us_index_path = Path(settings.us_index_path)
        self.us_stock_data_dir = Path(settings.us_stock_data_dir)

        # Ensure data directory exists
        self.us_stock_data_dir.mkdir(parents=True, exist_ok=True)

        # Track abnormal tickers for full re-download
        self.abnormal_tickers = set()
        self.abnormal_log_path = self.us_stock_data_dir / "abnormal_tickers.txt"

        logger.info(f"Yahoo collector initialized")
        logger.info(f"Index file: {self.us_index_path}")
        logger.info(f"Data directory: {self.us_stock_data_dir}")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")

    def _load_stock_symbols(self) -> List[str]:
        """Load stock symbols from us_index_path.

        Returns
        -------
        List[str]
            List of stock symbols
        """
        if not self.us_index_path.exists():
            raise FileNotFoundError(f"US index file not found: {self.us_index_path}")

        try:
            # Read tab-separated file: symbol \t start_date \t end_date
            df = pd.read_csv(self.us_index_path, sep='\t', header=None,
                           names=['symbol', 'start_date', 'end_date'])

            symbols = df['symbol'].unique().tolist()
            logger.info(f"Loaded {len(symbols)} symbols from index file")

            # Apply limit if specified
            if self.limit_nums is not None and self.limit_nums > 0:
                original_count = len(symbols)
                symbols = symbols[:self.limit_nums]
                logger.info(f"Applied limit_nums={self.limit_nums}: processing {len(symbols)} out of {original_count} symbols")

            return symbols

        except Exception as e:
            logger.error(f"Failed to load symbols from {self.us_index_path}: {e}")
            raise

    def _get_data_from_yahoo(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """Get stock data from Yahoo Finance.

        Parameters
        ----------
        symbol : str
            Stock symbol
        start : str
            Start date (YYYY-MM-DD)
        end : str
            End date (YYYY-MM-DD)

        Returns
        -------
        Optional[pd.DataFrame]
            Stock data DataFrame or None if failed
        """
        for attempt in range(self.RETRY_COUNT):
            try:
                time.sleep(self.delay)

                ticker = Ticker(symbol, asynchronous=False)
                data = ticker.history(interval=self.interval, start=start, end=end)

                if isinstance(data, pd.DataFrame) and not data.empty:
                    # Reset index to get date as column
                    data = data.reset_index()

                    # Add symbol column if not present
                    if 'symbol' not in data.columns:
                        data['symbol'] = symbol

                    # Ensure required columns exist
                    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
                    if all(col in data.columns for col in required_cols):
                        return data
                    else:
                        logger.warning(f"Missing required columns for {symbol}")

                elif isinstance(data, dict) and symbol in data:
                    # Handle case where data is returned as dict
                    symbol_data = data[symbol]
                    if isinstance(symbol_data, pd.DataFrame) and not symbol_data.empty:
                        symbol_data = symbol_data.reset_index()
                        symbol_data['symbol'] = symbol
                        return symbol_data

                logger.warning(f"Empty or invalid data for {symbol} (attempt {attempt + 1}/{self.RETRY_COUNT})")

            except Exception as e:
                logger.warning(f"Error fetching {symbol} (attempt {attempt + 1}/{self.RETRY_COUNT}): {e}")
                if attempt < self.RETRY_COUNT - 1:
                    time.sleep(self.delay * (attempt + 1))

        logger.error(f"Failed to fetch data for {symbol} after {self.RETRY_COUNT} attempts")
        return None

    def _get_existing_data_info(self, symbol: str) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
        """Get information about existing data file.

        Parameters
        ----------
        symbol : str
            Stock symbol

        Returns
        -------
        Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]
            (existing_data, min_date, max_date) or (None, None, None) if no file
        """
        file_path = self.us_stock_data_dir / f"{symbol.upper()}.csv"

        if not file_path.exists():
            return None, None, None

        try:
            df = pd.read_csv(file_path)

            if df.empty or 'date' not in df.columns:
                return None, None, None

            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')

            min_date = df['date'].min().strftime("%Y-%m-%d")
            max_date = df['date'].max().strftime("%Y-%m-%d")

            return df, min_date, max_date

        except Exception as e:
            logger.warning(f"Error reading existing data for {symbol}: {e}")
            return None, None, None

    def _detect_data_anomalies(self, df: pd.DataFrame, symbol: str) -> bool:
        """Detect data anomalies in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Stock data DataFrame
        symbol : str
            Stock symbol

        Returns
        -------
        bool
            True if anomalies detected, False otherwise
        """
        if df.empty or len(df) < 2:
            return False

        try:
            # Calculate daily returns
            df_sorted = df.sort_values('date').copy()
            df_sorted['prev_close'] = df_sorted['close'].shift(1)
            df_sorted['daily_return'] = (df_sorted['close'] / df_sorted['prev_close'] - 1).abs()

            # Check for extreme price changes
            extreme_changes = df_sorted['daily_return'] > self.ABNORMAL_CHANGE_THRESHOLD

            if extreme_changes.any():
                anomaly_count = extreme_changes.sum()
                logger.warning(f"Detected {anomaly_count} extreme price changes for {symbol}")
                return True

            # Check for zero or negative prices
            invalid_prices = (df_sorted['close'] <= 0) | (df_sorted['open'] <= 0) | \
                           (df_sorted['high'] <= 0) | (df_sorted['low'] <= 0)

            if invalid_prices.any():
                invalid_count = invalid_prices.sum()
                logger.warning(f"Detected {invalid_count} invalid prices for {symbol}")
                return True

            # Check for illogical OHLC relationships
            illogical_ohlc = (df_sorted['high'] < df_sorted['low']) | \
                           (df_sorted['high'] < df_sorted['open']) | \
                           (df_sorted['high'] < df_sorted['close']) | \
                           (df_sorted['low'] > df_sorted['open']) | \
                           (df_sorted['low'] > df_sorted['close'])

            if illogical_ohlc.any():
                illogical_count = illogical_ohlc.sum()
                logger.warning(f"Detected {illogical_count} illogical OHLC relationships for {symbol}")
                return True

            return False

        except Exception as e:
            logger.warning(f"Error detecting anomalies for {symbol}: {e}")
            return False

    def _merge_data(self, existing_df: pd.DataFrame, new_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Merge existing and new data.

        Parameters
        ----------
        existing_df : pd.DataFrame
            Existing data
        new_df : pd.DataFrame
            New data to merge
        symbol : str
            Stock symbol

        Returns
        -------
        pd.DataFrame
            Merged data
        """
        try:
            # Ensure date columns are datetime
            existing_df['date'] = pd.to_datetime(existing_df['date'])
            new_df['date'] = pd.to_datetime(new_df['date'])

            # Combine and remove duplicates, keeping new data
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
            combined_df = combined_df.sort_values('date').reset_index(drop=True)

            # Check for anomalies in merged data
            if self._detect_data_anomalies(combined_df, symbol):
                logger.warning(f"Anomalies detected in merged data for {symbol}, marking for full re-download")
                self.abnormal_tickers.add(symbol)
                return existing_df  # Return existing data for now

            logger.info(f"Successfully merged data for {symbol}: {len(combined_df)} total records")
            return combined_df

        except Exception as e:
            logger.error(f"Error merging data for {symbol}: {e}")
            self.abnormal_tickers.add(symbol)
            return existing_df

    def _save_data(self, df: pd.DataFrame, symbol: str) -> None:
        """Save data to CSV file.

        Parameters
        ----------
        df : pd.DataFrame
            Data to save
        symbol : str
            Stock symbol
        """
        if df.empty:
            logger.warning(f"No data to save for {symbol}")
            return

        try:
            file_path = self.us_stock_data_dir / f"{symbol.upper()}.csv"

            # Ensure symbol column exists
            if 'symbol' not in df.columns:
                df['symbol'] = symbol

            # Sort by date and save
            df_sorted = df.sort_values('date').reset_index(drop=True)
            df_sorted.to_csv(file_path, index=False)

            logger.info(f"Saved {len(df_sorted)} records for {symbol} to {file_path}")

        except Exception as e:
            logger.error(f"Error saving data for {symbol}: {e}")

    def _collect_symbol_data(self, symbol: str) -> None:
        """Collect data for a single symbol.

        Parameters
        ----------
        symbol : str
            Stock symbol to collect
        """
        logger.info(f"Processing symbol: {symbol}")

        # Get existing data info
        existing_df, min_date, max_date = self._get_existing_data_info(symbol)

        if existing_df is None:
            # No existing data, download full history
            logger.info(f"No existing data for {symbol}, downloading full history from {self.DEFAULT_START_DATE}")
            new_df = self._get_data_from_yahoo(symbol, self.DEFAULT_START_DATE, self.end_date)

            if new_df is not None:
                self._save_data(new_df, symbol)
            else:
                logger.error(f"Failed to download full data for {symbol}")

        else:
            # Existing data found, download incremental data
            logger.info(f"Existing data for {symbol}: {min_date} to {max_date}")

            # Determine date range for new data
            start_download = max(pd.Timestamp(self.start_date), pd.Timestamp(max_date) + pd.Timedelta(days=1))
            end_download = pd.Timestamp(self.end_date)

            if start_download <= end_download:
                start_str = start_download.strftime("%Y-%m-%d")
                end_str = end_download.strftime("%Y-%m-%d")

                logger.info(f"Downloading incremental data for {symbol}: {start_str} to {end_str}")
                new_df = self._get_data_from_yahoo(symbol, start_str, end_str)

                if new_df is not None and not new_df.empty:
                    # Merge with existing data
                    merged_df = self._merge_data(existing_df, new_df, symbol)
                    self._save_data(merged_df, symbol)
                else:
                    logger.info(f"No new data available for {symbol}")
            else:
                logger.info(f"Existing data for {symbol} is up to date")

    def _redownload_abnormal_tickers(self) -> None:
        """Re-download full data for tickers marked as abnormal."""
        if not self.abnormal_tickers:
            return

        logger.warning(f"Re-downloading full data for {len(self.abnormal_tickers)} abnormal tickers")

        # Save abnormal tickers to file for reference
        with open(self.abnormal_log_path, 'w') as f:
            f.write(f"Abnormal tickers detected on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:\n")
            for ticker in sorted(self.abnormal_tickers):
                f.write(f"{ticker}\n")

        for symbol in self.abnormal_tickers:
            logger.info(f"Re-downloading full data for abnormal ticker: {symbol}")
            new_df = self._get_data_from_yahoo(symbol, self.DEFAULT_START_DATE, self.end_date)

            if new_df is not None:
                self._save_data(new_df, symbol)
                logger.info(f"Successfully re-downloaded data for {symbol}")
            else:
                logger.error(f"Failed to re-download data for {symbol}")

    def _get_batch_data_from_yahoo(self, symbols: List[str], start: str, end: str) -> Optional[pd.DataFrame]:
        """Get batch stock data from Yahoo Finance for multiple symbols.

        Parameters
        ----------
        symbols : List[str]
            List of stock symbols
        start : str
            Start date (YYYY-MM-DD)
        end : str
            End date (YYYY-MM-DD)

        Returns
        -------
        Optional[pd.DataFrame]
            Combined stock data DataFrame with MultiIndex (symbol, date) or None if failed
        """
        if not symbols:
            return None

        for attempt in range(self.RETRY_COUNT):
            try:
                time.sleep(self.delay)

                # Use space-separated string for multiple symbols
                symbols_str = " ".join(symbols)
                ticker = Ticker(symbols_str, asynchronous=False)
                data = ticker.history(interval=self.interval, start=start, end=end)

                if isinstance(data, pd.DataFrame) and not data.empty:
                    # Check if we have MultiIndex (symbol, date)
                    if isinstance(data.index, pd.MultiIndex):
                        # Reset MultiIndex to get symbol and date as columns
                        data = data.reset_index()

                        # Ensure required columns exist
                        required_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
                        if all(col in data.columns for col in required_cols):
                            return data
                        else:
                            logger.warning(f"Missing required columns in batch data for {symbols}")
                    else:
                        # Single symbol case
                        data = data.reset_index()
                        if len(symbols) == 1:
                            data['symbol'] = symbols[0]
                            return data

                logger.warning(f"Empty or invalid batch data for {symbols} (attempt {attempt + 1}/{self.RETRY_COUNT})")

            except Exception as e:
                logger.warning(f"Error fetching batch data for {symbols} (attempt {attempt + 1}/{self.RETRY_COUNT}): {e}")
                if attempt < self.RETRY_COUNT - 1:
                    time.sleep(self.delay * (attempt + 1))

        logger.error(f"Failed to fetch batch data for {symbols} after {self.RETRY_COUNT} attempts")
        return None

    def _get_dynamic_batch_size(self, start_date: str, end_date: str) -> int:
        """Calculate optimal batch size based on date range and interval.

        Parameters
        ----------
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)

        Returns
        -------
        int
            Optimal batch size for the given date range and interval
        """
        try:
            period_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
        except Exception:
            period_days = 30  # fallback default

        # Base batch sizes for different intervals
        if self.interval == "1d":
            # Daily data batch sizing based on period length
            if period_days <= 5:
                return 50
            elif period_days <= 10:
                return 30
            elif period_days <= 20:
                return 20
            elif period_days <= 30:
                return 10
            else:
                # For longer periods, use smaller batches to avoid API limits
                return 5

        elif self.interval == "1min":
            # Minute data requires much smaller batches due to data volume
            if period_days <= 1:
                return 10
            elif period_days <= 5:
                return 5
            elif period_days <= 10:
                return 3
            else:
                # Very small batches for long minute data periods
                return 2

        else:
            # Default for other intervals
            return 10

    def _collect_batch_incremental_data(self, symbols_need_update: List[str], start_date: str, end_date: str) -> dict:
        """Collect incremental data for multiple symbols in batches.

        Parameters
        ----------
        symbols_need_update : List[str]
            List of symbols that need incremental updates
        start_date : str
            Start date for incremental data
        end_date : str
            End date for incremental data

        Returns
        -------
        dict
            Dictionary mapping symbol to new DataFrame data
        """
        batch_results = {}
        batch_size = self._get_dynamic_batch_size(start_date, end_date)

        period_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
        logger.info(f"Collecting incremental data for {len(symbols_need_update)} symbols")
        logger.info(f"Period: {period_days} days, Interval: {self.interval}, Dynamic batch size: {batch_size}")

        for i in range(0, len(symbols_need_update), batch_size):
            batch_symbols = symbols_need_update[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(symbols_need_update) + batch_size - 1) // batch_size

            logger.info(f"Processing batch {batch_num}/{total_batches}: {batch_symbols}")

            batch_data = self._get_batch_data_from_yahoo(batch_symbols, start_date, end_date)

            if batch_data is not None and not batch_data.empty:
                # Split batch data by symbol
                for symbol in batch_symbols:
                    symbol_data = batch_data[batch_data['symbol'] == symbol]
                    if not symbol_data.empty:
                        batch_results[symbol] = symbol_data.copy()
                        logger.info(f"Retrieved {len(symbol_data)} records for {symbol}")
                    else:
                        logger.warning(f"No data found for {symbol} in batch")
            else:
                logger.warning(f"Failed to get batch data for {batch_symbols}")
                # Fall back to individual downloads for this batch
                for symbol in batch_symbols:
                    individual_data = self._get_data_from_yahoo(symbol, start_date, end_date)
                    if individual_data is not None:
                        batch_results[symbol] = individual_data

            # Small delay between batches
            time.sleep(self.delay)

        return batch_results

    def collect(self) -> None:
        """Main collection method with optimized batch processing."""
        logger.info("Starting Yahoo Finance data collection")

        try:
            # Load stock symbols from config
            symbols = self._load_stock_symbols()
            logger.info(f"Starting data collection for {len(symbols)} symbols")

            # Phase 1: Analyze existing data and categorize symbols
            symbols_full_download = []  # Symbols with no existing data
            symbols_incremental = []    # Symbols needing incremental updates
            incremental_date_range = None

            logger.info("Analyzing existing data files...")
            for symbol in symbols:
                existing_df, min_date, max_date = self._get_existing_data_info(symbol)

                if existing_df is None:
                    symbols_full_download.append(symbol)
                else:
                    # Check if incremental update is needed
                    start_download = max(pd.Timestamp(self.start_date), pd.Timestamp(max_date) + pd.Timedelta(days=1))
                    end_download = pd.Timestamp(self.end_date)

                    if start_download <= end_download:
                        symbols_incremental.append(symbol)
                        if incremental_date_range is None:
                            incremental_date_range = (start_download.strftime("%Y-%m-%d"), end_download.strftime("%Y-%m-%d"))

            logger.info(f"Analysis complete - Full download: {len(symbols_full_download)}, Incremental: {len(symbols_incremental)}")

            # Phase 2: Handle full downloads (individual downloads for better error handling)
            if symbols_full_download:
                logger.info(f"Downloading full history for {len(symbols_full_download)} symbols...")
                for i, symbol in enumerate(symbols_full_download, 1):
                    logger.info(f"Full download progress: {i}/{len(symbols_full_download)} - {symbol}")
                    new_df = self._get_data_from_yahoo(symbol, self.DEFAULT_START_DATE, self.end_date)

                    if new_df is not None:
                        self._save_data(new_df, symbol)
                    else:
                        logger.error(f"Failed to download full data for {symbol}")

                    time.sleep(self.delay)

            # Phase 3: Handle incremental updates (batch downloads for efficiency)
            if symbols_incremental and incremental_date_range:
                start_date, end_date = incremental_date_range
                logger.info(f"Batch downloading incremental data ({start_date} to {end_date}) for {len(symbols_incremental)} symbols...")

                batch_results = self._collect_batch_incremental_data(symbols_incremental, start_date, end_date)

                # Process and merge the batch results
                for symbol in symbols_incremental:
                    existing_df, _, _ = self._get_existing_data_info(symbol)
                    new_df = batch_results.get(symbol)

                    if new_df is not None and not new_df.empty:
                        # Merge with existing data
                        merged_df = self._merge_data(existing_df, new_df, symbol)
                        self._save_data(merged_df, symbol)
                    elif existing_df is not None:
                        logger.info(f"No new data available for {symbol}, existing data is up to date")

            # Phase 4: Re-download any abnormal tickers
            if self.abnormal_tickers:
                logger.warning(f"Found {len(self.abnormal_tickers)} abnormal tickers, re-downloading...")
                self._redownload_abnormal_tickers()

            logger.info("Yahoo Finance data collection completed successfully!")
            logger.info(f"Data saved to: {self.us_stock_data_dir}")

            if self.abnormal_tickers:
                logger.info(f"Abnormal tickers log saved to: {self.abnormal_log_path}")

        except Exception as e:
            logger.error(f"Data collection failed: {e}")
            raise


def collect_yahoo_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1d",
    delay: float = 0.5,
    limit_nums: Optional[int] = None
) -> None:
    """Main entry point for Yahoo data collection.

    Parameters
    ----------
    start_date : Optional[str]
        Start date for data collection (YYYY-MM-DD)
    end_date : Optional[str]
        End date for data collection (YYYY-MM-DD)
    interval : str
        Data interval, default "1d"
    delay : float
        Delay between requests in seconds, default 0.5
    limit_nums : Optional[int]
        Limit the number of symbols to process (for testing), default None (process all)
    """
    collector = YahooCollector(
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        delay=delay,
        limit_nums=limit_nums
    )
    collector.collect()


if __name__ == "__main__":
    import fire
    fire.Fire(collect_yahoo_data)
