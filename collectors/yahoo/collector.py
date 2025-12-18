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
from datetime import date, datetime, timedelta
import time
import requests
from yahooquery import Ticker
from dateutil.tz import tzlocal
from market_prices import PricesYahoo

from config import settings
from utils import normalize_datetime_to_date

class YahooCollector:
    """Yahoo Finance data collector for US stocks."""

    # Constants
    DEFAULT_START_DATE = "2015-01-01"
    DEFAULT_WEEKLY_START_DATE = "2007-12-31"
    ABNORMAL_CHANGE_THRESHOLD = 0.5  # 50% change threshold for abnormal data detection
    RETRY_COUNT = 3
    DELAY_BETWEEN_REQUESTS = 0.5
    # Interval format for market-prices library (uppercase D)
    MARKET_PRICES_DAILY_INTERVAL = "1D"
    # Interval format for yahooquery library
    YAHOO_WEEKLY_INTERVAL = "1wk"

    def __init__(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d",
        delay: float = 0.5,
        limit_nums: Optional[int] = None,
        data_dir: Optional[str] = None
    ):
        """Initialize Yahoo Finance collector.

        Parameters
        ----------
        start_date : Optional[str]
            Start date for data collection (YYYY-MM-DD)
        end_date : Optional[str]
            End date for data collection (YYYY-MM-DD)
        interval : str
            Data interval, default "1d" (supports "1d", "1wk", etc.)
        delay : float
            Delay between requests in seconds, default 0.5
        limit_nums : Optional[int]
            Limit the number of symbols to process (for testing), default None (process all)
        data_dir : Optional[str]
            Custom data directory path, default None (uses config based on interval)
        """
        # Set default start date based on interval
        if start_date:
            self.start_date = start_date
        elif interval == "1wk":
            self.start_date = self.DEFAULT_WEEKLY_START_DATE
        else:
            self.start_date = self.DEFAULT_START_DATE
        
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.interval = interval
        self.delay = delay
        self.limit_nums = limit_nums

        # Get paths from config
        self.us_index_path = Path(settings.us_index_path).expanduser()
        
        # Determine data directory based on interval or custom path
        if data_dir:
            self.us_stock_data_dir = Path(data_dir).expanduser()
        elif interval == "1wk":
            self.us_stock_data_dir = Path(settings.us_stock_weekly_data_dir).expanduser()
        else:
            self.us_stock_data_dir = Path(settings.us_stock_data_dir).expanduser()

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

    def _normalize_and_filter_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter out rows with timestamps in the date column.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with a 'date' column

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame with only date entries (no timestamps)
        """
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            logger.warning("Data does not contain all required columns for filtering")
            return pd.DataFrame()  # Return empty DataFrame if columns missing

        # Only filter for daily and weekly intervals
        if self.interval not in ["1d", "1wk"]:
            return data

        # Make a copy and normalize date column
        filtered_data = data.copy()

        filtered_data['date'] = filtered_data['date'].apply(normalize_datetime_to_date)

        if self.interval == "1wk":
            # For weekly data, keep only rows where time is 00:00:00 and day is before the reference Monday
            # If today is Saturday (5) or Sunday (6), use this week's Monday
            # Otherwise (Mon-Fri), use last week's Monday
            today = date.today()
            days = today.weekday()
            if days >= 5:  # Saturday or Sunday
                reference_monday = today - timedelta(days=days)  # This week's Monday
            else:  # Monday to Friday
                reference_monday = today - timedelta(days=7 + days)  # Last week's Monday
            mask = (filtered_data['date'] <= reference_monday)
            filtered_data = filtered_data[mask]

        return filtered_data

    def _get_daily_data_from_yahoo(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """Get daily stock data from Yahoo Finance using market-prices library.

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

                prices = PricesYahoo(symbol)
                # Note: market-prices uses "1D" format (uppercase) for daily interval
                data = prices.get(self.MARKET_PRICES_DAILY_INTERVAL, start=start, end=end)

                if isinstance(data, pd.DataFrame) and not data.empty:
                    # Reset index to get date as column
                    data = data.reset_index()

                    # Rename 'date' column if it exists, otherwise use index column
                    if 'date' not in data.columns and 'index' in data.columns:
                        data = data.rename(columns={'index': 'date'})

                    # Flatten multi-level columns if present
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(-1)

                    # Add symbol column
                    data['symbol'] = symbol

                    # The market-prices library returns meaningful date data,
                    # no need for normalize_datetime_to_date
                    # Ensure date column is just the date part
                    if 'date' in data.columns:
                        data['date'] = pd.to_datetime(data['date']).dt.date

                    return data
                else:
                    logger.warning(f"Empty or invalid daily data for {symbol} (attempt {attempt + 1}/{self.RETRY_COUNT})")
                    return pd.DataFrame()

            except Exception as e:
                logger.warning(f"Error fetching daily data for {symbol} (attempt {attempt + 1}/{self.RETRY_COUNT}): {e}")
                if attempt < self.RETRY_COUNT - 1:
                    time.sleep(self.delay * (attempt + 1))

        logger.error(f"Failed to fetch daily data for {symbol} after {self.RETRY_COUNT} attempts")
        return None

    def _get_weekly_data_from_yahoo(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """Get weekly stock data from Yahoo Finance using yahooquery library.

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
                data = ticker.history(interval=self.YAHOO_WEEKLY_INTERVAL, start=start, end=end)
                filtered_data = None

                if isinstance(data, pd.DataFrame) and not data.empty:
                    # Reset index to get date as column
                    data = data.reset_index()

                    # Add symbol column if not present
                    if 'symbol' not in data.columns:
                        data['symbol'] = symbol

                    filtered_data = self._normalize_and_filter_data(data)

                elif isinstance(data, dict) and symbol in data:
                    # Handle case where data is returned as dict
                    symbol_data = data[symbol]
                    if isinstance(symbol_data, pd.DataFrame) and not symbol_data.empty:
                        symbol_data = symbol_data.reset_index()
                        symbol_data['symbol'] = symbol
                        filtered_data = self._normalize_and_filter_data(symbol_data)

                if filtered_data is not None and not filtered_data.empty:
                    return filtered_data
                else:
                    logger.warning(f"Empty or invalid weekly data for {symbol} (attempt {attempt + 1}/{self.RETRY_COUNT})")
                    return pd.DataFrame()  # Return empty DataFrame if no valid data
            except Exception as e:
                logger.warning(f"Error fetching weekly data for {symbol} (attempt {attempt + 1}/{self.RETRY_COUNT}): {e}")
                if attempt < self.RETRY_COUNT - 1:
                    time.sleep(self.delay * (attempt + 1))

        logger.error(f"Failed to fetch weekly data for {symbol} after {self.RETRY_COUNT} attempts")
        return None

    def _get_data_from_yahoo(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """Get stock data from Yahoo Finance.

        Routes to the appropriate implementation based on the interval setting.
        For daily data (1d), uses the market-prices library.
        For weekly data (1wk), uses the yahooquery library (market-prices doesn't support 1wk).

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
        if self.interval == "1wk":
            return self._get_weekly_data_from_yahoo(symbol, start, end)
        elif self.interval == "1d":
            return self._get_daily_data_from_yahoo(symbol, start, end)
        else:
            # For unsupported intervals, default to daily data with a warning
            logger.warning(f"Interval '{self.interval}' not explicitly supported, defaulting to daily data collection")
            return self._get_daily_data_from_yahoo(symbol, start, end)

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
            # Calculate daily returns using vectorized operations
            df_sorted = df.sort_values('date').copy()
            
            # Use shift for previous close - more efficient
            df_sorted['prev_close'] = df_sorted['close'].shift(1)
            
            # Vectorized return calculation
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_return = np.abs(df_sorted['close'] / df_sorted['prev_close'] - 1)
            
            # Check for extreme price changes
            extreme_changes = daily_return > self.ABNORMAL_CHANGE_THRESHOLD
            if np.any(extreme_changes):
                anomaly_count = np.sum(extreme_changes)
                logger.warning(f"Detected {anomaly_count} extreme price changes for {symbol}")
                return True

            # Check for zero or negative prices using vectorized operations
            price_cols = ['close', 'open', 'high', 'low']
            invalid_prices = (df_sorted[price_cols] <= 0).any(axis=1)
            
            if invalid_prices.any():
                invalid_count = invalid_prices.sum()
                logger.warning(f"Detected {invalid_count} invalid prices for {symbol}")
                return True

            # Check for illogical OHLC relationships using vectorized operations
            illogical_ohlc = (
                (df_sorted['high'] < df_sorted['low']) |
                (df_sorted['high'] < df_sorted['open']) |
                (df_sorted['high'] < df_sorted['close']) |
                (df_sorted['low'] > df_sorted['open']) |
                (df_sorted['low'] > df_sorted['close'])
            )

            if illogical_ohlc.any():
                illogical_count = illogical_ohlc.sum()
                logger.warning(f"Detected {illogical_count} illogical OHLC relationships for {symbol}")
                return True

            return False

        except Exception as e:
            logger.warning(f"Error detecting anomalies for {symbol}: {e}")
            return False

    def _detect_overlapping_data_anomalies(self, existing_df: pd.DataFrame, new_df: pd.DataFrame, symbol: str) -> bool:
        """Detect anomalies in overlapping date data between existing and new data.

        Parameters
        ----------
        existing_df : pd.DataFrame
            Existing data
        new_df : pd.DataFrame
            New data
        symbol : str
            Stock symbol

        Returns
        -------
        bool
            True if anomalies detected in overlapping data, False otherwise
        """
        try:
            # Find overlapping dates
            existing_dates = set(existing_df['date'])
            new_dates = set(new_df['date'])
            overlapping_dates = existing_dates.intersection(new_dates)

            if not overlapping_dates:
                logger.debug(f"No overlapping dates for {symbol}")
                return False

            logger.info(f"Checking {len(overlapping_dates)} overlapping dates for {symbol}")

            # Get overlapping records
            existing_overlap = existing_df[existing_df['date'].isin(overlapping_dates)].sort_values('date')
            new_overlap = new_df[new_df['date'].isin(overlapping_dates)].sort_values('date')

            # Check for inconsistencies in overlapping data
            price_cols = ['open', 'high', 'low', 'close']
            tolerance = 0.01  # 1% tolerance for price differences

            for date in overlapping_dates:
                existing_row = existing_overlap[existing_overlap['date'] == date].iloc[0]
                new_row = new_overlap[new_overlap['date'] == date].iloc[0]

                # Compare prices with tolerance
                for col in price_cols:
                    existing_val = existing_row[col]
                    new_val = new_row[col]

                    # Skip if either value is NaN
                    if pd.isna(existing_val) or pd.isna(new_val):
                        continue

                    # Calculate relative difference
                    if existing_val != 0:
                        rel_diff = abs(new_val - existing_val) / abs(existing_val)
                        if rel_diff > tolerance:
                            logger.warning(
                                f"Price inconsistency for {symbol} on {date}: "
                                f"{col} differs by {rel_diff*100:.2f}% "
                                f"(existing: {existing_val}, new: {new_val})"
                            )
                            return True

                # Check volume consistency (larger tolerance for volume)
                if 'volume' in existing_row and 'volume' in new_row:
                    existing_vol = existing_row['volume']
                    new_vol = new_row['volume']
                    
                    if not pd.isna(existing_vol) and not pd.isna(new_vol) and existing_vol != 0:
                        vol_diff = abs(new_vol - existing_vol) / abs(existing_vol)
                        if vol_diff > 0.1:  # 10% tolerance for volume
                            logger.warning(
                                f"Volume inconsistency for {symbol} on {date}: "
                                f"differs by {vol_diff*100:.2f}% "
                                f"(existing: {existing_vol}, new: {new_vol})"
                            )
                            return True

            return False

        except Exception as e:
            logger.warning(f"Error detecting overlapping anomalies for {symbol}: {e}")
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

            # Check for anomalies in overlapping data before merging
            if self._detect_overlapping_data_anomalies(existing_df, new_df, symbol):
                logger.warning(f"Anomalies detected in overlapping data for {symbol}, marking for full re-download")
                self.abnormal_tickers.add(symbol)
                return existing_df  # Return existing data for now

            # Combine and remove duplicates, keeping new data
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
            combined_df = combined_df.sort_values('date').reset_index(drop=True)

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
            default_start = self.DEFAULT_WEEKLY_START_DATE if self.interval == "1wk" else self.DEFAULT_START_DATE
            logger.info(f"No existing data for {symbol}, downloading full history from {default_start}")
            new_df = self._get_data_from_yahoo(symbol, default_start, self.end_date)

            if new_df is not None and not new_df.empty:
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

        default_start = self.DEFAULT_WEEKLY_START_DATE if self.interval == "1wk" else self.DEFAULT_START_DATE
        for symbol in self.abnormal_tickers:
            logger.info(f"Re-downloading full data for abnormal ticker: {symbol}")
            new_df = self._get_data_from_yahoo(symbol, default_start, self.end_date)

            if new_df is not None:
                self._save_data(new_df, symbol)
                logger.info(f"Successfully re-downloaded data for {symbol}")
            else:
                logger.error(f"Failed to re-download data for {symbol}")

    def _get_batch_daily_data_from_yahoo(self, symbols: List[str], start: str, end: str) -> Optional[pd.DataFrame]:
        """Get batch daily stock data from Yahoo Finance for multiple symbols using market-prices.

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
            Combined stock data DataFrame or None if failed
        """
        if not symbols:
            return None

        for attempt in range(self.RETRY_COUNT):
            try:
                time.sleep(self.delay)

                # PricesYahoo accepts multiple symbols at once
                prices = PricesYahoo(symbols)
                # Note: market-prices uses "1D" format (uppercase) for daily interval
                data = prices.get(self.MARKET_PRICES_DAILY_INTERVAL, start=start, end=end)

                if isinstance(data, pd.DataFrame) and not data.empty:
                    # Reset index to get date as column
                    data = data.reset_index()

                    # Rename 'date' column if it exists, otherwise use index column
                    if 'date' not in data.columns and 'index' in data.columns:
                        data = data.rename(columns={'index': 'date'})

                    # For multiple symbols, market-prices returns MultiIndex columns with symbol as top level
                    # We need to reshape the data to have a 'symbol' column
                    if isinstance(data.columns, pd.MultiIndex):
                        # Stack the symbol level to convert from wide to long format
                        date_col = data.columns[0]  # First column is the date
                        data = data.set_index(date_col)
                        data = data.stack(level=0, future_stack=True).reset_index()
                        data.columns = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
                    elif len(symbols) == 1:
                        # Single symbol case - add symbol column
                        data['symbol'] = symbols[0]

                    # The market-prices library returns meaningful date data,
                    # no need for normalize_datetime_to_date
                    # Ensure date column is just the date part
                    if 'date' in data.columns:
                        data['date'] = pd.to_datetime(data['date']).dt.date

                    return data
                else:
                    logger.warning(f"Empty or invalid batch daily data for {symbols} (attempt {attempt + 1}/{self.RETRY_COUNT})")
                    return None

            except Exception as e:
                logger.warning(f"Error fetching batch daily data for {symbols} (attempt {attempt + 1}/{self.RETRY_COUNT}): {e}")
                if attempt < self.RETRY_COUNT - 1:
                    time.sleep(self.delay * (attempt + 1))

        logger.error(f"Failed to fetch batch daily data for {symbols} after {self.RETRY_COUNT} attempts")
        return None

    def _get_batch_weekly_data_from_yahoo(self, symbols: List[str], start: str, end: str) -> Optional[pd.DataFrame]:
        """Get batch weekly stock data from Yahoo Finance for multiple symbols using yahooquery.

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
                data = ticker.history(interval=self.YAHOO_WEEKLY_INTERVAL, start=start, end=end)
                filtered_data = None
                if isinstance(data, pd.DataFrame) and not data.empty:
                    # Check if we have MultiIndex (symbol, date)
                    if isinstance(data.index, pd.MultiIndex):
                        # Reset MultiIndex to get symbol and date as columns
                        data = data.reset_index()

                        # Ensure required columns exist
                        filtered_data = self._normalize_and_filter_data(data)
                    else:
                        # Single symbol case
                        data = data.reset_index()
                        if len(symbols) == 1:
                            data['symbol'] = symbols[0]
                            filtered_data = self._normalize_and_filter_data(data)

                if filtered_data is not None and not filtered_data.empty:
                    return filtered_data
                else:
                    logger.warning(f"Empty or invalid batch weekly data for {symbols} (attempt {attempt + 1}/{self.RETRY_COUNT})")
                    return None  # Return None if no valid data

            except Exception as e:
                logger.warning(f"Error fetching batch weekly data for {symbols} (attempt {attempt + 1}/{self.RETRY_COUNT}): {e}")
                if attempt < self.RETRY_COUNT - 1:
                    time.sleep(self.delay * (attempt + 1))

        logger.error(f"Failed to fetch batch weekly data for {symbols} after {self.RETRY_COUNT} attempts")
        return None

    def _get_batch_data_from_yahoo(self, symbols: List[str], start: str, end: str) -> Optional[pd.DataFrame]:
        """Get batch stock data from Yahoo Finance for multiple symbols.

        Routes to the appropriate implementation based on the interval setting.
        For daily data (1d), uses the market-prices library.
        For weekly data (1wk), uses the yahooquery library (market-prices doesn't support 1wk).

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
            Combined stock data DataFrame or None if failed
        """
        if not symbols:
            return None

        if self.interval == "1wk":
            return self._get_batch_weekly_data_from_yahoo(symbols, start, end)
        elif self.interval == "1d":
            return self._get_batch_daily_data_from_yahoo(symbols, start, end)
        else:
            # For unsupported intervals, default to daily data with a warning
            logger.warning(f"Interval '{self.interval}' not explicitly supported for batch, defaulting to daily data collection")
            return self._get_batch_daily_data_from_yahoo(symbols, start, end)

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
            elif period_days <= 60:
                return 5
            else:
                return 1

        elif self.interval == "1min":
            # Minute data requires much smaller batches due to data volume
            if period_days <= 1:
                return 10
            elif period_days <= 5:
                return 5
            elif period_days <= 10:
                return 3
            else:
                return 1
        elif self.interval == "1wk":
            if period_days <= 30:
                return 50
            elif period_days <= 90:
                return 20
            elif period_days <= 180:
                return 10
            elif period_days <= 365:
                return 5
            else:
                return 1
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
                    start_download = max(pd.Timestamp(self.start_date), pd.Timestamp(max_date))
                    end_download = pd.Timestamp(self.end_date)

                    if start_download <= end_download:
                        symbols_incremental.append(symbol)
                        if incremental_date_range is None:
                            incremental_date_range = (start_download.strftime("%Y-%m-%d"), end_download.strftime("%Y-%m-%d"))

            logger.info(f"Analysis complete - Full download: {len(symbols_full_download)}, Incremental: {len(symbols_incremental)}")

            # Phase 2: Handle full downloads (individual downloads for better error handling)
            if symbols_full_download:
                logger.info(f"Downloading full history for {len(symbols_full_download)} symbols...")
                default_start = self.DEFAULT_WEEKLY_START_DATE if self.interval == "1wk" else self.DEFAULT_START_DATE
                for i, symbol in enumerate(symbols_full_download, 1):
                    logger.info(f"Full download progress: {i}/{len(symbols_full_download)} - {symbol}")
                    new_df = self._get_data_from_yahoo(symbol, default_start, self.end_date)

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
