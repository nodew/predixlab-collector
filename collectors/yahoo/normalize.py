"""Yahoo Finance data normalizer for US stocks.
This normalizer processes raw stock data from Yahoo Finance, applying standardization,
anomaly detection, and adjustment calculations to prepare data for analysis.
"""

import os
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Optional, List, Iterable
from datetime import datetime
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import settings

class YahooNormalizer:
    """Yahoo Finance data normalizer for US stocks."""

    # Standard column names for normalized data
    COLUMNS = ["open", "close", "high", "low", "volume"]
    DATE_FORMAT = "%Y-%m-%d"

    # Anomaly detection thresholds
    ABNORMAL_CHANGE_MIN = 89  # Minimum abnormal change factor
    ABNORMAL_CHANGE_MAX = 111  # Maximum abnormal change factor
    MAX_CORRECTION_ITERATIONS = 10  # Max iterations for anomaly correction

    def __init__(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_workers: int = None,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        calendar_list: Optional[List] = None,
        interval: str = "1d",
        source_dir: Optional[str] = None,
        target_dir: Optional[str] = None
    ):
        """Initialize Yahoo Finance normalizer.

        Parameters
        ----------
        start_date : Optional[str]
            Start date for normalization (YYYY-MM-DD)
        end_date : Optional[str]
            End date for normalization (YYYY-MM-DD)
        max_workers : int
            Number of worker processes for parallel processing, default None (auto-detect)
        date_field_name : str
            Date field name, default "date"
        symbol_field_name : str
            Symbol field name, default "symbol"
        calendar_list : Optional[List]
            Trading calendar list for reindexing, default None
        interval : str
            Data interval, default "1d" (supports "1d", "1wk", etc.)
        source_dir : Optional[str]
            Custom source data directory path, default None (uses config based on interval)
        target_dir : Optional[str]
            Custom target data directory path, default None (uses config based on interval)
        """
        self.start_date = start_date
        self.end_date = end_date
        self.max_workers = max_workers or max(multiprocessing.cpu_count() - 1, 1)
        self.date_field_name = date_field_name
        self.symbol_field_name = symbol_field_name
        self.calendar_list = calendar_list
        self.interval = interval

        # Get paths from config or custom paths
        if source_dir:
            self.us_stock_data_dir = Path(source_dir).expanduser()
        elif interval == "1wk":
            self.us_stock_data_dir = Path(settings.us_stock_weekly_data_dir).expanduser()
        else:
            self.us_stock_data_dir = Path(settings.us_stock_data_dir).expanduser()

        if target_dir:
            self.us_normalized_data_dir = Path(target_dir).expanduser()
        elif interval == "1wk":
            self.us_normalized_data_dir = Path(settings.us_normalized_weekly_data_dir).expanduser()
        else:
            self.us_normalized_data_dir = Path(settings.us_normalized_data_dir).expanduser()

        # Ensure normalized data directory exists
        self.us_normalized_data_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Yahoo normalizer initialized")
        logger.info(f"Source directory: {self.us_stock_data_dir}")
        logger.info(f"Target directory: {self.us_normalized_data_dir}")
        logger.info(f"Max workers: {self.max_workers}")

    @staticmethod
    def calc_change(df: pd.DataFrame, last_close: Optional[float] = None) -> pd.Series:
        """Calculate daily change/return series using vectorized operations.

        Parameters
        ----------
        df : pd.DataFrame
            Stock data DataFrame with 'close' column
        last_close : Optional[float]
            Last close price from previous period

        Returns
        -------
        pd.Series
            Daily change series
        """
        close_series = df["close"].ffill()
        prev_close_series = close_series.shift(1)

        if last_close is not None:
            prev_close_series.iloc[0] = float(last_close)

        # Vectorized calculation with proper handling of division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            change_series = close_series / prev_close_series - 1
        
        return change_series

    def normalize_yahoo_data(
        self,
        df: pd.DataFrame,
        calendar_list: Optional[List] = None,
        last_close: Optional[float] = None
    ) -> pd.DataFrame:
        """Normalize Yahoo Finance data.

        Parameters
        ----------
        df : pd.DataFrame
            Raw stock data DataFrame
        calendar_list : Optional[List]
            Trading calendar for reindexing, default None
        last_close : Optional[float]
            Last close price from previous period, default None

        Returns
        -------
        pd.DataFrame
            Normalized stock data DataFrame
        """
        if df.empty:
            return df

        # Get symbol for logging
        symbol = df.loc[df[self.symbol_field_name].first_valid_index(), self.symbol_field_name]

        # Copy required columns
        columns = copy.deepcopy(self.COLUMNS)
        df_norm = df.copy()

        # Set date index and handle timezone
        df_norm.set_index(self.date_field_name, inplace=True)
        df_norm.index = pd.to_datetime(df_norm.index)
        df_norm.index = df_norm.index.tz_localize(None)

        # Remove duplicate dates, keep first
        df_norm = df_norm[~df_norm.index.duplicated(keep="first")]

        # Reindex to trading calendar if provided
        if calendar_list is not None:
            cal_df = pd.DataFrame(index=calendar_list)
            date_range = cal_df.loc[
                pd.Timestamp(df_norm.index.min()).date() :
                pd.Timestamp(df_norm.index.max()).date() + pd.Timedelta(hours=23, minutes=59)
            ].index
            df_norm = df_norm.reindex(date_range)

        # Sort by index
        df_norm.sort_index(inplace=True)

        # Set invalid volume rows to NaN
        invalid_volume_mask = (df_norm["volume"] <= 0) | np.isnan(df_norm["volume"])
        columns_to_nan = list(set(df_norm.columns) - {self.symbol_field_name})
        df_norm.loc[invalid_volume_mask, columns_to_nan] = np.nan

        # Anomaly correction loop with improved efficiency
        correction_count = 0
        while correction_count < self.MAX_CORRECTION_ITERATIONS:
            change_series = self.calc_change(df_norm, last_close)

            # Find anomalous changes using vectorized operations
            anomaly_mask = (
                (change_series >= self.ABNORMAL_CHANGE_MIN) &
                (change_series <= self.ABNORMAL_CHANGE_MAX)
            )

            if not anomaly_mask.any():
                break

            # Correct anomalous prices by dividing by 100
            price_cols = ["high", "close", "low", "open"]
            if "adjclose" in df_norm.columns:
                price_cols.append("adjclose")

            # Use vectorized division instead of loc assignment
            for col in price_cols:
                if col in df_norm.columns:
                    df_norm.loc[anomaly_mask, col] /= 100

            correction_count += 1

        if correction_count >= self.MAX_CORRECTION_ITERATIONS:
            logger.warning(
                f"{symbol}: Abnormal change detected for {correction_count} consecutive iterations. "
                f"Please check data manually."
            )

        # Calculate final change series
        df_norm["change"] = self.calc_change(df_norm, last_close)
        columns.append("change")

        # Set invalid volume rows to NaN for all calculated columns
        df_norm.loc[invalid_volume_mask, columns] = np.nan

        # Restore symbol column
        df_norm[self.symbol_field_name] = symbol
        df_norm.index.names = [self.date_field_name]

        return df_norm.reset_index()

    def adjust_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply price adjustments for splits and dividends.

        Parameters
        ----------
        df : pd.DataFrame
            Normalized stock data DataFrame

        Returns
        -------
        pd.DataFrame
            Price-adjusted DataFrame
        """
        if df.empty:
            return df

        df_adj = df.copy()
        df_adj.set_index(self.date_field_name, inplace=True)

        # Calculate adjustment factor
        if "adjclose" in df_adj.columns:
            df_adj["factor"] = df_adj["adjclose"] / df_adj["close"]
            df_adj["factor"] = df_adj["factor"].ffill()
        else:
            df_adj["factor"] = 1.0

        # Apply adjustment factor to OHLC data
        for col in self.COLUMNS:
            if col not in df_adj.columns:
                continue

            if col == "volume":
                # Adjust volume by dividing by factor
                df_adj[col] = df_adj[col] / df_adj["factor"]
            else:
                # Adjust prices by multiplying by factor
                df_adj[col] = df_adj[col] * df_adj["factor"]

        df_adj.index.names = [self.date_field_name]
        return df_adj.reset_index()

    def manual_adjust_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Manual data adjustment: normalize all fields relative to first day close.

        Parameters
        ----------
        df : pd.DataFrame
            Price-adjusted DataFrame

        Returns
        -------
        pd.DataFrame
            Manually adjusted DataFrame
        """
        if df.empty:
            return df

        df_manual = df.copy()
        df_manual.sort_values(self.date_field_name, inplace=True)
        df_manual.set_index(self.date_field_name, inplace=True)

        # Get first valid close price
        first_close = self._get_first_close(df_manual)

        # Normalize all fields relative to first close
        for col in df_manual.columns:
            if col in [self.symbol_field_name, "adjclose", "change"]:
                continue

            if col == "volume":
                # Multiply volume by first close
                df_manual[col] = df_manual[col] * first_close
            else:
                # Divide prices by first close
                df_manual[col] = df_manual[col] / first_close

        return df_manual.reset_index()

    def _get_first_close(self, df: pd.DataFrame) -> float:
        """Get first valid close price.

        Parameters
        ----------
        df : pd.DataFrame
            Stock data DataFrame with date index

        Returns
        -------
        float
            First valid close price
        """
        df_valid = df.loc[df["close"].first_valid_index():]
        first_close = df_valid["close"].iloc[0]
        return first_close

    def normalize_single_symbol(self, symbol: str) -> bool:
        """Normalize data for a single symbol.

        Parameters
        ----------
        symbol : str
            Stock symbol to normalize

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            # Input and output file paths
            input_path = self.us_stock_data_dir / f"{symbol.upper()}.csv"
            output_path = self.us_normalized_data_dir / f"{symbol.upper()}.csv"

            if not input_path.exists():
                logger.warning(f"Input file not found for {symbol}: {input_path}")
                return False

            # Read raw data
            df = pd.read_csv(input_path)

            if df.empty:
                logger.warning(f"Empty data file for {symbol}")
                return False

            # Apply date range filter if specified
            if self.start_date or self.end_date:
                df[self.date_field_name] = pd.to_datetime(df[self.date_field_name])

                if self.start_date:
                    df = df[df[self.date_field_name] >= pd.Timestamp(self.start_date)]
                if self.end_date:
                    df = df[df[self.date_field_name] <= pd.Timestamp(self.end_date)]

                if df.empty:
                    logger.warning(f"No data in date range for {symbol}")
                    return False

            # Step 1: Basic normalization
            df_norm = self.normalize_yahoo_data(df, self.calendar_list)

            # Step 2: Price adjustment
            df_adj = self.adjust_price(df_norm)

            # Step 3: Manual adjustment (normalize to first day)
            df_final = self.manual_adjust_data(df_adj)

            # Save normalized data
            df_final.to_csv(output_path, index=False)

            logger.info(f"Successfully normalized {symbol}: {len(df_final)} records")
            return True

        except Exception as e:
            logger.error(f"Error normalizing {symbol}: {e}")
            return False

    def _get_symbol_files(self) -> List[str]:
        """Get list of symbol files to process.

        Returns
        -------
        List[str]
            List of symbols (without .csv extension)
        """
        if not self.us_stock_data_dir.exists():
            logger.error(f"Source directory does not exist: {self.us_stock_data_dir}")
            return []

        csv_files = list(self.us_stock_data_dir.glob("*.csv"))
        symbols = [f.stem for f in csv_files]

        logger.info(f"Found {len(symbols)} symbol files to process")
        return symbols

    def normalize(self) -> None:
        """Main normalization method with parallel processing."""
        logger.info("Starting Yahoo Finance data normalization")

        try:
            # Get list of symbols to process
            symbols = self._get_symbol_files()

            if not symbols:
                logger.warning("No symbol files found to normalize")
                return

            logger.info(f"Normalizing {len(symbols)} symbols using {self.max_workers} workers")

            # Process symbols in parallel
            success_count = 0
            failure_count = 0

            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all jobs
                future_to_symbol = {
                    executor.submit(self.normalize_single_symbol, symbol): symbol
                    for symbol in symbols
                }

                # Process completed jobs
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        success = future.result()
                        if success:
                            success_count += 1
                        else:
                            failure_count += 1
                    except Exception as e:
                        logger.error(f"Exception processing {symbol}: {e}")
                        failure_count += 1

                    # Progress reporting
                    total_processed = success_count + failure_count
                    if total_processed % 100 == 0 or total_processed == len(symbols):
                        logger.info(f"Progress: {total_processed}/{len(symbols)} "
                                  f"(Success: {success_count}, Failed: {failure_count})")

            logger.info("Yahoo Finance data normalization completed!")
            logger.info(f"Successfully normalized: {success_count} symbols")
            logger.info(f"Failed to normalize: {failure_count} symbols")
            logger.info(f"Normalized data saved to: {self.us_normalized_data_dir}")

        except Exception as e:
            logger.error(f"Data normalization failed: {e}")
            raise

def normalize_yahoo_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_workers: Optional[int] = None,
    date_field_name: str = "date",
    symbol_field_name: str = "symbol"
) -> None:
    """Main entry point for Yahoo data normalization.

    Parameters
    ----------
    start_date : Optional[str]
        Start date for normalization (YYYY-MM-DD)
    end_date : Optional[str]
        End date for normalization (YYYY-MM-DD)
    max_workers : Optional[int]
        Number of worker processes, default None (auto-detect)
    date_field_name : str
        Date field name, default "date"
    symbol_field_name : str
        Symbol field name, default "symbol"
    """
    normalizer = YahooNormalizer(
        start_date=start_date,
        end_date=end_date,
        max_workers=max_workers,
        date_field_name=date_field_name,
        symbol_field_name=symbol_field_name
    )
    normalizer.normalize()

if __name__ == "__main__":
    import fire
    fire.Fire(normalize_yahoo_data)
