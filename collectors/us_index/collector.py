"""US Index collector module.

This module collects the latest constituents of US stock indices (SP500 and NASDAQ100)
and merges them into a single file as specified by the us_index_path configuration.
"""

import pandas as pd
import requests
from io import StringIO
from pathlib import Path
from loguru import logger
from typing import List

from config import settings

class USIndexCollector:
    """Collector for US stock index constituents (SP500 and NASDAQ100)."""

    WIKI_URL = "https://en.wikipedia.org/wiki"
    SP500_URL = f"{WIKI_URL}/List_of_S%26P_500_companies"
    NASDAQ100_URL = f"{WIKI_URL}/NASDAQ-100"

    def __init__(self):
        """Initialize the US index collector."""
        self.us_index_path = Path(settings.us_index_path).expanduser()
        self.us_index_path.parent.mkdir(parents=True, exist_ok=True)
        # Use a persistent session with a polite User-Agent and common headers
        # to avoid being blocked by Wikipedia (403). Wikipedia may block
        # requests that don't identify the client.
        self.session = requests.Session()
        self.session.headers.update({
            # Use a realistic browser User-Agent to better mimic normal browser traffic
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        })

    def get_sp500_symbols_with_dates(self) -> pd.DataFrame:
        """Get current SP500 constituents with their addition dates from Wikipedia.

        Returns:
            DataFrame with columns: symbol, date_added
        """
        logger.info("Fetching SP500 constituents with dates from Wikipedia...")

        try:
            resp = self.session.get(self.SP500_URL, timeout=30)
            resp.raise_for_status()

            # Parse HTML tables
            df_list = pd.read_html(StringIO(resp.text))

            # Find the main constituents table with Symbol and Date added columns
            for df in df_list:
                if "Symbol" in df.columns and "Date added" in df.columns:
                    # Clean the data
                    df = df[["Symbol", "Date added"]].copy()
                    df = df.dropna(subset=["Symbol"])
                    df.columns = ["symbol", "date_added"]

                    # Convert date_added to standard format
                    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")

                    # For symbols with missing dates, use a default early date
                    df["date_added"] = df["date_added"].fillna(pd.Timestamp("1999-01-01"))
                    df["date_added"] = df["date_added"].dt.strftime("%Y-%m-%d")

                    logger.info(f"Found {len(df)} SP500 symbols with dates")
                    return df

            raise ValueError("Could not find SP500 symbols table with dates")

        except Exception as e:
            logger.error(f"Failed to fetch SP500 symbols: {e}")
            raise

    def get_nasdaq100_symbols_with_dates(self) -> pd.DataFrame:
        """Get current NASDAQ100 constituents with dates from Wikipedia.

        Returns:
            DataFrame with columns: symbol, date_added
        """
        logger.info("Fetching NASDAQ100 constituents from Wikipedia...")

        try:
            resp = self.session.get(self.NASDAQ100_URL, timeout=30)
            resp.raise_for_status()

            # Parse HTML tables
            df_list = pd.read_html(StringIO(resp.text))

            # Find the table with Ticker column (NASDAQ100 uses "Ticker" instead of "Symbol")
            for df in df_list:
                if "Ticker" in df.columns and len(df) >= 100:
                    # For NASDAQ100, we may not always have date information in the main table
                    # Use a default date for all NASDAQ100 symbols
                    df_result = df[["Ticker"]].copy()
                    df_result = df_result.dropna(subset=["Ticker"])
                    df_result.columns = ["symbol"]
                    df_result["date_added"] = "2003-01-02"  # Default NASDAQ100 start date

                    logger.info(f"Found {len(df_result)} NASDAQ100 symbols")
                    return df_result

            raise ValueError("Could not find NASDAQ100 symbols table")

        except Exception as e:
            logger.error(f"Failed to fetch NASDAQ100 symbols: {e}")
            raise

    def merge_and_save_symbols(self, sp500_df: pd.DataFrame, nasdaq100_df: pd.DataFrame) -> None:
        """Merge SP500 and NASDAQ100 DataFrames and save to configured file.

        Args:
            sp500_df: DataFrame with SP500 symbols and their addition dates
            nasdaq100_df: DataFrame with NASDAQ100 symbols and their addition dates
        """
        # Combine both DataFrames
        combined_df = pd.concat([sp500_df, nasdaq100_df], ignore_index=True)

        # Replace dots with hyphens in symbols (e.g., BRK.B -> BRK-B)
        combined_df['symbol'] = combined_df['symbol'].str.replace('.', '-', regex=False)

        # Remove duplicates, keeping the first occurrence (which will have the earlier date)
        combined_df = combined_df.drop_duplicates(subset=['symbol'], keep='first')

        # Sort alphabetically by symbol
        combined_df = combined_df.sort_values('symbol').reset_index(drop=True)

        # Add end_date column
        combined_df['end_date'] = "2099-12-31"

        # Ensure column order is correct
        combined_df = combined_df[['symbol', 'date_added', 'end_date']]

        logger.info(f"Total unique symbols after merge: {len(combined_df)} (SP500: {len(sp500_df)}, NASDAQ100: {len(nasdaq100_df)})")

        # Save to file in tab-separated format: symbol \t start_date \t end_date
        combined_df.to_csv(self.us_index_path, sep='\t', header=False, index=False)

        logger.info(f"Saved merged US index symbols to: {self.us_index_path}")

    def collect(self) -> None:
        """Collect latest US index constituents and save merged result."""
        logger.info("Starting US index collection...")

        try:
            # Fetch symbols with dates from both indices
            sp500_df = self.get_sp500_symbols_with_dates()
            nasdaq100_df = self.get_nasdaq100_symbols_with_dates()

            # Merge and save
            self.merge_and_save_symbols(sp500_df, nasdaq100_df)

            logger.info("US index collection completed successfully!")

        except Exception as e:
            logger.error(f"US index collection failed: {e}")
            raise


def collect_us_index():
    """Main entry point for US index collection."""
    collector = USIndexCollector()
    collector.collect()


if __name__ == "__main__":
    collect_us_index()
