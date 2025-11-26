"""CN Index collector module.

This module collects the latest constituents of Chinese stock indices (CSI 300 and CSI 500)
and merges them into a single file as specified by the cn_index_path configuration.

CSI 300 (沪深300): Comprises the 300 largest and most liquid A-share stocks
CSI 500 (中证500): Comprises the 500 medium-cap stocks (excluding CSI 300 constituents)
"""

import pandas as pd
import requests
from pathlib import Path
from loguru import logger
from typing import List
import time

from config import settings


class CNIndexCollector:
    """Collector for Chinese stock index constituents (CSI 300 and CSI 500)."""

    # CSI Index API endpoint (using Eastmoney API)
    # The same endpoint is used for both indices, differentiated by the 'fs' parameter
    # CSI 300 index code: 000300
    # CSI 500 index code: 000905
    API_URL = "https://push2.eastmoney.com/api/qt/clist/get"
    
    # Default parameters for API requests
    DEFAULT_PARAMS = {
        "pn": 1,  # page number
        "pz": 1000,  # page size
        "po": 1,  # order
        "np": 1,
        "fltt": 2,
        "invt": 2,
        "fid": "f3",  # sort field
        "fs": "",  # filter string - will be set per index
        "fields": "f12,f14",  # fields: code, name
    }

    def __init__(self):
        """Initialize the CN index collector."""
        self.cn_index_path = Path(settings.cn_index_path).expanduser()
        self.cn_index_path.parent.mkdir(parents=True, exist_ok=True)
        # Use a persistent session with appropriate headers
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://quote.eastmoney.com/",
        })

    def _fetch_index_constituents(self, index_code: str, index_name: str) -> pd.DataFrame:
        """Fetch index constituents from Eastmoney API.

        Args:
            index_code: Index code (e.g., "000300" for CSI 300)
            index_name: Index name for logging

        Returns:
            DataFrame with columns: symbol, date_added
        """
        logger.info(f"Fetching {index_name} (code: {index_code}) constituents...")

        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries + 1):
            try:
                params = self.DEFAULT_PARAMS.copy()
                # fs parameter format: b:index_code for index constituents
                params["fs"] = f"b:{index_code}"
                
                resp = self.session.get(self.API_URL, params=params, timeout=30)
                resp.raise_for_status()
                
                data = resp.json()
                
                if data.get("data") is None or data["data"].get("diff") is None:
                    raise ValueError(f"No data received for {index_name}")
                
                constituents = data["data"]["diff"]
                
                if not constituents:
                    raise ValueError(f"Empty constituents list for {index_name}")
                
                # Parse constituents
                symbols = []
                for item in constituents:
                    code = item.get("f12", "")
                    if code:
                        # Determine exchange suffix based on code
                        # Codes starting with 6 are Shanghai (SS), starting with 0 or 3 are Shenzhen (SZ)
                        if code.startswith("6"):
                            symbol = f"{code}.SS"
                        else:
                            symbol = f"{code}.SZ"
                        symbols.append(symbol)
                
                # Create DataFrame
                df = pd.DataFrame({
                    "symbol": symbols,
                    "date_added": "2005-01-01"  # Default date for Chinese indices
                })
                
                logger.info(f"Found {len(df)} {index_name} constituents")
                return df

            except Exception as e:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed to fetch {index_name}: {e}")
                    logger.info(f"Retrying in {delay:.1f} seconds... ({attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    logger.error(f"Failed to fetch {index_name} after {max_retries + 1} attempts: {e}")
                    raise

    def get_csi300_symbols_with_dates(self) -> pd.DataFrame:
        """Get current CSI 300 (沪深300) constituents with their addition dates.

        Returns:
            DataFrame with columns: symbol, date_added
        """
        return self._fetch_index_constituents("000300", "CSI 300 (沪深300)")

    def get_csi500_symbols_with_dates(self) -> pd.DataFrame:
        """Get current CSI 500 (中证500) constituents with their addition dates.

        Returns:
            DataFrame with columns: symbol, date_added
        """
        return self._fetch_index_constituents("000905", "CSI 500 (中证500)")

    def merge_and_save_symbols(self, csi300_df: pd.DataFrame, csi500_df: pd.DataFrame) -> None:
        """Merge CSI 300 and CSI 500 DataFrames and save to configured file.

        Args:
            csi300_df: DataFrame with CSI 300 symbols and their addition dates
            csi500_df: DataFrame with CSI 500 symbols and their addition dates
        """
        # Combine both DataFrames
        combined_df = pd.concat([csi300_df, csi500_df], ignore_index=True)

        # Remove duplicates, keeping the first occurrence
        combined_df = combined_df.drop_duplicates(subset=['symbol'], keep='first')

        # Sort alphabetically by symbol
        combined_df = combined_df.sort_values('symbol').reset_index(drop=True)

        # Add end_date column
        combined_df['end_date'] = "2099-12-31"

        # Ensure column order is correct
        combined_df = combined_df[['symbol', 'date_added', 'end_date']]

        logger.info(f"Total unique symbols after merge: {len(combined_df)} (CSI300: {len(csi300_df)}, CSI500: {len(csi500_df)})")

        # Save to file in tab-separated format: symbol \t start_date \t end_date
        combined_df.to_csv(self.cn_index_path, sep='\t', header=False, index=False)

        logger.info(f"Saved merged CN index symbols to: {self.cn_index_path}")

    def collect(self) -> None:
        """Collect latest CN index constituents and save merged result."""
        logger.info("Starting CN index collection...")

        try:
            # Fetch symbols with dates from both indices
            csi300_df = self.get_csi300_symbols_with_dates()
            csi500_df = self.get_csi500_symbols_with_dates()

            # Merge and save
            self.merge_and_save_symbols(csi300_df, csi500_df)

            logger.info("CN index collection completed successfully!")

        except Exception as e:
            logger.error(f"CN index collection failed: {e}")
            raise


def collect_cn_index():
    """Main entry point for CN index collection."""
    collector = CNIndexCollector()
    collector.collect()


if __name__ == "__main__":
    collect_cn_index()
