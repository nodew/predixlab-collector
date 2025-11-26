"""CN Index collector module.

This module collects the latest constituents of Chinese stock indices (CSI 300 and CSI 500)
and saves them to separate files as specified by the configuration.

CSI 300 (沪深300): Comprises the 300 largest and most liquid A-share stocks
CSI 500 (中证500): Comprises the 500 medium-cap stocks (excluding CSI 300 constituents)
"""

import pandas as pd
import requests
from pathlib import Path
from loguru import logger
from typing import Literal
import time

from config import settings


# Index configuration mapping
INDEX_CONFIG = {
    "csi300": {
        "code": "000300",
        "name": "CSI 300 (沪深300)",
        "path_attr": "csi300_index_path",
    },
    "csi500": {
        "code": "000905",
        "name": "CSI 500 (中证500)",
        "path_attr": "csi500_index_path",
    },
}


class CNIndexCollector:
    """Collector for Chinese stock index constituents (CSI 300 and CSI 500)."""

    # CSI Index API endpoint (using Eastmoney API)
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

    def __init__(self, index: Literal["csi300", "csi500"] = "csi300"):
        """Initialize the CN index collector.
        
        Args:
            index: Index to collect, either "csi300" or "csi500". Defaults to "csi300".
        """
        if index not in INDEX_CONFIG:
            raise ValueError(f"Invalid index '{index}'. Must be one of: {list(INDEX_CONFIG.keys())}")
        
        self.index = index
        self.index_config = INDEX_CONFIG[index]
        self.index_path = Path(getattr(settings, self.index_config["path_attr"])).expanduser()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use a persistent session with appropriate headers
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://quote.eastmoney.com/",
        })

    def _fetch_index_constituents(self) -> pd.DataFrame:
        """Fetch index constituents from Eastmoney API.

        Returns:
            DataFrame with columns: symbol
        """
        index_code = self.index_config["code"]
        index_name = self.index_config["name"]
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
                
                # Parse constituents - only save current active companies
                symbols = []
                for item in constituents:
                    code = item.get("f12", "")
                    if code:
                        # Determine exchange prefix based on code
                        # Codes starting with 6 are Shanghai (SH), starting with 0 or 3 are Shenzhen (SZ)
                        if code.startswith("6"):
                            symbol = f"SH{code}"
                        else:
                            symbol = f"SZ{code}"
                        symbols.append(symbol)
                
                # Create DataFrame with only symbol column (current active companies)
                df = pd.DataFrame({"symbol": symbols})
                
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

    def _save_symbols(self, df: pd.DataFrame) -> None:
        """Save index symbols to the configured file.

        Args:
            df: DataFrame with symbols
        """
        index_name = self.index_config["name"]
        
        # Sort alphabetically by symbol
        df = df.sort_values('symbol').reset_index(drop=True)

        # Save to file - one symbol per line
        with open(self.index_path, 'w', encoding='utf-8') as f:
            for symbol in df['symbol']:
                f.write(f"{symbol}\n")

        logger.info(f"Saved {len(df)} {index_name} symbols to: {self.index_path}")

    def collect(self) -> None:
        """Collect latest CN index constituents and save to file."""
        index_name = self.index_config["name"]
        logger.info(f"Starting {index_name} collection...")

        try:
            # Fetch and save symbols for the specified index
            df = self._fetch_index_constituents()
            self._save_symbols(df)

            logger.info(f"{index_name} collection completed successfully!")

        except Exception as e:
            logger.error(f"CN index collection failed: {e}")
            raise


def collect_cn_index(index: Literal["csi300", "csi500"] = "csi300"):
    """Main entry point for CN index collection.
    
    Args:
        index: Index to collect, either "csi300" or "csi500". Defaults to "csi300".
    """
    collector = CNIndexCollector(index=index)
    collector.collect()


if __name__ == "__main__":
    collect_cn_index()
