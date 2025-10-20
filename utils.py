"""Utility functions for qstock-collector.

Provides common helper functions used across different modules.
"""
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger
import pandas as pd


def read_last_trading_date(
    calendar_path: Path,
    default_date: str = "2015-01-01",
    offset: int = -2
) -> str:
    """Read the last trading date from a calendar file.
    
    Args:
        calendar_path: Path to the calendar file
        default_date: Default date to return if file doesn't exist or is empty
        offset: Offset from the end of file (-2 means second to last line)
        
    Returns:
        Last trading date in YYYY-MM-DD format.
        
    Raises:
        ValueError: If date format is invalid.
    """
    calendar_path = Path(calendar_path).expanduser()
    
    if not calendar_path.exists():
        logger.warning(
            f"Calendar file not found: {calendar_path}, "
            f"defaulting to {default_date}"
        )
        return default_date
    
    try:
        with calendar_path.open('r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        if not lines:
            logger.warning(
                f"Calendar file is empty: {calendar_path}, "
                f"defaulting to {default_date}"
            )
            return default_date
        
        # Get the appropriate line based on offset
        if abs(offset) > len(lines):
            last_trading_date = lines[0]
        else:
            last_trading_date = lines[offset]
        
        # Validate date format
        datetime.strptime(last_trading_date, "%Y-%m-%d")
        logger.info(f"Last trading date from calendar: {last_trading_date}")
        
        return last_trading_date
        
    except ValueError as e:
        logger.error(f"Invalid date format in calendar file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error reading calendar file {calendar_path}: {e}")
        return default_date


def count_symbols_in_index(index_path: Path) -> int:
    """Count the number of symbols in an index file.
    
    Args:
        index_path: Path to the index file (tab-separated format)
        
    Returns:
        Number of unique symbols in the index file.
    """
    index_path = Path(index_path).expanduser()
    
    if not index_path.exists():
        logger.warning(f"Index file not found: {index_path}")
        return 0
    
    try:
        with index_path.open('r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())
    except Exception as e:
        logger.error(f"Error counting symbols in {index_path}: {e}")
        return 0


def get_current_date() -> str:
    """Get current date in YYYY-MM-DD format.
    
    Returns:
        Current date string.
    """
    return datetime.now().strftime("%Y-%m-%d")


def format_duration(seconds: float) -> str:
    """Format duration in seconds to a readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string (e.g., "1h 30m 45s" or "5m 23s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    return f"{minutes}m {secs}s"


def validate_date_format(date_string: str) -> bool:
    """Validate if a string is in YYYY-MM-DD format.
    
    Args:
        date_string: Date string to validate
        
    Returns:
        True if valid, False otherwise.
    """
    try:
        datetime.strptime(date_string, "%Y-%m-%d")
        return True
    except ValueError:
        return False

def normalize_datetime_to_date(val):
    """Convert to datetime and strip timezone without changing the time value"""
    if pd.isna(val):
        return pd.NaT
    
    if isinstance(val, datetime):
        return val.date()

    # If already a date, return as is
    if isinstance(val, date):
        return val

    # Convert to Timestamp if needed
    dt = val if isinstance(val, pd.Timestamp) else pd.to_datetime(val, errors='coerce')
    
    # Return date, stripping timezone if present
    if pd.notna(dt):
        return (dt.tz_localize(None) if dt.tz else dt).date()
    return pd.NaT
