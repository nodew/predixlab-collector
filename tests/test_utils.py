"""Unit tests for utils.py module."""
import pytest
from datetime import datetime, date
from pathlib import Path
from unittest.mock import patch, mock_open
import pandas as pd
import numpy as np

from utils import (
    read_last_trading_date,
    count_symbols_in_index,
    get_current_date,
    format_duration,
    validate_date_format,
    normalize_datetime_to_date,
)


class TestReadLastTradingDate:
    """Tests for read_last_trading_date function."""

    def test_returns_default_date_when_file_not_exists(self, tmp_path):
        """Test that default date is returned when calendar file doesn't exist."""
        non_existent_path = tmp_path / "non_existent.txt"
        result = read_last_trading_date(non_existent_path, default_date="2020-01-01")
        assert result == "2020-01-01"

    def test_returns_default_date_when_file_empty(self, tmp_path):
        """Test that default date is returned when calendar file is empty."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")
        result = read_last_trading_date(empty_file, default_date="2020-01-01")
        assert result == "2020-01-01"

    def test_returns_second_to_last_date_by_default(self, tmp_path):
        """Test that second to last date is returned by default (offset=-2)."""
        calendar_file = tmp_path / "calendar.txt"
        calendar_file.write_text("2024-01-01\n2024-01-02\n2024-01-03\n2024-01-04\n")
        result = read_last_trading_date(calendar_file)
        # offset=-2 means third to last line from the end
        assert result == "2024-01-03"

    def test_returns_date_with_custom_offset(self, tmp_path):
        """Test that correct date is returned with custom offset."""
        calendar_file = tmp_path / "calendar.txt"
        calendar_file.write_text("2024-01-01\n2024-01-02\n2024-01-03\n2024-01-04\n")
        result = read_last_trading_date(calendar_file, offset=-1)
        assert result == "2024-01-04"

    def test_returns_first_date_when_offset_exceeds_lines(self, tmp_path):
        """Test that first date is returned when offset exceeds number of lines."""
        calendar_file = tmp_path / "calendar.txt"
        calendar_file.write_text("2024-01-01\n2024-01-02\n")
        result = read_last_trading_date(calendar_file, offset=-10)
        assert result == "2024-01-01"

    def test_raises_value_error_for_invalid_date_format(self, tmp_path):
        """Test that ValueError is raised for invalid date format."""
        calendar_file = tmp_path / "calendar.txt"
        calendar_file.write_text("invalid-date\n")
        with pytest.raises(ValueError):
            read_last_trading_date(calendar_file, offset=-1)

    def test_handles_whitespace_lines(self, tmp_path):
        """Test that whitespace lines are filtered out."""
        calendar_file = tmp_path / "calendar.txt"
        calendar_file.write_text("2024-01-01\n\n2024-01-02\n  \n2024-01-03\n")
        result = read_last_trading_date(calendar_file, offset=-1)
        assert result == "2024-01-03"


class TestCountSymbolsInIndex:
    """Tests for count_symbols_in_index function."""

    def test_returns_zero_when_file_not_exists(self, tmp_path):
        """Test that zero is returned when index file doesn't exist."""
        non_existent_path = tmp_path / "non_existent.txt"
        result = count_symbols_in_index(non_existent_path)
        assert result == 0

    def test_counts_symbols_correctly(self, tmp_path):
        """Test that symbols are counted correctly."""
        index_file = tmp_path / "index.txt"
        index_file.write_text("AAPL\t2015-01-01\t2099-12-31\nGOOG\t2015-01-01\t2099-12-31\nMSFT\t2015-01-01\t2099-12-31\n")
        result = count_symbols_in_index(index_file)
        assert result == 3

    def test_ignores_empty_lines(self, tmp_path):
        """Test that empty lines are ignored when counting."""
        index_file = tmp_path / "index.txt"
        index_file.write_text("AAPL\t2015-01-01\t2099-12-31\n\nGOOG\t2015-01-01\t2099-12-31\n  \n")
        result = count_symbols_in_index(index_file)
        assert result == 2

    def test_returns_zero_for_empty_file(self, tmp_path):
        """Test that zero is returned for empty file."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")
        result = count_symbols_in_index(empty_file)
        assert result == 0


class TestGetCurrentDate:
    """Tests for get_current_date function."""

    def test_returns_valid_date_format(self):
        """Test that current date is returned in YYYY-MM-DD format."""
        result = get_current_date()
        # Validate format
        assert len(result) == 10
        assert result[4] == "-"
        assert result[7] == "-"
        # Validate it's a valid date
        datetime.strptime(result, "%Y-%m-%d")

    def test_returns_today_date(self):
        """Test that the function returns today's date."""
        result = get_current_date()
        expected = datetime.now().strftime("%Y-%m-%d")
        assert result == expected


class TestFormatDuration:
    """Tests for format_duration function."""

    def test_formats_seconds_only(self):
        """Test formatting when duration is less than a minute."""
        result = format_duration(45)
        assert result == "0m 45s"

    def test_formats_minutes_and_seconds(self):
        """Test formatting when duration is minutes and seconds."""
        result = format_duration(125)  # 2 minutes 5 seconds
        assert result == "2m 5s"

    def test_formats_hours_minutes_and_seconds(self):
        """Test formatting when duration includes hours."""
        result = format_duration(3665)  # 1 hour 1 minute 5 seconds
        assert result == "1h 1m 5s"

    def test_handles_zero_duration(self):
        """Test formatting zero duration."""
        result = format_duration(0)
        assert result == "0m 0s"

    def test_handles_fractional_seconds(self):
        """Test that fractional seconds are truncated."""
        result = format_duration(65.7)
        assert result == "1m 5s"


class TestValidateDateFormat:
    """Tests for validate_date_format function."""

    def test_valid_date_format(self):
        """Test that valid date format returns True."""
        assert validate_date_format("2024-01-15") is True
        assert validate_date_format("2020-12-31") is True

    def test_invalid_date_format(self):
        """Test that invalid date format returns False."""
        assert validate_date_format("01-15-2024") is False
        assert validate_date_format("2024/01/15") is False
        assert validate_date_format("invalid") is False
        assert validate_date_format("") is False
        assert validate_date_format("20240115") is False

    def test_invalid_date_values(self):
        """Test that invalid date values return False."""
        assert validate_date_format("2024-13-01") is False  # Invalid month
        assert validate_date_format("2024-02-30") is False  # Invalid day


class TestNormalizeDatetimeToDate:
    """Tests for normalize_datetime_to_date function."""

    def test_converts_datetime_to_date(self):
        """Test conversion of datetime to date."""
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = normalize_datetime_to_date(dt)
        assert result == date(2024, 1, 15)

    def test_returns_date_unchanged(self):
        """Test that date object is returned unchanged."""
        d = date(2024, 1, 15)
        result = normalize_datetime_to_date(d)
        assert result == date(2024, 1, 15)

    def test_converts_timestamp_to_date(self):
        """Test conversion of pandas Timestamp to date."""
        ts = pd.Timestamp("2024-01-15 10:30:00")
        result = normalize_datetime_to_date(ts)
        assert result == date(2024, 1, 15)

    def test_converts_timezone_aware_timestamp(self):
        """Test conversion of timezone-aware timestamp."""
        ts = pd.Timestamp("2024-01-15 10:30:00", tz="UTC")
        result = normalize_datetime_to_date(ts)
        assert result == date(2024, 1, 15)

    def test_handles_nan_value(self):
        """Test that NaN returns NaT."""
        result = normalize_datetime_to_date(pd.NaT)
        assert pd.isna(result)

    def test_handles_none_value(self):
        """Test that None-like values return NaT."""
        result = normalize_datetime_to_date(np.nan)
        assert pd.isna(result)

    def test_converts_string_to_date(self):
        """Test conversion of date string to date."""
        result = normalize_datetime_to_date("2024-01-15")
        assert result == date(2024, 1, 15)
