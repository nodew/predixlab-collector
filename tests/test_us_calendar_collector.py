"""Unit tests for collectors/us_calendar/collector.py module."""
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import date

from collectors.us_calendar.collector import USCalendarCollector, collect_us_calendar


class TestUSCalendarCollector:
    """Tests for USCalendarCollector class."""

    @pytest.fixture
    def collector(self, tmp_path):
        """Create a USCalendarCollector with a temporary output path."""
        with patch("collectors.us_calendar.collector.settings") as mock_settings:
            mock_settings.us_calendar_path = str(tmp_path / "us_calendar.txt")
            mock_settings.us_weekly_calendar_path = str(tmp_path / "us_weekly_calendar.txt")
            return USCalendarCollector()

    def test_init_daily_interval(self, tmp_path):
        """Test initialization with daily interval uses correct paths."""
        with patch("collectors.us_calendar.collector.settings") as mock_settings:
            mock_settings.us_calendar_path = str(tmp_path / "us.txt")
            mock_settings.us_weekly_calendar_path = str(tmp_path / "us_weekly.txt")

            collector = USCalendarCollector(interval="1d")

            assert collector.start_date == USCalendarCollector.DEFAULT_START_DATE
            assert collector.interval == "1d"
            assert str(collector.us_calendar_path) == str(tmp_path / "us.txt")

    def test_init_weekly_interval(self, tmp_path):
        """Test initialization with weekly interval uses correct paths."""
        with patch("collectors.us_calendar.collector.settings") as mock_settings:
            mock_settings.us_calendar_path = str(tmp_path / "us.txt")
            mock_settings.us_weekly_calendar_path = str(tmp_path / "us_weekly.txt")

            collector = USCalendarCollector(interval="1wk")

            assert collector.start_date == USCalendarCollector.DEFAULT_WEEKLY_START_DATE
            assert collector.interval == "1wk"
            assert str(collector.us_calendar_path) == str(tmp_path / "us_weekly.txt")

    def test_init_custom_start_date(self, tmp_path):
        """Test initialization with custom start date."""
        with patch("collectors.us_calendar.collector.settings") as mock_settings:
            mock_settings.us_calendar_path = str(tmp_path / "us.txt")

            collector = USCalendarCollector(start_date="2020-01-01", interval="1d")

            assert collector.start_date == "2020-01-01"

    def test_get_us_trading_dates(self, tmp_path):
        """Test fetching US trading dates from Yahoo Finance."""
        with patch("collectors.us_calendar.collector.settings") as mock_settings, \
             patch("collectors.us_calendar.collector.PricesYahoo") as mock_prices_class:

            mock_settings.us_calendar_path = str(tmp_path / "us.txt")

            # Create mock historical data
            dates = pd.date_range("2024-01-01", periods=10, freq="B")
            mock_hist = pd.DataFrame({"open": [100] * len(dates)}, index=dates)

            mock_prices = MagicMock()
            mock_prices.get.return_value = mock_hist
            mock_prices_class.return_value = mock_prices

            collector = USCalendarCollector(start_date="2024-01-01", interval="1d")

            result = collector.get_us_trading_dates()

            assert len(result) > 0
            assert all(isinstance(d, date) for d in result)

    def test_get_us_trading_dates_retry_on_failure(self, tmp_path):
        """Test that get_us_trading_dates retries on failure."""
        with patch("collectors.us_calendar.collector.settings") as mock_settings, \
             patch("collectors.us_calendar.collector.PricesYahoo") as mock_prices_class, \
             patch("collectors.us_calendar.collector.time.sleep"):

            mock_settings.us_calendar_path = str(tmp_path / "us.txt")

            # First call fails, second succeeds
            dates = pd.date_range("2024-01-01", periods=5, freq="B")
            mock_hist = pd.DataFrame({"open": [100] * len(dates)}, index=dates)

            mock_prices = MagicMock()
            mock_prices.get.side_effect = [Exception("Network error"), mock_hist]
            mock_prices_class.return_value = mock_prices

            collector = USCalendarCollector(start_date="2024-01-01", interval="1d")

            result = collector.get_us_trading_dates()

            assert len(result) > 0
            assert mock_prices.get.call_count == 2

    def test_save_calendar_creates_new_file(self, tmp_path):
        """Test saving calendar to a new file."""
        with patch("collectors.us_calendar.collector.settings") as mock_settings:
            calendar_path = tmp_path / "us.txt"
            mock_settings.us_calendar_path = str(calendar_path)

            collector = USCalendarCollector(interval="1d")

            trading_dates = [
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 4),
            ]

            collector.save_calendar(trading_dates)

            assert calendar_path.exists()
            content = calendar_path.read_text()
            lines = [line for line in content.split("\n") if line.strip()]

            assert len(lines) == 3
            assert "2024-01-02" in content
            assert "2024-01-03" in content
            assert "2024-01-04" in content

    def test_save_calendar_merges_with_existing(self, tmp_path):
        """Test saving calendar merges with existing dates."""
        with patch("collectors.us_calendar.collector.settings") as mock_settings:
            calendar_path = tmp_path / "us.txt"
            mock_settings.us_calendar_path = str(calendar_path)

            # Create existing calendar file
            calendar_path.write_text("2024-01-02\n2024-01-03\n")

            collector = USCalendarCollector(interval="1d")

            # New dates overlap with existing
            trading_dates = [
                date(2024, 1, 3),  # Duplicate
                date(2024, 1, 4),  # New
                date(2024, 1, 5),  # New
            ]

            collector.save_calendar(trading_dates)

            content = calendar_path.read_text()
            lines = [line for line in content.split("\n") if line.strip()]

            # Should have 4 unique dates
            assert len(lines) == 4
            assert sorted(lines) == ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]

    def test_save_calendar_sorts_dates(self, tmp_path):
        """Test that saved calendar dates are sorted."""
        with patch("collectors.us_calendar.collector.settings") as mock_settings:
            calendar_path = tmp_path / "us.txt"
            mock_settings.us_calendar_path = str(calendar_path)

            collector = USCalendarCollector(interval="1d")

            # Unsorted dates
            trading_dates = [
                date(2024, 1, 5),
                date(2024, 1, 2),
                date(2024, 1, 4),
                date(2024, 1, 3),
            ]

            collector.save_calendar(trading_dates)

            lines = calendar_path.read_text().strip().split("\n")

            assert lines == ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]

    def test_collect_integrates_all_methods(self, tmp_path):
        """Test that collect method integrates all steps."""
        with patch("collectors.us_calendar.collector.settings") as mock_settings, \
             patch("collectors.us_calendar.collector.PricesYahoo") as mock_prices_class:

            calendar_path = tmp_path / "us.txt"
            mock_settings.us_calendar_path = str(calendar_path)

            # Create mock historical data
            dates = pd.date_range("2024-01-01", periods=5, freq="B")
            mock_hist = pd.DataFrame({"open": [100] * len(dates)}, index=dates)

            mock_prices = MagicMock()
            mock_prices.get.return_value = mock_hist
            mock_prices_class.return_value = mock_prices

            collector = USCalendarCollector(start_date="2024-01-01", interval="1d")
            collector.collect()

            assert calendar_path.exists()
            lines = [line for line in calendar_path.read_text().split("\n") if line.strip()]
            assert len(lines) > 0

    def test_collect_raises_when_no_dates(self, tmp_path):
        """Test that collect raises exception when no dates collected."""
        with patch("collectors.us_calendar.collector.settings") as mock_settings, \
             patch("collectors.us_calendar.collector.PricesYahoo") as mock_prices_class:

            mock_settings.us_calendar_path = str(tmp_path / "us.txt")

            # Return empty data
            mock_prices = MagicMock()
            mock_prices.get.return_value = pd.DataFrame()
            mock_prices_class.return_value = mock_prices

            collector = USCalendarCollector(start_date="2024-01-01", interval="1d")

            with pytest.raises(ValueError, match="No data received"):
                collector.collect()


class TestCollectUSCalendar:
    """Tests for collect_us_calendar function."""

    def test_collect_us_calendar_creates_collector(self, tmp_path):
        """Test that collect_us_calendar creates collector and calls collect."""
        with patch("collectors.us_calendar.collector.settings") as mock_settings, \
             patch.object(USCalendarCollector, "collect") as mock_collect:

            mock_settings.us_calendar_path = str(tmp_path / "us.txt")
            mock_settings.us_weekly_calendar_path = str(tmp_path / "us_weekly.txt")

            collect_us_calendar(start_date="2024-01-01", interval="1d")

            mock_collect.assert_called_once()

    def test_collect_us_calendar_weekly(self, tmp_path):
        """Test collect_us_calendar with weekly interval."""
        with patch("collectors.us_calendar.collector.settings") as mock_settings, \
             patch.object(USCalendarCollector, "collect") as mock_collect, \
             patch.object(USCalendarCollector, "__init__", return_value=None) as mock_init:

            mock_settings.us_calendar_path = str(tmp_path / "us.txt")
            mock_settings.us_weekly_calendar_path = str(tmp_path / "us_weekly.txt")

            collect_us_calendar(interval="1wk")

            # Verify the collector was initialized with weekly interval
            mock_init.assert_called_once_with(start_date=None, interval="1wk")
            mock_collect.assert_called_once()
