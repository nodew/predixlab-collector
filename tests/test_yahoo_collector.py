"""Unit tests for collectors/yahoo/collector.py module."""
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, datetime

from collectors.yahoo.collector import YahooCollector, collect_yahoo_data


class TestYahooCollector:
    """Tests for YahooCollector class."""

    @pytest.fixture
    def collector(self, tmp_path):
        """Create a YahooCollector with temporary directories."""
        with patch('collectors.yahoo.collector.settings') as mock_settings:
            mock_settings.us_index_path = str(tmp_path / "us_index.txt")
            mock_settings.us_stock_data_dir = str(tmp_path / "stock_data")
            mock_settings.us_stock_weekly_data_dir = str(tmp_path / "stock_weekly_data")
            
            # Create index file
            index_path = tmp_path / "us_index.txt"
            index_path.write_text("AAPL\t2015-01-01\t2099-12-31\nMSFT\t2015-01-01\t2099-12-31\n")
            
            collector = YahooCollector(
                start_date="2024-01-01",
                end_date="2024-01-31",
                interval="1d",
                delay=0.01,  # Short delay for testing
                limit_nums=None
            )
            return collector

    def test_init_creates_data_directory(self, tmp_path):
        """Test that __init__ creates the data directory."""
        with patch('collectors.yahoo.collector.settings') as mock_settings:
            mock_settings.us_index_path = str(tmp_path / "us_index.txt")
            mock_settings.us_stock_data_dir = str(tmp_path / "new_data_dir")
            mock_settings.us_stock_weekly_data_dir = str(tmp_path / "weekly_data_dir")
            
            (tmp_path / "us_index.txt").write_text("AAPL\t2015-01-01\t2099-12-31\n")
            
            collector = YahooCollector()
            
            assert (tmp_path / "new_data_dir").exists()

    def test_init_weekly_interval_uses_correct_dates(self, tmp_path):
        """Test that weekly interval uses correct default start date."""
        with patch('collectors.yahoo.collector.settings') as mock_settings:
            mock_settings.us_index_path = str(tmp_path / "us_index.txt")
            mock_settings.us_stock_data_dir = str(tmp_path / "stock_data")
            mock_settings.us_stock_weekly_data_dir = str(tmp_path / "stock_weekly_data")
            
            (tmp_path / "us_index.txt").write_text("AAPL\t2015-01-01\t2099-12-31\n")
            
            collector = YahooCollector(interval="1wk")
            
            assert collector.start_date == YahooCollector.DEFAULT_WEEKLY_START_DATE
            assert str(collector.us_stock_data_dir) == str(tmp_path / "stock_weekly_data")

    def test_load_stock_symbols(self, collector):
        """Test loading stock symbols from index file."""
        symbols = collector._load_stock_symbols()
        
        assert len(symbols) == 2
        assert 'AAPL' in symbols
        assert 'MSFT' in symbols

    def test_load_stock_symbols_with_limit(self, tmp_path):
        """Test loading stock symbols with limit_nums."""
        with patch('collectors.yahoo.collector.settings') as mock_settings:
            mock_settings.us_index_path = str(tmp_path / "us_index.txt")
            mock_settings.us_stock_data_dir = str(tmp_path / "stock_data")
            mock_settings.us_stock_weekly_data_dir = str(tmp_path / "stock_weekly_data")
            
            index_path = tmp_path / "us_index.txt"
            index_path.write_text("AAPL\t2015-01-01\t2099-12-31\nMSFT\t2015-01-01\t2099-12-31\nGOOG\t2015-01-01\t2099-12-31\n")
            
            collector = YahooCollector(limit_nums=2)
            symbols = collector._load_stock_symbols()
            
            assert len(symbols) == 2

    def test_load_stock_symbols_file_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised when index file missing."""
        with patch('collectors.yahoo.collector.settings') as mock_settings:
            mock_settings.us_index_path = str(tmp_path / "missing.txt")
            mock_settings.us_stock_data_dir = str(tmp_path / "stock_data")
            mock_settings.us_stock_weekly_data_dir = str(tmp_path / "stock_weekly_data")
            
            collector = YahooCollector()
            
            with pytest.raises(FileNotFoundError):
                collector._load_stock_symbols()

    def test_normalize_and_filter_data(self, collector):
        """Test normalizing and filtering data."""
        data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5, freq='B'),
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [102, 103, 104, 105, 106],
            'volume': [1000, 1100, 1200, 1300, 1400],
            'symbol': ['AAPL'] * 5
        })
        
        result = collector._normalize_and_filter_data(data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert 'date' in result.columns

    def test_normalize_and_filter_data_returns_empty_for_missing_columns(self, collector):
        """Test that empty DataFrame is returned when required columns missing."""
        data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5, freq='B'),
            'price': [100, 101, 102, 103, 104],  # Missing required columns
        })
        
        result = collector._normalize_and_filter_data(data)
        
        assert result.empty

    def test_get_existing_data_info_no_file(self, collector):
        """Test _get_existing_data_info returns None for missing file."""
        existing_df, min_date, max_date = collector._get_existing_data_info("UNKNOWN")
        
        assert existing_df is None
        assert min_date is None
        assert max_date is None

    def test_get_existing_data_info_with_file(self, collector, tmp_path):
        """Test _get_existing_data_info reads existing file."""
        # Create a mock data file
        data_file = collector.us_stock_data_dir / "AAPL.csv"
        data_file.parent.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame({
            'date': ['2024-01-02', '2024-01-03', '2024-01-04'],
            'close': [100, 101, 102],
            'symbol': ['AAPL'] * 3
        })
        df.to_csv(data_file, index=False)
        
        existing_df, min_date, max_date = collector._get_existing_data_info("AAPL")
        
        assert existing_df is not None
        assert len(existing_df) == 3
        assert min_date == "2024-01-02"
        assert max_date == "2024-01-04"

    def test_detect_data_anomalies_no_anomalies(self, collector):
        """Test anomaly detection with clean data."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5, freq='B'),
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [102, 103, 104, 105, 106],
        })
        
        result = collector._detect_data_anomalies(df, "AAPL")
        
        assert result is False

    def test_detect_data_anomalies_extreme_price_change(self, collector):
        """Test anomaly detection with extreme price change."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3, freq='B'),
            'open': [100, 101, 200],  # 100% jump
            'high': [105, 106, 210],
            'low': [95, 96, 190],
            'close': [102, 103, 200],  # Extreme change
        })
        
        result = collector._detect_data_anomalies(df, "AAPL")
        
        assert result is True

    def test_detect_data_anomalies_negative_prices(self, collector):
        """Test anomaly detection with negative prices."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3, freq='B'),
            'open': [100, 101, -102],  # Negative price
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
        })
        
        result = collector._detect_data_anomalies(df, "AAPL")
        
        assert result is True

    def test_detect_data_anomalies_illogical_ohlc(self, collector):
        """Test anomaly detection with illogical OHLC relationships."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3, freq='B'),
            'open': [100, 101, 102],
            'high': [105, 90, 107],  # High < Low
            'low': [95, 96, 97],
            'close': [102, 103, 104],
        })
        
        result = collector._detect_data_anomalies(df, "AAPL")
        
        assert result is True

    def test_save_data(self, collector):
        """Test saving data to CSV file."""
        df = pd.DataFrame({
            'date': ['2024-01-02', '2024-01-03', '2024-01-04'],
            'close': [100, 101, 102],
        })
        
        collector._save_data(df, "TEST")
        
        saved_file = collector.us_stock_data_dir / "TEST.csv"
        assert saved_file.exists()
        
        saved_df = pd.read_csv(saved_file)
        assert len(saved_df) == 3
        assert 'symbol' in saved_df.columns

    def test_save_data_empty_dataframe(self, collector):
        """Test that empty DataFrame is not saved."""
        df = pd.DataFrame()
        
        collector._save_data(df, "EMPTY")
        
        saved_file = collector.us_stock_data_dir / "EMPTY.csv"
        assert not saved_file.exists()

    def test_merge_data(self, collector):
        """Test merging existing and new data."""
        existing_df = pd.DataFrame({
            'date': ['2024-01-02', '2024-01-03'],
            'close': [100, 101],
            'open': [99, 100],
            'high': [102, 103],
            'low': [98, 99],
            'volume': [1000, 1100],
        })
        
        new_df = pd.DataFrame({
            'date': ['2024-01-04', '2024-01-05'],
            'close': [102, 103],
            'open': [101, 102],
            'high': [104, 105],
            'low': [100, 101],
            'volume': [1200, 1300],
        })
        
        result = collector._merge_data(existing_df, new_df, "AAPL")
        
        assert len(result) == 4
        assert result['date'].min() == pd.Timestamp('2024-01-02')
        assert result['date'].max() == pd.Timestamp('2024-01-05')

    def test_merge_data_removes_duplicates(self, collector):
        """Test that merge removes duplicate dates, keeping new data."""
        existing_df = pd.DataFrame({
            'date': ['2024-01-02', '2024-01-03'],
            'close': [100, 101],
            'open': [99, 100],
            'high': [102, 103],
            'low': [98, 99],
            'volume': [1000, 1100],
        })
        
        new_df = pd.DataFrame({
            'date': ['2024-01-03', '2024-01-04'],  # 2024-01-03 is duplicate
            'close': [101, 102],  # Same close for 2024-01-03 (no anomaly)
            'open': [100, 101],
            'high': [103, 104],
            'low': [99, 100],
            'volume': [1100, 1200],  # Same volume for 2024-01-03 (no anomaly)
        })
        
        result = collector._merge_data(existing_df, new_df, "AAPL")
        
        assert len(result) == 3
        # Verify all dates are present
        dates = result['date'].tolist()
        assert pd.Timestamp('2024-01-02') in dates
        assert pd.Timestamp('2024-01-03') in dates
        assert pd.Timestamp('2024-01-04') in dates

    def test_get_dynamic_batch_size_daily(self, tmp_path):
        """Test dynamic batch size calculation for daily interval."""
        with patch('collectors.yahoo.collector.settings') as mock_settings:
            mock_settings.us_index_path = str(tmp_path / "us_index.txt")
            mock_settings.us_stock_data_dir = str(tmp_path / "stock_data")
            mock_settings.us_stock_weekly_data_dir = str(tmp_path / "stock_weekly_data")
            
            (tmp_path / "us_index.txt").write_text("AAPL\t2015-01-01\t2099-12-31\n")
            
            collector = YahooCollector(interval="1d")
            
            # Short period should have larger batch size
            assert collector._get_dynamic_batch_size("2024-01-01", "2024-01-05") == 50
            
            # Longer period should have smaller batch size
            assert collector._get_dynamic_batch_size("2024-01-01", "2024-03-01") == 5

    def test_get_dynamic_batch_size_weekly(self, tmp_path):
        """Test dynamic batch size calculation for weekly interval."""
        with patch('collectors.yahoo.collector.settings') as mock_settings:
            mock_settings.us_index_path = str(tmp_path / "us_index.txt")
            mock_settings.us_stock_data_dir = str(tmp_path / "stock_data")
            mock_settings.us_stock_weekly_data_dir = str(tmp_path / "stock_weekly_data")
            
            (tmp_path / "us_index.txt").write_text("AAPL\t2015-01-01\t2099-12-31\n")
            
            collector = YahooCollector(interval="1wk")
            
            # Short period should have larger batch size
            assert collector._get_dynamic_batch_size("2024-01-01", "2024-01-30") == 50
            
            # Longer period should have smaller batch size
            assert collector._get_dynamic_batch_size("2024-01-01", "2024-06-01") == 10


class TestCollectYahooData:
    """Tests for collect_yahoo_data function."""

    def test_collect_yahoo_data_creates_collector(self, tmp_path):
        """Test that collect_yahoo_data creates collector and calls collect."""
        with patch('collectors.yahoo.collector.settings') as mock_settings, \
             patch.object(YahooCollector, 'collect') as mock_collect:
            
            mock_settings.us_index_path = str(tmp_path / "us_index.txt")
            mock_settings.us_stock_data_dir = str(tmp_path / "stock_data")
            mock_settings.us_stock_weekly_data_dir = str(tmp_path / "stock_weekly_data")
            
            (tmp_path / "us_index.txt").write_text("AAPL\t2015-01-01\t2099-12-31\n")
            
            collect_yahoo_data(
                start_date="2024-01-01",
                end_date="2024-01-31",
                interval="1d",
                delay=0.5,
                limit_nums=10
            )
            
            mock_collect.assert_called_once()
