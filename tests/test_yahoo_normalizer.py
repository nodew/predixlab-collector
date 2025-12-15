"""Unit tests for collectors/yahoo/normalize.py module."""
import pytest
from unittest.mock import patch
import pandas as pd
import numpy as np

from collectors.yahoo.normalize import YahooNormalizer, normalize_yahoo_data


class TestYahooNormalizer:
    """Tests for YahooNormalizer class."""

    @pytest.fixture
    def normalizer(self, tmp_path):
        """Create a YahooNormalizer with temporary directories."""
        with patch("collectors.yahoo.normalize.settings") as mock_settings:
            mock_settings.us_stock_data_dir = str(tmp_path / "stock_data")
            mock_settings.us_normalized_data_dir = str(tmp_path / "normalized_data")
            mock_settings.us_stock_weekly_data_dir = str(tmp_path / "stock_weekly_data")
            mock_settings.us_normalized_weekly_data_dir = str(tmp_path / "normalized_weekly_data")

            # Create source directory with sample data
            (tmp_path / "stock_data").mkdir(parents=True)

            return YahooNormalizer(interval="1d")

    def test_init_creates_target_directory(self, tmp_path):
        """Test that __init__ creates the target directory."""
        with patch("collectors.yahoo.normalize.settings") as mock_settings:
            mock_settings.us_stock_data_dir = str(tmp_path / "stock_data")
            mock_settings.us_normalized_data_dir = str(tmp_path / "new_normalized_dir")
            mock_settings.us_stock_weekly_data_dir = str(tmp_path / "stock_weekly_data")
            mock_settings.us_normalized_weekly_data_dir = str(tmp_path / "normalized_weekly_data")

            _ = YahooNormalizer(interval="1d")

            assert (tmp_path / "new_normalized_dir").exists()

    def test_init_weekly_interval_uses_correct_paths(self, tmp_path):
        """Test that weekly interval uses correct directory paths."""
        with patch("collectors.yahoo.normalize.settings") as mock_settings:
            mock_settings.us_stock_data_dir = str(tmp_path / "stock_data")
            mock_settings.us_normalized_data_dir = str(tmp_path / "normalized_data")
            mock_settings.us_stock_weekly_data_dir = str(tmp_path / "stock_weekly_data")
            mock_settings.us_normalized_weekly_data_dir = str(tmp_path / "normalized_weekly_data")

            normalizer = YahooNormalizer(interval="1wk")

            assert str(normalizer.us_stock_data_dir) == str(tmp_path / "stock_weekly_data")
            assert str(normalizer.us_normalized_data_dir) == str(tmp_path / "normalized_weekly_data")

    def test_init_custom_source_target_dirs(self, tmp_path):
        """Test initialization with custom source and target directories."""
        with patch("collectors.yahoo.normalize.settings") as mock_settings:
            mock_settings.us_stock_data_dir = str(tmp_path / "default_stock")
            mock_settings.us_normalized_data_dir = str(tmp_path / "default_normalized")
            mock_settings.us_stock_weekly_data_dir = str(tmp_path / "stock_weekly_data")
            mock_settings.us_normalized_weekly_data_dir = str(tmp_path / "normalized_weekly_data")

            normalizer = YahooNormalizer(
                source_dir=str(tmp_path / "custom_source"),
                target_dir=str(tmp_path / "custom_target"),
                interval="1d"
            )

            assert str(normalizer.us_stock_data_dir) == str(tmp_path / "custom_source")
            assert str(normalizer.us_normalized_data_dir) == str(tmp_path / "custom_target")

    def test_calc_change(self, normalizer):
        """Test calculating daily change series."""
        df = pd.DataFrame({
            "close": [100, 110, 105, 115, 120]
        })

        result = YahooNormalizer.calc_change(df)

        assert len(result) == 5
        assert pd.isna(result.iloc[0])  # First value should be NaN
        assert result.iloc[1] == pytest.approx(0.1)  # 10% increase
        assert result.iloc[2] == pytest.approx(-0.0454545, rel=1e-3)  # ~4.5% decrease

    def test_calc_change_with_last_close(self, normalizer):
        """Test calculating change with last_close parameter."""
        df = pd.DataFrame({
            "close": [100, 110, 105]
        })

        result = YahooNormalizer.calc_change(df, last_close=90)

        assert len(result) == 3
        assert result.iloc[0] == pytest.approx(0.1111, rel=1e-3)  # 100/90 - 1

    def test_normalize_yahoo_data_basic(self, normalizer):
        """Test basic normalization of Yahoo data."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-02", periods=5, freq="B"),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [105.0, 106.0, 107.0, 108.0, 109.0],
            "low": [95.0, 96.0, 97.0, 98.0, 99.0],
            "close": [102.0, 103.0, 104.0, 105.0, 106.0],
            "volume": [1000, 1100, 1200, 1300, 1400],
            "symbol": ["AAPL"] * 5
        })

        result = normalizer.normalize_yahoo_data(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert "change" in result.columns
        assert "date" in result.columns

    def test_normalize_yahoo_data_empty_df(self, normalizer):
        """Test normalization returns empty DataFrame for empty input."""
        df = pd.DataFrame()

        result = normalizer.normalize_yahoo_data(df)

        assert result.empty

    def test_normalize_yahoo_data_removes_duplicates(self, normalizer):
        """Test that duplicate dates are removed."""
        df = pd.DataFrame({
            "date": ["2024-01-02", "2024-01-02", "2024-01-03"],  # Duplicate date
            "open": [100.0, 100.5, 101.0],
            "high": [105.0, 105.5, 106.0],
            "low": [95.0, 95.5, 96.0],
            "close": [102.0, 102.5, 103.0],
            "volume": [1000, 1050, 1100],
            "symbol": ["AAPL"] * 3
        })

        result = normalizer.normalize_yahoo_data(df)

        assert len(result) == 2  # One duplicate removed

    def test_normalize_yahoo_data_handles_invalid_volume(self, normalizer):
        """Test that rows with invalid volume are set to NaN."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-02", periods=3, freq="B"),
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [95.0, 96.0, 97.0],
            "close": [102.0, 103.0, 104.0],
            "volume": [1000, 0, 1200],  # Invalid volume on second row
            "symbol": ["AAPL"] * 3
        })

        result = normalizer.normalize_yahoo_data(df)

        # Second row values should be NaN (except symbol)
        assert pd.isna(result.iloc[1]["close"])

    def test_adjust_price(self, normalizer):
        """Test price adjustment for splits and dividends."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-02", periods=3, freq="B"),
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [95.0, 96.0, 97.0],
            "close": [102.0, 103.0, 104.0],
            "volume": [1000, 1100, 1200],
            "adjclose": [102.0, 103.0, 104.0],  # Same as close (no adjustment needed)
            "symbol": ["AAPL"] * 3
        })

        result = normalizer.adjust_price(df)

        assert isinstance(result, pd.DataFrame)
        assert "factor" in result.columns
        assert result["factor"].iloc[0] == pytest.approx(1.0)

    def test_adjust_price_with_split(self, normalizer):
        """Test price adjustment with stock split."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-02", periods=3, freq="B"),
            "open": [100.0, 101.0, 51.0],  # Price halved on third day (2:1 split)
            "high": [105.0, 106.0, 54.0],
            "low": [95.0, 96.0, 48.0],
            "close": [102.0, 103.0, 52.0],
            "volume": [1000, 1100, 2200],  # Volume doubled
            "adjclose": [51.0, 51.5, 52.0],  # Adjusted close reflects split
            "symbol": ["AAPL"] * 3
        })

        result = normalizer.adjust_price(df)

        assert isinstance(result, pd.DataFrame)
        # Factor should show adjustment
        assert "factor" in result.columns

    def test_adjust_price_empty_df(self, normalizer):
        """Test adjust_price returns empty DataFrame for empty input."""
        df = pd.DataFrame()

        result = normalizer.adjust_price(df)

        assert result.empty

    def test_manual_adjust_data(self, normalizer):
        """Test manual data adjustment normalizes to first day close."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-02", periods=3, freq="B"),
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [95.0, 96.0, 97.0],
            "close": [102.0, 103.0, 104.0],
            "volume": [1000, 1100, 1200],
            "symbol": ["AAPL"] * 3
        })

        result = normalizer.manual_adjust_data(df)

        # First close should be 1.0 after normalization
        assert result.iloc[0]["close"] == pytest.approx(1.0)

    def test_manual_adjust_data_empty_df(self, normalizer):
        """Test manual_adjust_data returns empty DataFrame for empty input."""
        df = pd.DataFrame()

        result = normalizer.manual_adjust_data(df)

        assert result.empty

    def test_get_first_close(self, normalizer):
        """Test getting first valid close price."""
        df = pd.DataFrame({
            "close": [np.nan, 100.0, 101.0]
        }, index=pd.date_range("2024-01-02", periods=3, freq="B"))

        result = normalizer._get_first_close(df)

        assert result == 100.0

    def test_normalize_single_symbol(self, normalizer, tmp_path):
        """Test normalizing a single symbol."""
        # Create source data file
        source_dir = normalizer.us_stock_data_dir
        source_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame({
            "date": pd.date_range("2024-01-02", periods=3, freq="B"),
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [95.0, 96.0, 97.0],
            "close": [102.0, 103.0, 104.0],
            "volume": [1000, 1100, 1200],
            "adjclose": [102.0, 103.0, 104.0],
            "symbol": ["AAPL"] * 3
        })
        df.to_csv(source_dir / "AAPL.csv", index=False)

        result = normalizer.normalize_single_symbol("AAPL")

        assert result is True
        assert (normalizer.us_normalized_data_dir / "AAPL.csv").exists()

    def test_normalize_single_symbol_missing_file(self, normalizer):
        """Test normalizing symbol with missing source file."""
        result = normalizer.normalize_single_symbol("MISSING")

        assert result is False

    def test_normalize_single_symbol_empty_file(self, normalizer):
        """Test normalizing symbol with empty source file."""
        source_dir = normalizer.us_stock_data_dir
        source_dir.mkdir(parents=True, exist_ok=True)

        # Create empty CSV
        pd.DataFrame().to_csv(source_dir / "EMPTY.csv", index=False)

        result = normalizer.normalize_single_symbol("EMPTY")

        assert result is False

    def test_normalize_single_symbol_with_date_filter(self, normalizer):
        """Test normalizing with date range filter."""
        # Create source data file
        source_dir = normalizer.us_stock_data_dir
        source_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10, freq="B"),
            "open": [100.0] * 10,
            "high": [105.0] * 10,
            "low": [95.0] * 10,
            "close": [102.0] * 10,
            "volume": [1000] * 10,
            "adjclose": [102.0] * 10,
            "symbol": ["AAPL"] * 10
        })
        df.to_csv(source_dir / "AAPL.csv", index=False)

        normalizer.start_date = "2024-01-03"
        normalizer.end_date = "2024-01-08"

        result = normalizer.normalize_single_symbol("AAPL")

        assert result is True

    def test_get_symbol_files(self, normalizer):
        """Test getting list of symbol files."""
        source_dir = normalizer.us_stock_data_dir
        source_dir.mkdir(parents=True, exist_ok=True)

        # Create some CSV files
        (source_dir / "AAPL.csv").write_text("data")
        (source_dir / "MSFT.csv").write_text("data")
        (source_dir / "GOOG.csv").write_text("data")

        symbols = normalizer._get_symbol_files()

        assert len(symbols) == 3
        assert "AAPL" in symbols
        assert "MSFT" in symbols
        assert "GOOG" in symbols

    def test_get_symbol_files_empty_directory(self, normalizer):
        """Test getting symbol files from empty directory."""
        source_dir = normalizer.us_stock_data_dir
        source_dir.mkdir(parents=True, exist_ok=True)

        symbols = normalizer._get_symbol_files()

        assert symbols == []

    def test_get_symbol_files_missing_directory(self, tmp_path):
        """Test getting symbol files when source directory doesn't exist."""
        with patch("collectors.yahoo.normalize.settings") as mock_settings:
            mock_settings.us_stock_data_dir = str(tmp_path / "missing_dir")
            mock_settings.us_normalized_data_dir = str(tmp_path / "normalized")
            mock_settings.us_stock_weekly_data_dir = str(tmp_path / "stock_weekly_data")
            mock_settings.us_normalized_weekly_data_dir = str(tmp_path / "normalized_weekly_data")

            normalizer = YahooNormalizer(interval="1d")
            symbols = normalizer._get_symbol_files()

            assert symbols == []


class TestNormalizeYahooData:
    """Tests for normalize_yahoo_data function."""

    def test_normalize_yahoo_data_creates_normalizer(self, tmp_path):
        """Test that normalize_yahoo_data creates normalizer and calls normalize."""
        with patch("collectors.yahoo.normalize.settings") as mock_settings, \
             patch.object(YahooNormalizer, "normalize") as mock_normalize:

            mock_settings.us_stock_data_dir = str(tmp_path / "stock_data")
            mock_settings.us_normalized_data_dir = str(tmp_path / "normalized_data")
            mock_settings.us_stock_weekly_data_dir = str(tmp_path / "stock_weekly_data")
            mock_settings.us_normalized_weekly_data_dir = str(tmp_path / "normalized_weekly_data")

            normalize_yahoo_data(
                start_date="2024-01-01",
                end_date="2024-01-31",
                max_workers=4
            )

            mock_normalize.assert_called_once()
