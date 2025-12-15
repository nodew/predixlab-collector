"""Unit tests for collectors/us_index/collector.py module."""
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from collectors.us_index import USIndexCollector


class TestUSIndexCollector:
    """Tests for USIndexCollector class."""

    @pytest.fixture
    def collector(self, tmp_path):
        """Create a USIndexCollector with a temporary output path."""
        with patch("collectors.us_index.collector.settings") as mock_settings:
            mock_settings.us_index_path = str(tmp_path / "us_index.txt")
            return USIndexCollector()

    def test_init_creates_parent_directory(self, tmp_path):
        """Test that __init__ creates parent directory for index file."""
        with patch("collectors.us_index.collector.settings") as mock_settings:
            nested_path = tmp_path / "nested" / "dir" / "us_index.txt"
            mock_settings.us_index_path = str(nested_path)

            _ = USIndexCollector()

            assert nested_path.parent.exists()

    def test_get_sp500_symbols_with_dates(self, collector):
        """Test fetching SP500 symbols with dates."""
        # Create mock HTML response with a table
        html_content = """
        <html><body>
        <table>
            <tr><th>Symbol</th><th>Date added</th></tr>
            <tr><td>AAPL</td><td>1982-11-30</td></tr>
            <tr><td>MSFT</td><td>1994-06-01</td></tr>
            <tr><td>GOOG</td><td>2006-04-03</td></tr>
        </table>
        </body></html>
        """

        mock_response = MagicMock()
        mock_response.text = html_content
        mock_response.raise_for_status = MagicMock()

        collector.session.get = MagicMock(return_value=mock_response)

        result = collector.get_sp500_symbols_with_dates()

        assert isinstance(result, pd.DataFrame)
        assert "symbol" in result.columns
        assert "date_added" in result.columns
        assert len(result) == 3
        assert "AAPL" in result["symbol"].values

    def test_get_nasdaq100_symbols_with_dates(self, collector):
        """Test fetching NASDAQ100 symbols with dates."""
        # Create mock HTML response with a table (NASDAQ100 uses "Ticker" column)
        html_content = """
        <html><body>
        <table>
            <tr><th>Company</th><th>Ticker</th></tr>
        """ + "\n".join([f"<tr><td>Company{i}</td><td>TICK{i}</td></tr>" for i in range(105)]) + """
        </table>
        </body></html>
        """

        mock_response = MagicMock()
        mock_response.text = html_content
        mock_response.raise_for_status = MagicMock()

        collector.session.get = MagicMock(return_value=mock_response)

        result = collector.get_nasdaq100_symbols_with_dates()

        assert isinstance(result, pd.DataFrame)
        assert "symbol" in result.columns
        assert "date_added" in result.columns
        assert len(result) >= 100

    def test_merge_and_save_symbols(self, tmp_path):
        """Test merging and saving symbols from both indices."""
        with patch("collectors.us_index.collector.settings") as mock_settings:
            output_path = tmp_path / "us_index.txt"
            mock_settings.us_index_path = str(output_path)

            collector = USIndexCollector()

            # Create sample DataFrames
            sp500_df = pd.DataFrame({
                "symbol": ["AAPL", "MSFT", "GOOG"],
                "date_added": ["2015-01-01", "2015-01-02", "2015-01-03"]
            })

            nasdaq100_df = pd.DataFrame({
                "symbol": ["NVDA", "AMZN", "AAPL"],  # AAPL is duplicate
                "date_added": ["2015-02-01", "2015-02-02", "2015-02-03"]
            })

            collector.merge_and_save_symbols(sp500_df, nasdaq100_df)

            # Verify file was created
            assert output_path.exists()

            # Read and verify content
            content = output_path.read_text()
            lines = [line for line in content.split("\n") if line.strip()]

            # Should have 4 unique symbols (AAPL deduplicated)
            assert len(lines) == 5  # AAPL, AMZN, GOOG, MSFT, NVDA (sorted)

            # Verify format: symbol \t start_date \t end_date
            first_line = lines[0].split("\t")
            assert len(first_line) == 3
            assert first_line[2] == "2099-12-31"  # end_date

    def test_merge_replaces_dots_with_hyphens(self, tmp_path):
        """Test that dots in symbols are replaced with hyphens."""
        with patch("collectors.us_index.collector.settings") as mock_settings:
            output_path = tmp_path / "us_index.txt"
            mock_settings.us_index_path = str(output_path)

            collector = USIndexCollector()

            sp500_df = pd.DataFrame({
                "symbol": ["BRK.B", "BF.B"],
                "date_added": ["2015-01-01", "2015-01-02"]
            })

            nasdaq100_df = pd.DataFrame({
                "symbol": [],
                "date_added": []
            })

            collector.merge_and_save_symbols(sp500_df, nasdaq100_df)

            content = output_path.read_text()
            assert "BRK-B" in content
            assert "BF-B" in content
            assert "BRK.B" not in content

    def test_collect_calls_all_methods(self, tmp_path):
        """Test that collect method calls all required methods."""
        with patch("collectors.us_index.collector.settings") as mock_settings:
            output_path = tmp_path / "us_index.txt"
            mock_settings.us_index_path = str(output_path)

            collector = USIndexCollector()

            # Mock the data fetching methods
            sp500_df = pd.DataFrame({
                "symbol": ["AAPL", "MSFT"],
                "date_added": ["2015-01-01", "2015-01-02"]
            })

            nasdaq100_df = pd.DataFrame({
                "symbol": ["NVDA"],
                "date_added": ["2015-02-01"]
            })

            collector.get_sp500_symbols_with_dates = MagicMock(return_value=sp500_df)
            collector.get_nasdaq100_symbols_with_dates = MagicMock(return_value=nasdaq100_df)

            collector.collect()

            collector.get_sp500_symbols_with_dates.assert_called_once()
            collector.get_nasdaq100_symbols_with_dates.assert_called_once()
            assert output_path.exists()

    def test_collect_raises_on_error(self, tmp_path):
        """Test that collect raises exception on error."""
        with patch("collectors.us_index.collector.settings") as mock_settings:
            output_path = tmp_path / "us_index.txt"
            mock_settings.us_index_path = str(output_path)

            collector = USIndexCollector()

            # Mock to raise exception
            collector.get_sp500_symbols_with_dates = MagicMock(
                side_effect=Exception("Network error")
            )

            with pytest.raises(Exception, match="Network error"):
                collector.collect()


class TestCollectUSIndex:
    """Tests for collect_us_index function."""

    def test_collect_us_index_creates_collector(self, tmp_path):
        """Test that collect_us_index creates collector and calls collect."""
        with patch("collectors.us_index.collector.settings") as mock_settings, \
             patch.object(USIndexCollector, "collect") as mock_collect:

            mock_settings.us_index_path = str(tmp_path / "us_index.txt")

            from collectors.us_index import collect_us_index
            collect_us_index()

            mock_collect.assert_called_once()
