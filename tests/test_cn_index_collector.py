"""Unit tests for collectors/cn_index/collector.py module."""
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from pathlib import Path

from collectors.cn_index import CNIndexCollector


class TestCNIndexCollector:
    """Tests for CNIndexCollector class."""

    @pytest.fixture
    def collector(self, tmp_path):
        """Create a CNIndexCollector with a temporary output path."""
        with patch('collectors.cn_index.collector.settings') as mock_settings:
            mock_settings.csi300_index_path = str(tmp_path / "csi300.txt")
            mock_settings.csi500_index_path = str(tmp_path / "csi500.txt")
            collector = CNIndexCollector()
            return collector

    def test_init_creates_parent_directory(self, tmp_path):
        """Test that __init__ creates parent directory for index files."""
        with patch('collectors.cn_index.collector.settings') as mock_settings:
            nested_path_300 = tmp_path / "nested" / "dir" / "csi300.txt"
            nested_path_500 = tmp_path / "nested" / "dir" / "csi500.txt"
            mock_settings.csi300_index_path = str(nested_path_300)
            mock_settings.csi500_index_path = str(nested_path_500)
            
            collector = CNIndexCollector()
            
            assert nested_path_300.parent.exists()
            assert nested_path_500.parent.exists()

    def test_get_csi300_symbols_with_dates(self, collector):
        """Test fetching CSI 300 symbols with dates."""
        # Create mock JSON response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "diff": [
                    {"f12": "600000", "f14": "浦发银行"},
                    {"f12": "600016", "f14": "民生银行"},
                    {"f12": "000001", "f14": "平安银行"},
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()
        
        collector.session.get = MagicMock(return_value=mock_response)
        
        result = collector.get_csi300_symbols_with_dates()
        
        assert isinstance(result, pd.DataFrame)
        assert 'symbol' in result.columns
        assert 'date_added' in result.columns
        assert len(result) == 3
        # Check Shanghai symbols end with .SS
        assert '600000.SS' in result['symbol'].values
        assert '600016.SS' in result['symbol'].values
        # Check Shenzhen symbols end with .SZ
        assert '000001.SZ' in result['symbol'].values

    def test_get_csi500_symbols_with_dates(self, collector):
        """Test fetching CSI 500 symbols with dates."""
        # Create mock JSON response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "diff": [
                    {"f12": "600100", "f14": "同方股份"},
                    {"f12": "300059", "f14": "东方财富"},
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()
        
        collector.session.get = MagicMock(return_value=mock_response)
        
        result = collector.get_csi500_symbols_with_dates()
        
        assert isinstance(result, pd.DataFrame)
        assert 'symbol' in result.columns
        assert 'date_added' in result.columns
        assert len(result) == 2

    def test_fetch_index_retries_on_failure(self, tmp_path):
        """Test that fetch retries on failure."""
        with patch('collectors.cn_index.collector.settings') as mock_settings, \
             patch('collectors.cn_index.collector.time.sleep'):
            
            mock_settings.csi300_index_path = str(tmp_path / "csi300.txt")
            mock_settings.csi500_index_path = str(tmp_path / "csi500.txt")
            collector = CNIndexCollector()
            
            # First call fails, second succeeds
            mock_response_success = MagicMock()
            mock_response_success.json.return_value = {
                "data": {
                    "diff": [{"f12": "600000", "f14": "Test"}]
                }
            }
            mock_response_success.raise_for_status = MagicMock()
            
            collector.session.get = MagicMock(side_effect=[
                Exception("Network error"),
                mock_response_success
            ])
            
            result = collector.get_csi300_symbols_with_dates()
            
            assert len(result) == 1
            assert collector.session.get.call_count == 2

    def test_save_csi300_symbols(self, tmp_path):
        """Test saving CSI 300 symbols to file."""
        with patch('collectors.cn_index.collector.settings') as mock_settings:
            output_path = tmp_path / "csi300.txt"
            mock_settings.csi300_index_path = str(output_path)
            mock_settings.csi500_index_path = str(tmp_path / "csi500.txt")
            
            collector = CNIndexCollector()
            
            # Create sample DataFrame
            csi300_df = pd.DataFrame({
                'symbol': ['600000.SS', '600016.SS', '000001.SZ'],
                'date_added': ['2005-01-01', '2005-01-01', '2005-01-01']
            })
            
            collector.save_csi300_symbols(csi300_df)
            
            # Verify file was created
            assert output_path.exists()
            
            # Read and verify content
            content = output_path.read_text()
            lines = [l for l in content.split('\n') if l.strip()]
            
            assert len(lines) == 3
            
            # Verify format: symbol \t start_date \t end_date
            first_line = lines[0].split('\t')
            assert len(first_line) == 3
            assert first_line[2] == '2099-12-31'  # end_date

    def test_save_csi500_symbols(self, tmp_path):
        """Test saving CSI 500 symbols to file."""
        with patch('collectors.cn_index.collector.settings') as mock_settings:
            output_path = tmp_path / "csi500.txt"
            mock_settings.csi300_index_path = str(tmp_path / "csi300.txt")
            mock_settings.csi500_index_path = str(output_path)
            
            collector = CNIndexCollector()
            
            # Create sample DataFrame
            csi500_df = pd.DataFrame({
                'symbol': ['600100.SS', '300059.SZ'],
                'date_added': ['2005-01-01', '2005-01-01']
            })
            
            collector.save_csi500_symbols(csi500_df)
            
            # Verify file was created
            assert output_path.exists()
            
            # Read and verify content
            content = output_path.read_text()
            lines = [l for l in content.split('\n') if l.strip()]
            
            assert len(lines) == 2

    def test_collect_saves_separate_files(self, tmp_path):
        """Test that collect method saves CSI 300 and CSI 500 to separate files."""
        with patch('collectors.cn_index.collector.settings') as mock_settings:
            csi300_path = tmp_path / "csi300.txt"
            csi500_path = tmp_path / "csi500.txt"
            mock_settings.csi300_index_path = str(csi300_path)
            mock_settings.csi500_index_path = str(csi500_path)
            
            collector = CNIndexCollector()
            
            # Mock the data fetching methods
            csi300_df = pd.DataFrame({
                'symbol': ['600000.SS', '600016.SS'],
                'date_added': ['2005-01-01', '2005-01-01']
            })
            
            csi500_df = pd.DataFrame({
                'symbol': ['600100.SS', '300059.SZ'],
                'date_added': ['2005-01-01', '2005-01-01']
            })
            
            collector.get_csi300_symbols_with_dates = MagicMock(return_value=csi300_df)
            collector.get_csi500_symbols_with_dates = MagicMock(return_value=csi500_df)
            
            collector.collect()
            
            collector.get_csi300_symbols_with_dates.assert_called_once()
            collector.get_csi500_symbols_with_dates.assert_called_once()
            
            # Verify both files were created
            assert csi300_path.exists()
            assert csi500_path.exists()
            
            # Verify CSI 300 file content
            csi300_lines = [l for l in csi300_path.read_text().split('\n') if l.strip()]
            assert len(csi300_lines) == 2
            
            # Verify CSI 500 file content
            csi500_lines = [l for l in csi500_path.read_text().split('\n') if l.strip()]
            assert len(csi500_lines) == 2

    def test_collect_raises_on_error(self, tmp_path):
        """Test that collect raises exception on error."""
        with patch('collectors.cn_index.collector.settings') as mock_settings:
            mock_settings.csi300_index_path = str(tmp_path / "csi300.txt")
            mock_settings.csi500_index_path = str(tmp_path / "csi500.txt")
            
            collector = CNIndexCollector()
            
            # Mock to raise exception
            collector.get_csi300_symbols_with_dates = MagicMock(
                side_effect=Exception("Network error")
            )
            
            with pytest.raises(Exception, match="Network error"):
                collector.collect()

    def test_symbol_exchange_suffix(self, collector):
        """Test that correct exchange suffix is assigned."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "diff": [
                    {"f12": "600000", "f14": "Shanghai Stock"},  # 6xx -> .SS
                    {"f12": "000001", "f14": "Shenzhen Stock"},  # 0xx -> .SZ
                    {"f12": "300059", "f14": "ChiNext Stock"},   # 3xx -> .SZ
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()
        
        collector.session.get = MagicMock(return_value=mock_response)
        
        result = collector.get_csi300_symbols_with_dates()
        
        assert '600000.SS' in result['symbol'].values
        assert '000001.SZ' in result['symbol'].values
        assert '300059.SZ' in result['symbol'].values


class TestCollectCNIndex:
    """Tests for collect_cn_index function."""

    def test_collect_cn_index_creates_collector(self, tmp_path):
        """Test that collect_cn_index creates collector and calls collect."""
        with patch('collectors.cn_index.collector.settings') as mock_settings, \
             patch.object(CNIndexCollector, 'collect') as mock_collect:
            
            mock_settings.csi300_index_path = str(tmp_path / "csi300.txt")
            mock_settings.csi500_index_path = str(tmp_path / "csi500.txt")
            
            from collectors.cn_index import collect_cn_index
            collect_cn_index()
            
            mock_collect.assert_called_once()
