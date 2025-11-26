"""Unit tests for collectors/cn_index/collector.py module."""
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from pathlib import Path

from collectors.cn_index import CNIndexCollector, collect_cn_index


class TestCNIndexCollector:
    """Tests for CNIndexCollector class."""

    @pytest.fixture
    def csi300_collector(self, tmp_path):
        """Create a CNIndexCollector for CSI 300 with a temporary output path."""
        with patch('collectors.cn_index.collector.settings') as mock_settings:
            mock_settings.csi300_index_path = str(tmp_path / "csi300.txt")
            mock_settings.csi500_index_path = str(tmp_path / "csi500.txt")
            collector = CNIndexCollector(index="csi300")
            return collector

    @pytest.fixture
    def csi500_collector(self, tmp_path):
        """Create a CNIndexCollector for CSI 500 with a temporary output path."""
        with patch('collectors.cn_index.collector.settings') as mock_settings:
            mock_settings.csi300_index_path = str(tmp_path / "csi300.txt")
            mock_settings.csi500_index_path = str(tmp_path / "csi500.txt")
            collector = CNIndexCollector(index="csi500")
            return collector

    def test_init_default_csi300(self, tmp_path):
        """Test that default index is csi300."""
        with patch('collectors.cn_index.collector.settings') as mock_settings:
            mock_settings.csi300_index_path = str(tmp_path / "csi300.txt")
            mock_settings.csi500_index_path = str(tmp_path / "csi500.txt")
            
            collector = CNIndexCollector()
            
            assert collector.index == "csi300"
            assert collector.index_config["code"] == "000300"

    def test_init_csi500(self, tmp_path):
        """Test initialization with csi500."""
        with patch('collectors.cn_index.collector.settings') as mock_settings:
            mock_settings.csi300_index_path = str(tmp_path / "csi300.txt")
            mock_settings.csi500_index_path = str(tmp_path / "csi500.txt")
            
            collector = CNIndexCollector(index="csi500")
            
            assert collector.index == "csi500"
            assert collector.index_config["code"] == "000905"

    def test_init_invalid_index(self, tmp_path):
        """Test that invalid index raises ValueError."""
        with patch('collectors.cn_index.collector.settings') as mock_settings:
            mock_settings.csi300_index_path = str(tmp_path / "csi300.txt")
            mock_settings.csi500_index_path = str(tmp_path / "csi500.txt")
            
            with pytest.raises(ValueError, match="Invalid index"):
                CNIndexCollector(index="invalid")

    def test_init_creates_parent_directory(self, tmp_path):
        """Test that __init__ creates parent directory for index file."""
        with patch('collectors.cn_index.collector.settings') as mock_settings:
            nested_path = tmp_path / "nested" / "dir" / "csi300.txt"
            mock_settings.csi300_index_path = str(nested_path)
            mock_settings.csi500_index_path = str(tmp_path / "csi500.txt")
            
            collector = CNIndexCollector(index="csi300")
            
            assert nested_path.parent.exists()

    def test_fetch_csi300_constituents(self, csi300_collector):
        """Test fetching CSI 300 symbols."""
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
        
        csi300_collector.session.get = MagicMock(return_value=mock_response)
        
        result = csi300_collector._fetch_index_constituents()
        
        assert isinstance(result, pd.DataFrame)
        assert 'symbol' in result.columns
        assert len(result) == 3
        # Check Shanghai symbols end with .SS
        assert '600000.SS' in result['symbol'].values
        assert '600016.SS' in result['symbol'].values
        # Check Shenzhen symbols end with .SZ
        assert '000001.SZ' in result['symbol'].values

    def test_fetch_csi500_constituents(self, csi500_collector):
        """Test fetching CSI 500 symbols."""
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
        
        csi500_collector.session.get = MagicMock(return_value=mock_response)
        
        result = csi500_collector._fetch_index_constituents()
        
        assert isinstance(result, pd.DataFrame)
        assert 'symbol' in result.columns
        assert len(result) == 2

    def test_fetch_index_retries_on_failure(self, tmp_path):
        """Test that fetch retries on failure."""
        with patch('collectors.cn_index.collector.settings') as mock_settings, \
             patch('collectors.cn_index.collector.time.sleep'):
            
            mock_settings.csi300_index_path = str(tmp_path / "csi300.txt")
            mock_settings.csi500_index_path = str(tmp_path / "csi500.txt")
            collector = CNIndexCollector(index="csi300")
            
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
            
            result = collector._fetch_index_constituents()
            
            assert len(result) == 1
            assert collector.session.get.call_count == 2

    def test_save_symbols(self, tmp_path):
        """Test saving symbols to file."""
        with patch('collectors.cn_index.collector.settings') as mock_settings:
            output_path = tmp_path / "csi300.txt"
            mock_settings.csi300_index_path = str(output_path)
            mock_settings.csi500_index_path = str(tmp_path / "csi500.txt")
            
            collector = CNIndexCollector(index="csi300")
            
            # Create sample DataFrame
            df = pd.DataFrame({
                'symbol': ['600000.SS', '600016.SS', '000001.SZ']
            })
            
            collector._save_symbols(df)
            
            # Verify file was created
            assert output_path.exists()
            
            # Read and verify content
            content = output_path.read_text()
            lines = [l for l in content.split('\n') if l.strip()]
            
            assert len(lines) == 3
            # Verify sorted order
            assert lines[0] == '000001.SZ'
            assert lines[1] == '600000.SS'
            assert lines[2] == '600016.SS'

    def test_collect_csi300(self, tmp_path):
        """Test that collect method saves CSI 300 to file."""
        with patch('collectors.cn_index.collector.settings') as mock_settings:
            csi300_path = tmp_path / "csi300.txt"
            mock_settings.csi300_index_path = str(csi300_path)
            mock_settings.csi500_index_path = str(tmp_path / "csi500.txt")
            
            collector = CNIndexCollector(index="csi300")
            
            # Mock the data fetching method
            df = pd.DataFrame({
                'symbol': ['600000.SS', '600016.SS']
            })
            
            collector._fetch_index_constituents = MagicMock(return_value=df)
            
            collector.collect()
            
            collector._fetch_index_constituents.assert_called_once()
            
            # Verify file was created
            assert csi300_path.exists()
            
            # Verify content
            lines = [l for l in csi300_path.read_text().split('\n') if l.strip()]
            assert len(lines) == 2

    def test_collect_csi500(self, tmp_path):
        """Test that collect method saves CSI 500 to file."""
        with patch('collectors.cn_index.collector.settings') as mock_settings:
            csi500_path = tmp_path / "csi500.txt"
            mock_settings.csi300_index_path = str(tmp_path / "csi300.txt")
            mock_settings.csi500_index_path = str(csi500_path)
            
            collector = CNIndexCollector(index="csi500")
            
            # Mock the data fetching method
            df = pd.DataFrame({
                'symbol': ['600100.SS', '300059.SZ']
            })
            
            collector._fetch_index_constituents = MagicMock(return_value=df)
            
            collector.collect()
            
            collector._fetch_index_constituents.assert_called_once()
            
            # Verify file was created
            assert csi500_path.exists()
            
            # Verify content
            lines = [l for l in csi500_path.read_text().split('\n') if l.strip()]
            assert len(lines) == 2

    def test_collect_raises_on_error(self, tmp_path):
        """Test that collect raises exception on error."""
        with patch('collectors.cn_index.collector.settings') as mock_settings:
            mock_settings.csi300_index_path = str(tmp_path / "csi300.txt")
            mock_settings.csi500_index_path = str(tmp_path / "csi500.txt")
            
            collector = CNIndexCollector(index="csi300")
            
            # Mock to raise exception
            collector._fetch_index_constituents = MagicMock(
                side_effect=Exception("Network error")
            )
            
            with pytest.raises(Exception, match="Network error"):
                collector.collect()

    def test_symbol_exchange_suffix(self, csi300_collector):
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
        
        csi300_collector.session.get = MagicMock(return_value=mock_response)
        
        result = csi300_collector._fetch_index_constituents()
        
        assert '600000.SS' in result['symbol'].values
        assert '000001.SZ' in result['symbol'].values
        assert '300059.SZ' in result['symbol'].values


class TestCollectCNIndex:
    """Tests for collect_cn_index function."""

    def test_collect_cn_index_default_csi300(self, tmp_path):
        """Test that collect_cn_index defaults to csi300."""
        with patch('collectors.cn_index.collector.settings') as mock_settings, \
             patch.object(CNIndexCollector, 'collect') as mock_collect:
            
            mock_settings.csi300_index_path = str(tmp_path / "csi300.txt")
            mock_settings.csi500_index_path = str(tmp_path / "csi500.txt")
            
            collect_cn_index()
            
            mock_collect.assert_called_once()

    def test_collect_cn_index_with_csi300(self, tmp_path):
        """Test collect_cn_index with csi300 parameter."""
        with patch('collectors.cn_index.collector.settings') as mock_settings, \
             patch.object(CNIndexCollector, 'collect') as mock_collect:
            
            mock_settings.csi300_index_path = str(tmp_path / "csi300.txt")
            mock_settings.csi500_index_path = str(tmp_path / "csi500.txt")
            
            collect_cn_index(index="csi300")
            
            mock_collect.assert_called_once()

    def test_collect_cn_index_with_csi500(self, tmp_path):
        """Test collect_cn_index with csi500 parameter."""
        with patch('collectors.cn_index.collector.settings') as mock_settings, \
             patch.object(CNIndexCollector, 'collect') as mock_collect:
            
            mock_settings.csi300_index_path = str(tmp_path / "csi300.txt")
            mock_settings.csi500_index_path = str(tmp_path / "csi500.txt")
            
            collect_cn_index(index="csi500")
            
            mock_collect.assert_called_once()
