"""Unit tests for config.py module."""
import pytest
from unittest.mock import patch
from pathlib import Path

from config import Settings


class TestSettingsValidation:
    """Tests for Settings validation."""

    def test_mongodb_url_validation_valid(self):
        """Test that valid MongoDB URL passes validation."""
        settings = Settings(mongodb_url="mongodb://localhost:27017")
        assert settings.mongodb_url == "mongodb://localhost:27017"

        settings_srv = Settings(mongodb_url="mongodb+srv://user:pass@cluster.example.com")
        assert settings_srv.mongodb_url == "mongodb+srv://user:pass@cluster.example.com"

    def test_mongodb_url_validation_invalid(self):
        """Test that invalid MongoDB URL raises ValueError."""
        with pytest.raises(ValueError, match="MongoDB URL must start with"):
            Settings(mongodb_url="http://localhost:27017")

    def test_mongodb_url_validation_empty(self):
        """Test that empty MongoDB URL passes validation."""
        settings = Settings(mongodb_url="")
        assert settings.mongodb_url == ""

    def test_sender_email_validation_valid(self):
        """Test that valid sender email passes validation."""
        settings = Settings(acs_sender_email="test@example.com")
        assert settings.acs_sender_email == "test@example.com"

    def test_sender_email_validation_invalid(self):
        """Test that invalid sender email raises ValueError."""
        with pytest.raises(ValueError, match="Invalid sender email format"):
            Settings(acs_sender_email="invalid-email")

    def test_sender_email_validation_empty(self):
        """Test that empty sender email passes validation."""
        settings = Settings(acs_sender_email="")
        assert settings.acs_sender_email == ""

    def test_to_emails_validation_valid(self):
        """Test that valid recipient emails pass validation."""
        settings = Settings(acs_to_emails="user1@example.com,user2@example.com")
        assert settings.acs_to_emails == "user1@example.com,user2@example.com"

    def test_to_emails_validation_invalid(self):
        """Test that invalid recipient email raises ValueError."""
        with pytest.raises(ValueError, match="Invalid email format"):
            Settings(acs_to_emails="user1@example.com,invalid-email")

    def test_to_emails_validation_empty(self):
        """Test that empty recipient emails pass validation."""
        settings = Settings(acs_to_emails="")
        assert settings.acs_to_emails == ""


class TestSettingsHelperMethods:
    """Tests for Settings helper methods."""

    def test_get_to_emails_list_single_email(self):
        """Test get_to_emails_list with single email."""
        settings = Settings(acs_to_emails="user@example.com")
        result = settings.get_to_emails_list()
        assert result == ["user@example.com"]

    def test_get_to_emails_list_multiple_emails(self):
        """Test get_to_emails_list with multiple emails."""
        settings = Settings(acs_to_emails="user1@example.com,user2@example.com,user3@example.com")
        result = settings.get_to_emails_list()
        assert result == ["user1@example.com", "user2@example.com", "user3@example.com"]

    def test_get_to_emails_list_with_whitespace(self):
        """Test get_to_emails_list strips whitespace."""
        settings = Settings(acs_to_emails=" user1@example.com , user2@example.com ")
        result = settings.get_to_emails_list()
        assert result == ["user1@example.com", "user2@example.com"]

    def test_get_to_emails_list_empty(self):
        """Test get_to_emails_list with empty string."""
        settings = Settings(acs_to_emails="")
        result = settings.get_to_emails_list()
        assert result == []

    def test_is_email_configured_true(self):
        """Test is_email_configured returns True when fully configured."""
        settings = Settings(
            acs_connection_string="connection-string",
            acs_sender_email="sender@example.com",
            acs_to_emails="recipient@example.com"
        )
        assert settings.is_email_configured() is True

    def test_is_email_configured_false_missing_connection_string(self):
        """Test is_email_configured returns False when connection string missing."""
        settings = Settings(
            acs_connection_string="",
            acs_sender_email="sender@example.com",
            acs_to_emails="recipient@example.com"
        )
        assert settings.is_email_configured() is False

    def test_is_email_configured_false_missing_sender(self):
        """Test is_email_configured returns False when sender email missing."""
        settings = Settings(
            acs_connection_string="connection-string",
            acs_sender_email="",
            acs_to_emails="recipient@example.com"
        )
        assert settings.is_email_configured() is False

    def test_is_email_configured_false_missing_recipients(self):
        """Test is_email_configured returns False when recipients missing."""
        settings = Settings(
            acs_connection_string="connection-string",
            acs_sender_email="sender@example.com",
            acs_to_emails=""
        )
        assert settings.is_email_configured() is False


class TestSettingsEnsureDirectoriesExist:
    """Tests for ensure_directories_exist method."""

    def test_ensure_directories_exist_creates_directories(self, tmp_path):
        """Test that ensure_directories_exist creates all required directories."""
        settings = Settings(
            calendar_dir=str(tmp_path / "calendar"),
            index_dir=str(tmp_path / "instruments"),
            stock_data_dir=str(tmp_path / "stock_data"),
            stock_weekly_data_dir=str(tmp_path / "stock_weekly_data"),
            us_stock_data_dir=str(tmp_path / "stock_data" / "us_data"),
            cn_stock_data_dir=str(tmp_path / "stock_data" / "cn_data"),
            us_stock_weekly_data_dir=str(tmp_path / "stock_weekly_data" / "us_data"),
            cn_stock_weekly_data_dir=str(tmp_path / "stock_weekly_data" / "cn_data"),
            normalized_data_dir=str(tmp_path / "normalized_data"),
            normalized_weekly_data_dir=str(tmp_path / "normalized_weekly_data"),
            us_normalized_data_dir=str(tmp_path / "normalized_data" / "us_data"),
            cn_normalized_data_dir=str(tmp_path / "normalized_data" / "cn_data"),
            us_normalized_weekly_data_dir=str(tmp_path / "normalized_weekly_data" / "us_data"),
            cn_normalized_weekly_data_dir=str(tmp_path / "normalized_weekly_data" / "cn_data"),
        )

        settings.ensure_directories_exist()

        # Verify directories were created
        assert (tmp_path / "calendar").exists()
        assert (tmp_path / "instruments").exists()
        assert (tmp_path / "stock_data" / "us_data").exists()
        assert (tmp_path / "stock_data" / "cn_data").exists()
        assert (tmp_path / "normalized_data" / "us_data").exists()


class TestSettingsDefaultValues:
    """Tests for Settings default values."""

    def test_default_calendar_paths(self):
        """Test default calendar path values."""
        settings = Settings()
        assert settings.calendar_dir == "data/calendar"
        assert settings.us_calendar_path == "data/calendar/us.txt"
        assert settings.cn_calendar_path == "data/calendar/cn.txt"

    def test_default_index_paths(self):
        """Test default index path values."""
        settings = Settings()
        assert settings.index_dir == "data/instruments"
        assert settings.us_index_path == "data/instruments/us.txt"
        assert settings.csi300_index_path == "data/instruments/csi300.txt"
        assert settings.csi500_index_path == "data/instruments/csi500.txt"

    def test_default_mongodb_settings(self):
        """Test default MongoDB settings."""
        settings = Settings()
        assert settings.mongodb_url == "mongodb://localhost:27017"
        assert settings.database_name == "predixlab"
        assert settings.jobs_collection == "jobs"
