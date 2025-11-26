"""Configuration module for qstock-collector.

Provides centralized configuration management with validation.
"""
from pathlib import Path
from typing import List, Optional
from pydantic import field_validator, Field
from pydantic_settings import BaseSettings
from loguru import logger


class Settings(BaseSettings):
    """Application settings with validation.
    
    All file paths are relative to the project root by default.
    Environment variables can override any setting.
    """
    # Calendar paths
    calendar_dir: str = "data/calendar"
    us_calendar_path: str = "data/calendar/us.txt"
    cn_calendar_path: str = "data/calendar/cn.txt"
    us_weekly_calendar_path: str = "data/calendar/us_weekly.txt"
    cn_weekly_calendar_path: str = "data/calendar/cn_weekly.txt"

    # Index paths
    index_dir: str = "data/instruments"
    us_index_path: str = "data/instruments/us.txt"
    csi300_index_path: str = "data/instruments/csi300.txt"
    csi500_index_path: str = "data/instruments/csi500.txt"

    # Stock data paths
    stock_data_dir: str = "data/stock_data"
    stock_weekly_data_dir: str = "data/stock_weekly_data"
    us_stock_data_dir: str = "data/stock_data/us_data"
    cn_stock_data_dir: str = "data/stock_data/cn_data"
    us_stock_weekly_data_dir: str = "data/stock_weekly_data/us_data"
    cn_stock_weekly_data_dir: str = "data/stock_weekly_data/cn_data"

    # Normalized data paths
    normalized_data_dir: str = "data/normalized_data"
    normalized_weekly_data_dir: str = "data/normalized_weekly_data"
    us_normalized_data_dir: str = "data/normalized_data/us_data"
    cn_normalized_data_dir: str = "data/normalized_data/cn_data"
    us_normalized_weekly_data_dir: str = "data/normalized_weekly_data/us_data"
    cn_normalized_weekly_data_dir: str = "data/normalized_weekly_data/cn_data"

    # MongoDB settings
    mongodb_url: str = Field(default='mongodb://localhost:27017')
    database_name: str = Field(default='predixlab', min_length=1)
    jobs_collection: str = Field(default='jobs', min_length=1)

    # Azure Communication Service email settings
    acs_connection_string: str = Field(default='')
    acs_sender_email: str = Field(default='')
    acs_to_emails: str = Field(default='')  # Comma-separated list of recipient emails

    @field_validator('mongodb_url')
    @classmethod
    def validate_mongodb_url(cls, v: str) -> str:
        """Validate MongoDB URL format."""
        if v and not v.startswith(('mongodb://', 'mongodb+srv://')):
            raise ValueError('MongoDB URL must start with mongodb:// or mongodb+srv://')
        return v

    @field_validator('acs_sender_email')
    @classmethod
    def validate_sender_email(cls, v: str) -> str:
        """Validate sender email format."""
        if v and '@' not in v:
            raise ValueError('Invalid sender email format')
        return v

    @field_validator('acs_to_emails')
    @classmethod
    def validate_to_emails(cls, v: str) -> str:
        """Validate recipient emails format."""
        if v:
            emails = [e.strip() for e in v.split(',') if e.strip()]
            for email in emails:
                if '@' not in email:
                    raise ValueError(f'Invalid email format: {email}')
        return v

    def get_to_emails_list(self) -> List[str]:
        """Get recipient emails as a list.
        
        Returns:
            List of validated email addresses.
        """
        if not self.acs_to_emails:
            return []
        return [email.strip() for email in self.acs_to_emails.split(',') if email.strip()]

    def is_email_configured(self) -> bool:
        """Check if email notification is properly configured.
        
        Returns:
            True if all required email settings are present.
        """
        return bool(
            self.acs_connection_string and
            self.acs_sender_email and
            self.acs_to_emails
        )

    def ensure_directories_exist(self) -> None:
        """Create all required directories if they don't exist."""
        directories = [
            self.calendar_dir,
            self.index_dir,
            self.stock_data_dir,
            self.stock_weekly_data_dir,
            self.us_stock_data_dir,
            self.cn_stock_data_dir,
            self.us_stock_weekly_data_dir,
            self.cn_stock_weekly_data_dir,
            self.normalized_data_dir,
            self.normalized_weekly_data_dir,
            self.us_normalized_data_dir,
            self.cn_normalized_data_dir,
            self.us_normalized_weekly_data_dir,
            self.cn_normalized_weekly_data_dir,
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        logger.debug("All required directories verified/created")

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


try:
    settings = Settings()
    logger.info("Configuration loaded successfully")
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    raise
