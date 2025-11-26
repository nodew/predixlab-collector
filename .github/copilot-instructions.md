# Copilot Instructions for QStock Collector

## Project Overview

This is a Python-based stock market data collection and processing service that collects US and Chinese stock market data from Yahoo Finance and Eastmoney, and provides data normalization capabilities.

## Technology Stack

- **Language**: Python 3.12+
- **Package Manager**: uv (recommended) or pip
- **Testing**: pytest with pytest-mock and pytest-cov
- **Configuration**: pydantic-settings for configuration management
- **Logging**: loguru for structured logging
- **Database**: MongoDB for job status persistence
- **Notifications**: Azure Communication Service for email notifications

## Project Structure

```
qstock-collector/
├── collectors/                 # Data collector modules
│   ├── us_calendar/           # US trading calendar collector
│   ├── cn_calendar/           # Chinese trading calendar collector
│   ├── us_index/              # US index component collector (S&P 500 + NASDAQ 100)
│   ├── cn_index/              # Chinese index component collector (CSI 300 + CSI 500)
│   └── yahoo/                 # Yahoo Finance data collector and normalizer
├── tests/                     # Unit tests
├── config.py                  # Configuration settings
├── main.py                    # Main entry point (CLI)
├── notification.py            # Email notification service
├── utils.py                   # Utility functions
└── pyproject.toml            # Project configuration
```

## Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Write docstrings for all public functions and classes
- Keep functions focused and single-purpose

### Testing

- Write unit tests for all new functionality
- Place tests in the `tests/` directory
- Use pytest fixtures for test setup
- Mock external dependencies (Yahoo Finance API, MongoDB, etc.)
- Run tests with: `uv run pytest` or `python -m pytest tests/`

### Configuration

- Use environment variables for sensitive data (API keys, connection strings)
- Configuration is managed through `config.py` using pydantic-settings
- Default values should be reasonable for local development

### Error Handling

- Use loguru for logging errors and warnings
- Provide meaningful error messages
- Handle API rate limiting gracefully

## Common Commands

```bash
# Install dependencies
uv sync

# Install with test dependencies
uv sync --extra test

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=. --cov-report=html

# Run the main CLI
python main.py <command>
```

## Key Modules

### collectors/yahoo/collector.py
Handles downloading stock data from Yahoo Finance with batching and rate limiting.

### collectors/yahoo/normalize.py
Normalizes and standardizes collected stock data for analysis.

### collectors/us_index/collector.py
Collects S&P 500 and NASDAQ 100 index constituents from Wikipedia.

### collectors/cn_index/collector.py
Collects CSI 300 (沪深300) and CSI 500 (中证500) index constituents from Eastmoney.

### collectors/us_calendar/collector.py
Collects US trading calendar dates from Yahoo Finance.

### collectors/cn_calendar/collector.py
Collects Chinese trading calendar dates from Yahoo Finance.

### config.py
Central configuration using pydantic-settings with environment variable support.

### notification.py
Email notification service using Azure Communication Service.

## When Making Changes

1. Ensure all tests pass before committing
2. Add tests for new functionality
3. Update documentation if adding new features
4. Follow existing code patterns and conventions
5. Use meaningful commit messages
