from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    calendar_dir: str = "data/calendar"
    us_calendar_path: str = "data/calendar/us.txt"
    cn_calendar_path: str = "data/calendar/cn.txt"

    index_dir: str = "data/instruments"
    us_index_path: str = "data/instruments/us.txt"
    cn_index_path: str = "data/instruments/cn.txt"

    stock_data_dir: str = "data/stock_data"
    us_stock_data_dir: str = "data/stock_data/us_data"
    cn_stock_data_dir: str = "data/stock_data/cn_data"

    normalized_data_dir: str = "data/normalized_data"
    us_normalized_data_dir: str = "data/normalized_data/us_data"
    cn_normalized_data_dir: str = "data/normalized_data/cn_data"

    # MongoDB settings
    mongodb_url: str = 'mongodb://localhost:27017'
    database_name: str = 'predixlab'
    jobs_collection: str = 'jobs'

    # ACS email settings
    acs_connection_string: str = ''
    acs_sender_email: str = ''
    acs_to_emails: str = ''  # Comma-separated list of recipient emails

    class Config:
        env_file = ".env"

settings = Settings()
