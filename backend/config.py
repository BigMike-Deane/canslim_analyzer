"""
Configuration for CANSLIM Analyzer Web App
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(ENV_PATH)


class Settings:
    # App settings
    APP_NAME = "CANSLIM Analyzer"
    VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

    # Authentication (optional)
    AUTH_USERNAME = os.getenv("CANSLIM_AUTH_USERNAME", "")
    AUTH_PASSWORD = os.getenv("CANSLIM_AUTH_PASSWORD", "")

    @property
    def AUTH_ENABLED(self):
        return bool(self.AUTH_USERNAME and self.AUTH_PASSWORD)

    # CORS settings
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

    # Email settings (from existing .env)
    EMAIL_ADDRESS = os.getenv("CANSLIM_EMAIL", "")
    EMAIL_APP_PASSWORD = os.getenv("CANSLIM_APP_PASSWORD", "")
    EMAIL_RECIPIENT = os.getenv("CANSLIM_RECIPIENT", "")

    # Analysis settings
    DEFAULT_TOP_STOCKS = 20
    MAX_STOCKS_PER_REQUEST = 100
    SCORE_CACHE_HOURS = 4  # How long to cache scores before refreshing

    # Background job settings
    ANALYSIS_BATCH_SIZE = 50  # Process stocks in batches


settings = Settings()
