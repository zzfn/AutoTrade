"""
Configuration management for AutoTrade.

Centralizes environment variable loading and provides defaults.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env file
load_dotenv()


class Config:
    """
    Central configuration class.
    
    Loads settings from environment variables with sensible defaults.
    """
    
    # Base paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    AUTOTRADE_DIR: Path = PROJECT_ROOT / "autotrade"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    DATA_DIR: Path = PROJECT_ROOT / "data"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    
    # Database connection for state persistence
    # Default to SQLite in data directory
    DB_CONNECTION_STR: str = os.getenv(
        "DB_CONNECTION_STR", 
        f"sqlite:///{DATA_DIR / 'trade_state.sqlite'}"
    )
    
    # Alpaca API credentials
    ALPACA_API_KEY: Optional[str] = os.getenv("ALPACA_API_KEY")
    ALPACA_API_SECRET: Optional[str] = os.getenv("ALPACA_API_SECRET")
    ALPACA_IS_PAPER: bool = os.getenv("ALPACA_IS_PAPER", "true").lower() == "true"
    
    # Web server settings
    WEB_HOST: str = os.getenv("WEB_HOST", "0.0.0.0")
    WEB_PORT: int = int(os.getenv("WEB_PORT", "8000"))
    
    # Strategy settings
    DEFAULT_MODEL_NAME: str = os.getenv("DEFAULT_MODEL_NAME", "deepalaph")
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    
    @classmethod
    def get_alpaca_config(cls) -> dict:
        """Get Alpaca broker configuration."""
        return {
            "API_KEY": cls.ALPACA_API_KEY,
            "API_SECRET": cls.ALPACA_API_SECRET,
            "PAPER": cls.ALPACA_IS_PAPER,
        }
    
    @classmethod
    def get_db_connection_str(cls) -> str:
        """Get database connection string for state persistence."""
        return cls.DB_CONNECTION_STR
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure all required directories exist."""
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)


# Global config instance
config = Config()

# Ensure directories on import
config.ensure_directories()
