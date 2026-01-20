"""
Core infrastructure module.

This module contains shared utilities:
- config: Configuration loading (DB_CONNECTION_STR, etc.)
"""

from .config import Config, config

__all__ = ["Config", "config"]
