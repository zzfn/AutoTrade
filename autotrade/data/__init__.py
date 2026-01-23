"""
Data module.

This module contains data ingestion and transformation:
- qlib_adapter: DataFrame -> Qlib Feature mapping, data storage and loading
- providers: Alpaca data provider for fetching market data
"""

from .providers import BaseDataProvider, DataProviderFactory
from .qlib_adapter import (
    QlibDataAdapter,
    lumibot_to_qlib,
    prepare_feature_tensor,
)

__all__ = [
    # Data adapter
    "QlibDataAdapter",
    "lumibot_to_qlib",
    "prepare_feature_tensor",
    # Data providers
    "BaseDataProvider",
    "DataProviderFactory",
]
