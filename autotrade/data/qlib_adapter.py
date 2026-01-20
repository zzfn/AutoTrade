"""
Qlib Data Adapter Module.

Transforms Lumibot OHLCV DataFrame into Qlib-compatible Feature Tensor.

Task: Implement data/qlib_adapter.py - Function to transform Lumibot OHLCV DataFrame 
into Qlib-compatible Feature Tensor
"""

from typing import Optional, Union

import numpy as np
import pandas as pd
from loguru import logger


class QlibDataAdapter:
    """
    Adapter for converting Lumibot data format to Qlib-compatible format.
    
    Lumibot's `get_historical_prices` returns a Bars object with a DataFrame
    that may have different column naming conventions and index structures.
    
    This adapter ensures the data is properly formatted for Qlib feature generation
    and model inference.
    """
    
    # Standard Qlib column names (lowercase)
    STANDARD_COLUMNS = ["open", "high", "low", "close", "volume"]
    
    # Common column name mappings
    COLUMN_MAPPINGS = {
        # Uppercase variants
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        # Alternative names
        "OPEN": "open",
        "HIGH": "high",
        "LOW": "low",
        "CLOSE": "close",
        "VOLUME": "volume",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "adj_close": "close",
        "Adj Close": "close",
    }
    
    def __init__(self, fill_missing: bool = True, validate: bool = True):
        """
        Initialize the adapter.
        
        Args:
            fill_missing: Whether to forward-fill missing values.
            validate: Whether to validate the output DataFrame.
        """
        self.fill_missing = fill_missing
        self.validate = validate
    
    def transform(self, df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Transform a DataFrame from Lumibot format to Qlib format.
        
        Args:
            df: Input DataFrame from Lumibot's get_historical_prices.
            symbol: Optional symbol name (for logging/debugging).
        
        Returns:
            DataFrame with standardized column names and clean data.
        """
        if df is None or df.empty:
            logger.warning(f"Empty DataFrame received{f' for {symbol}' if symbol else ''}")
            return pd.DataFrame()
        
        # Make a copy to avoid modifying original
        result = df.copy()
        
        # Rename columns to standard format
        result = self._standardize_columns(result)
        
        # Handle missing values
        if self.fill_missing:
            result = self._handle_missing(result)
        
        # Validate if requested
        if self.validate:
            self._validate_dataframe(result, symbol)
        
        return result
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns to standard lowercase OHLCV format."""
        rename_map = {}
        
        for col in df.columns:
            if col in self.COLUMN_MAPPINGS:
                rename_map[col] = self.COLUMN_MAPPINGS[col]
            elif col.lower() in self.STANDARD_COLUMNS:
                rename_map[col] = col.lower()
        
        if rename_map:
            df = df.rename(columns=rename_map)
        
        return df
    
    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the DataFrame."""
        # Forward fill for OHLCV data
        df = df.ffill()
        
        # Drop any remaining NaN rows (typically at the start)
        df = df.dropna(subset=["close"])
        
        return df
    
    def _validate_dataframe(self, df: pd.DataFrame, symbol: Optional[str] = None) -> bool:
        """Validate that the DataFrame has required columns."""
        missing = [col for col in self.STANDARD_COLUMNS if col not in df.columns]
        
        if missing:
            logger.warning(f"Missing columns{f' for {symbol}' if symbol else ''}: {missing}")
            return False
        
        if df.empty:
            logger.warning(f"DataFrame is empty{f' for {symbol}' if symbol else ''}")
            return False
        
        return True
    
    def transform_multi(
        self, 
        data_dict: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """
        Transform multiple DataFrames at once.
        
        Args:
            data_dict: Dictionary mapping symbol -> DataFrame.
        
        Returns:
            Dictionary mapping symbol -> transformed DataFrame.
        """
        results = {}
        
        for symbol, df in data_dict.items():
            try:
                transformed = self.transform(df, symbol=symbol)
                if not transformed.empty:
                    results[symbol] = transformed
            except Exception as e:
                logger.error(f"Failed to transform data for {symbol}: {e}")
        
        return results


def lumibot_to_qlib(
    bars_or_df: Union[pd.DataFrame, "Bars"],  # noqa: F821 - Bars is from lumibot
    symbol: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience function to transform Lumibot Bars/DataFrame to Qlib format.
    
    Args:
        bars_or_df: Either a Lumibot Bars object or a DataFrame.
        symbol: Optional symbol for logging.
    
    Returns:
        Qlib-compatible DataFrame.
    
    Example:
        >>> history = strategy.get_historical_prices("AAPL", 60, "day")
        >>> df = lumibot_to_qlib(history)
        >>> features = feature_generator.generate(df)
    """
    # Handle Lumibot Bars object
    if hasattr(bars_or_df, 'df'):
        df = bars_or_df.df
    else:
        df = bars_or_df
    
    adapter = QlibDataAdapter()
    return adapter.transform(df, symbol=symbol)


def prepare_feature_tensor(
    df: pd.DataFrame,
    feature_columns: Optional[list[str]] = None,
) -> np.ndarray:
    """
    Convert a feature DataFrame to a numpy tensor for model input.
    
    Args:
        df: DataFrame with computed features.
        feature_columns: List of columns to include. If None, uses all columns.
    
    Returns:
        2D numpy array suitable for model prediction.
    """
    if feature_columns:
        df = df[feature_columns]
    
    # Replace any remaining NaN/inf values
    tensor = df.values.astype(np.float32)
    tensor = np.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    
    return tensor
