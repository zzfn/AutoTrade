"""
Qlib Data Adapter Module.

Transforms Lumibot OHLCV DataFrame into Qlib-compatible Feature Tensor.
Provides data management and storage capabilities for ML training.

Storage Format: Parquet
- File naming: {SYMBOL}_{INTERVAL}.parquet (e.g., AAPL_5MIN.parquet)
- Directory: datasets/
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from loguru import logger

from .providers import BaseDataProvider, DataProviderFactory


class QlibDataAdapter:
    """
    Adapter for converting Lumibot data format to Qlib-compatible format.

    Combines two functionalities:
    1. Data transformation (format conversion)
    2. Data management (fetch, store, load)

    Storage Format (Parquet):
    - datasets/AAPL_5MIN.parquet
    - datasets/MSFT_5MIN.parquet
    - datasets/AAPL_1D.parquet

    Lumibot's `get_historical_prices` returns a Bars object with a DataFrame
    that may have different column naming conventions and index structures.
    """

    # Standard Qlib column names (lowercase)
    STANDARD_COLUMNS = ["open", "high", "low", "close", "volume"]

    # Interval mapping for filename
    INTERVAL_MAP = {
        "1min": "1MIN",
        "5min": "5MIN",
        "15min": "15MIN",
        "30min": "30MIN",
        "1h": "1H",
        "4h": "4H",
        "1d": "1D",
    }

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

    def __init__(
        self,
        data_dir: Union[str, Path] = "datasets",
        provider: Optional[BaseDataProvider] = None,
        interval: str = "1d",
        fill_missing: bool = True,
        validate: bool = True,
    ):
        """
        Initialize the adapter.

        Args:
            data_dir: Directory for data storage (used for fetch/store/load operations)
            provider: Data provider instance (auto-created if None)
            interval: Data frequency ('1d', '1h', '1min', '5min') - 用于文件命名
            fill_missing: Whether to forward-fill missing values (for transform)
            validate: Whether to validate the output DataFrame (for transform)
        """
        # Data management attributes
        self.interval = interval
        self.data_dir = Path(data_dir)
        # 不再自动创建目录，只在需要时创建

        # Data provider
        self._provider = provider

        # Transformation attributes
        self.fill_missing = fill_missing
        self.validate = validate

        # Get interval suffix for filename
        self.interval_suffix = self.INTERVAL_MAP.get(interval, interval.upper())

    @property
    def provider(self) -> BaseDataProvider:
        """Get data provider (lazy loading)"""
        if self._provider is None:
            self._provider = DataProviderFactory.get_provider()
        return self._provider

    # ========== Data Management Methods (migrated from research.data) ==========

    def fetch_and_store(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        update_mode: str = "replace",
    ) -> dict:
        """
        Fetch data and store in Qlib format.

        Args:
            symbols: Stock symbol list
            start_date: Start date
            end_date: End date
            update_mode: Update mode - 'replace' or 'append'

        Returns:
            Dict with processing results
        """
        logger.info(f"开始获取数据 ({self.interval}): {symbols}, {start_date} - {end_date}")

        # 1. Fetch raw data
        df = self.provider.fetch_data(symbols, start_date, end_date, interval=self.interval)

        if df.empty:
            return {"status": "error", "message": "未获取到数据"}

        # 显示实际获取到的数据范围（带时区）
        if isinstance(df.index, pd.MultiIndex):
            actual_start = df.index.get_level_values("timestamp").min()
            actual_end = df.index.get_level_values("timestamp").max()
        else:
            actual_start = df.index.min()
            actual_end = df.index.max()

        # 格式化时间显示时区
        start_str = actual_start.strftime("%Y-%m-%d %H:%M:%S %Z")
        end_str = actual_end.strftime("%Y-%m-%d %H:%M:%S %Z")
        logger.info(f"实际获取到数据范围: {start_str} - {end_str}, 共 {len(df)} 条记录")

        # 2. Convert and store
        result = self._convert_and_store(df, update_mode)

        return result

    def _convert_and_store(self, df: pd.DataFrame, update_mode: str) -> dict:
        """
        Convert DataFrame to Qlib format and store as Parquet.

        Parquet format:
        - One file per stock: {SYMBOL}_{INTERVAL}.parquet
        - Data sorted by time
        """
        processed_symbols = []

        # Get all symbols
        if isinstance(df.index, pd.MultiIndex):
            symbols = df.index.get_level_values("symbol").unique()
        else:
            symbols = df["symbol"].unique() if "symbol" in df.columns else []

        for symbol in symbols:
            try:
                # Extract single stock data
                if isinstance(df.index, pd.MultiIndex):
                    symbol_df = df.xs(symbol, level="symbol")
                else:
                    symbol_df = df[df["symbol"] == symbol].copy()
                    symbol_df = symbol_df.set_index("timestamp")

                # Ensure time sorting
                symbol_df = symbol_df.sort_index()

                # Store in Parquet format
                self._store_symbol_data(symbol, symbol_df, update_mode)
                processed_symbols.append(symbol)

            except Exception as e:
                logger.error(f"处理 {symbol} 失败: {e}")

        return {
            "status": "success",
            "processed_symbols": processed_symbols,
            "total_records": len(df),
        }

    def _store_symbol_data(
        self, symbol: str, df: pd.DataFrame, update_mode: str
    ) -> None:
        """
        Store single stock data in Parquet format.

        File format: {SYMBOL}_{INTERVAL}.parquet (e.g., AAPL_5MIN.parquet)

        Args:
            symbol: Stock symbol
            df: DataFrame with OHLCV data
            update_mode: 'replace' or 'append'
        """
        # 确保目录存在（只在真正存储数据时创建）
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename: AAPL_5MIN.parquet
        filename = f"{symbol}_{self.interval_suffix}.parquet"
        filepath = self.data_dir / filename

        # Ensure index is named
        if df.index.name is None:
            df.index.name = "timestamp"

        # 如果有时区信息，转换为 UTC 并保留（parquet 会自动去掉时区，但我们会记住）
        has_tz = df.index.tz is not None
        if has_tz:
            # 转换为 UTC 并去掉时区（存储为 naive datetime，但实际是 UTC）
            df_to_store = df.copy()
            df_to_store.index = df_to_store.index.tz_convert("UTC").tz_localize(None)
        else:
            df_to_store = df

        # Append mode: load existing data and merge
        if update_mode == "append" and filepath.exists():
            try:
                # Load existing data（作为 UTC）
                old_df = pd.read_parquet(filepath)
                old_df.index = old_df.index.tz_localize("UTC")

                # Prepare new data
                new_df = df.copy()
                if new_df.index.tz is None:
                    new_df.index = new_df.index.tz_localize("UTC")
                else:
                    new_df.index = new_df.index.tz_convert("UTC")

                # Merge: concat + deduplicate, keep new data (keep='last')
                combined = pd.concat([old_df, new_df]).sort_index()
                combined = combined[~combined.index.duplicated(keep="last")]

                # 去掉时区后存储
                df_to_store = combined.copy()
                df_to_store.index = df_to_store.index.tz_localize(None)

            except Exception as e:
                logger.warning(f"加载现有数据失败，将覆盖: {e}")

        # Save to Parquet（存储为 naive datetime，实际是 UTC）
        df_to_store.to_parquet(filepath, index=True)
        logger.debug(f"存储 {symbol} 数据完成: {filepath}")

    def load_data(
        self,
        symbols: list[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Load Parquet format data.

        Args:
            symbols: Stock symbol list
            start_date: Start date (optional)
            end_date: End date (optional)

        Returns:
            DataFrame with columns: open, high, low, close, volume
            MultiIndex: (datetime, symbol)
        """
        all_data = []

        for symbol in symbols:
            # Generate filename: AAPL_5MIN.parquet
            filename = f"{symbol}_{self.interval_suffix}.parquet"
            filepath = self.data_dir / filename

            if not filepath.exists():
                logger.warning(f"未找到 {symbol} 的数据: {filepath}")
                continue

            try:
                # Load from Parquet（添加 UTC 时区）
                df = pd.read_parquet(filepath)
                # 将 naive datetime 视为 UTC
                df.index = df.index.tz_localize("UTC")

                # Add symbol column
                df["symbol"] = symbol

                # Apply date filter
                if start_date:
                    start_dt = pd.to_datetime(start_date)
                    if start_dt.tz is None:
                        start_dt = start_dt.tz_localize("UTC")
                    df = df[df.index >= start_dt]
                if end_date:
                    end_dt = pd.to_datetime(end_date)
                    if end_dt.tz is None:
                        end_dt = end_dt.tz_localize("UTC")
                    df = df[df.index <= end_dt]

                all_data.append(df)

            except Exception as e:
                logger.error(f"加载 {symbol} 数据失败: {e}")

        if not all_data:
            return pd.DataFrame()

        # Combine all symbols
        result = pd.concat(all_data, ignore_index=False)

        # Set MultiIndex (timestamp, symbol)
        result = result.reset_index()
        if "timestamp" not in result.columns:
            # Assume index is already timestamp
            result = result.rename_axis("timestamp").reset_index()

        result = result.set_index(["timestamp", "symbol"])

        # Deduplicate to ensure (timestamp, symbol) uniqueness
        if result.index.has_duplicates:
            result = result[~result.index.duplicated(keep="last")]
        result = result.sort_index()

        return result

    def get_available_symbols(self) -> list[str]:
        """Get all available stock symbols from Parquet files"""
        if not self.data_dir.exists():
            return []

        symbols = []
        for file in self.data_dir.glob("*.parquet"):
            # Extract symbol from filename: AAPL_5MIN.parquet -> AAPL
            parts = file.stem.split("_")
            if len(parts) >= 2:
                symbols.append(parts[0])

        return sorted(set(symbols))

    def get_date_range(self, symbol: str) -> Optional[tuple[datetime, datetime]]:
        """Get date range for a stock (returns UTC timezone-aware datetimes)"""
        filename = f"{symbol}_{self.interval_suffix}.parquet"
        filepath = self.data_dir / filename

        if not filepath.exists():
            return None

        try:
            df = pd.read_parquet(filepath)
            if df.empty:
                return None

            # 添加 UTC 时区
            df.index = df.index.tz_localize("UTC")
            return (df.index.min(), df.index.max())

        except Exception as e:
            logger.error(f"获取 {symbol} 日期范围失败: {e}")
            return None

    # ========== Data Transformation Methods (original functionality) ==========

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

        # Standardize column names
        result = self._standardize_columns(result)

        # Handle missing values
        if self.fill_missing:
            result = self._handle_missing(result)

        # Validate
        if self.validate:
            if not self._validate_dataframe(result, symbol):
                logger.warning(f"DataFrame validation failed{f' for {symbol}' if symbol else ''}")
                return pd.DataFrame()

        return result

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to Qlib format"""
        # Rename columns based on mapping
        rename_dict = {
            col: self.COLUMN_MAPPINGS.get(col, col)
            for col in df.columns
            if col in self.COLUMN_MAPPINGS
        }
        if rename_dict:
            df = df.rename(columns=rename_dict)

        # Ensure all standard columns exist
        for col in self.STANDARD_COLUMNS:
            if col not in df.columns:
                # Try to find a similar column
                similar = [c for c in df.columns if c.lower() == col.lower()]
                if similar:
                    df = df.rename(columns={similar[0]: col})
                else:
                    logger.warning(f"Missing column: {col}")
                    # Fill with NaN
                    df[col] = np.nan

        # Keep only standard columns
        df = df[self.STANDARD_COLUMNS]

        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        # Forward fill
        df = df.ffill()

        # Backward fill remaining NaNs
        df = df.bfill()

        # Replace any remaining NaNs with 0
        df = df.fillna(0)

        # Replace inf with 0
        df = df.replace([np.inf, -np.inf], 0)

        return df

    def _validate_dataframe(self, df: pd.DataFrame, symbol: Optional[str] = None) -> bool:
        """Validate the output DataFrame"""
        # Check if empty
        if df.empty:
            logger.warning(f"DataFrame is empty{f' for {symbol}' if symbol else ''}")
            return False

        # Check columns
        if not all(col in df.columns for col in self.STANDARD_COLUMNS):
            missing = [col for col in self.STANDARD_COLUMNS if col not in df.columns]
            logger.warning(f"Missing columns: {missing}")
            return False

        # Check for all NaN columns
        for col in self.STANDARD_COLUMNS:
            if df[col].isna().all():
                logger.warning(f"Column {col} is all NaN")
                return False

        return True

    def transform_multi(
        self, df: pd.DataFrame, symbol_col: str = "symbol"
    ) -> pd.DataFrame:
        """
        Transform multi-stock DataFrame from Lumibot format to Qlib format.

        Args:
            df: Input MultiIndex DataFrame from Lumibot.
            symbol_col: Name of the symbol column (if not MultiIndex).

        Returns:
            DataFrame with standardized columns and clean data.
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame received")
            return pd.DataFrame()

        # Check if MultiIndex
        if isinstance(df.index, pd.MultiIndex):
            # Process each symbol separately
            results = []
            for symbol in df.index.get_level_values(symbol_col).unique():
                symbol_df = df.xs(symbol, level=symbol_col)
                transformed = self.transform(symbol_df, symbol)
                if not transformed.empty:
                    transformed[symbol_col] = symbol
                    results.append(transformed)

            if results:
                result = pd.concat(results, ignore_index=True)
                # Set MultiIndex
                if "timestamp" in result.columns:
                    result = result.set_index(["timestamp", symbol_col])
                elif isinstance(result.index, pd.DatetimeIndex):
                    result = result.reset_index()
                    result = result.set_index(["index", symbol_col])
                    result = result.rename_axis(["timestamp", symbol_col])
                return result

        return df


def lumibot_to_qlib(
    history, symbol: Optional[str] = None, fill_missing: bool = True
) -> pd.DataFrame:
    """
    Convenience function to transform Lumibot Bars/DataFrame to Qlib format.

    Args:
        history: Lumibot Bars object or DataFrame
        symbol: Optional symbol name for logging
        fill_missing: Whether to fill missing values

    Returns:
        DataFrame in Qlib format
    """
    adapter = QlibDataAdapter(fill_missing=fill_missing, validate=False)

    # Extract DataFrame if Bars object
    if hasattr(history, 'df'):
        df = history.df
    elif isinstance(history, pd.DataFrame):
        df = history
    else:
        logger.warning(f"Unsupported type: {type(history)}")
        return pd.DataFrame()

    return adapter.transform(df, symbol)


def prepare_feature_tensor(
    df: pd.DataFrame, feature_cols: Optional[list] = None
) -> np.ndarray:
    """
    Prepare feature tensor for model input.

    Args:
        df: DataFrame with OHLCV data
        feature_cols: Columns to include (default: all standard columns)

    Returns:
        numpy array of shape (n_samples, n_features)
    """
    if feature_cols is None:
        feature_cols = ["open", "high", "low", "close", "volume"]

    # Ensure all columns exist
    available_cols = [col for col in feature_cols if col in df.columns]

    if not available_cols:
        logger.warning("No valid feature columns found")
        return np.array([])

    # Extract features
    features = df[available_cols].values

    # Handle NaN/Inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features
