"""
Qlib Data Adapter Module.

Transforms Lumibot OHLCV DataFrame into Qlib-compatible Feature Tensor.
Provides data management and storage capabilities for ML training.

Tasks:
- Implement data/qlib_adapter.py - Function to transform Lumibot OHLCV DataFrame
- Migrate from research.data.qlib_adapter
"""

import pickle
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

    Lumibot's `get_historical_prices` returns a Bars object with a DataFrame
    that may have different column naming conventions and index structures.

    This adapter ensures the data is properly formatted for Qlib feature generation
    and model inference, with support for persistent storage and incremental updates.
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
            interval: Data frequency ('1d', '1h', '1min', '5min')
            fill_missing: Whether to forward-fill missing values (for transform)
            validate: Whether to validate the output DataFrame (for transform)
        """
        # Data management attributes
        self.interval = interval
        self.base_dir = Path(data_dir)
        self.data_dir = self.base_dir / interval
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        self.instruments_dir = self.data_dir / "instruments"
        self.features_dir = self.data_dir / "features"
        self.calendars_dir = self.data_dir / "calendars"

        for dir_path in [self.instruments_dir, self.features_dir, self.calendars_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Data provider
        self._provider = provider

        # Transformation attributes
        self.fill_missing = fill_missing
        self.validate = validate

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

        # 2. Convert and store
        result = self._convert_and_store(df, update_mode)

        # 3. Update calendar and instruments
        self._update_calendar(df)
        self._update_instruments(symbols, start_date, end_date)

        return result

    def _convert_and_store(self, df: pd.DataFrame, update_mode: str) -> dict:
        """
        Convert DataFrame to Qlib format and store.

        Qlib format requirements:
        - One directory per stock
        - One .bin file per feature
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

                # Store in Qlib format
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
        Store single stock data.

        Qlib format:
        - features/{symbol}/$open.bin
        - features/{symbol}/$high.bin
        - features/{symbol}/$low.bin
        - features/{symbol}/$close.bin
        - features/{symbol}/$volume.bin
        """
        symbol_dir = self.features_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)

        feature_cols = ["open", "high", "low", "close", "volume"]
        dates_file = symbol_dir / "_dates.pkl"

        # Append mode: load existing data and merge
        if update_mode == "append" and dates_file.exists():
            # Load existing data as DataFrame
            with open(dates_file, "rb") as f:
                existing_dates = pickle.load(f)

            existing_data = {"timestamp": pd.to_datetime(existing_dates)}
            for col in feature_cols:
                feature_file = symbol_dir / f"${col}.bin"
                if feature_file.exists():
                    existing_data[col] = np.fromfile(feature_file, dtype=np.float32)

            # Build existing data DataFrame
            if len(existing_data.get("open", [])) == len(existing_dates):
                old_df = pd.DataFrame(existing_data).set_index("timestamp")
                # Prepare new data
                new_df = df[feature_cols].copy()
                new_df.index = pd.to_datetime(new_df.index)
                # Merge: concat + deduplicate, keep new data (keep='last')
                combined = pd.concat([old_df, new_df]).sort_index()
                combined = combined[~combined.index.duplicated(keep="last")]
                df = combined

        # Save dates
        with open(dates_file, "wb") as f:
            pickle.dump(df.index.tolist(), f)

        # Save each feature
        for col in feature_cols:
            if col in df.columns:
                feature_file = symbol_dir / f"${col}.bin"
                df[col].values.astype(np.float32).tofile(feature_file)

        logger.debug(f"存储 {symbol} 数据完成")

    def _update_calendar(self, df: pd.DataFrame) -> None:
        """Update trading calendar"""
        if isinstance(df.index, pd.MultiIndex):
            dates = df.index.get_level_values(0).unique()
        else:
            dates = df.index.unique()

        # Select calendar file name based on frequency
        if self.interval == "1d":
            cal_name = "day.txt"
            format_str = "%Y-%m-%d"
        elif self.interval == "1min":
            cal_name = "min.txt"
            format_str = "%Y-%m-%d %H:%M:%S"
        else:  # 1h or others
            cal_name = "hour.txt"
            format_str = "%Y-%m-%d %H:%M:%S"
        calendar_file = self.calendars_dir / cal_name

        # Load existing calendar
        existing_dates = set()
        if calendar_file.exists():
            with open(calendar_file, "r") as f:
                existing_dates = set(line.strip() for line in f)

        # Merge and sort
        all_dates = sorted(
            existing_dates
            | set(pd.to_datetime(d).strftime(format_str) for d in dates)
        )

        with open(calendar_file, "w") as f:
            f.write("\n".join(all_dates))

        logger.debug(f"更新日历 ({self.interval}): {len(all_dates)} 条记录")

    def _update_instruments(
        self, symbols: list[str], start_date: datetime, end_date: datetime
    ) -> None:
        """Update stock list"""
        instruments_file = self.instruments_dir / "all.txt"

        # Load existing list
        existing_instruments = {}
        if instruments_file.exists():
            with open(instruments_file, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        existing_instruments[parts[0]] = (parts[1], parts[2])

        # Update
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        for symbol in symbols:
            if symbol in existing_instruments:
                # Expand date range
                old_start, old_end = existing_instruments[symbol]
                new_start = min(old_start, start_str)
                new_end = max(old_end, end_str)
                existing_instruments[symbol] = (new_start, new_end)
            else:
                existing_instruments[symbol] = (start_str, end_str)

        # Write
        with open(instruments_file, "w") as f:
            for symbol, (s, e) in sorted(existing_instruments.items()):
                f.write(f"{symbol}\t{s}\t{e}\n")

        logger.debug(f"更新股票列表: {len(existing_instruments)} 只股票")

    def load_data(
        self,
        symbols: list[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Load Qlib format data.

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
            symbol_dir = self.features_dir / symbol
            if not symbol_dir.exists():
                logger.warning(f"未找到 {symbol} 的数据")
                continue

            # Load date index
            dates_file = symbol_dir / "_dates.pkl"
            if not dates_file.exists():
                continue

            with open(dates_file, "rb") as f:
                dates = pickle.load(f)

            # Load features
            data = {"timestamp": dates}
            lengths = [len(dates)]
            for col in ["open", "high", "low", "close", "volume"]:
                feature_file = symbol_dir / f"${col}.bin"
                if feature_file.exists():
                    values = np.fromfile(feature_file, dtype=np.float32)
                    data[col] = values
                    lengths.append(len(values))

            # Defensive handling: align lengths to avoid DataFrame construction failure
            min_len = min(lengths) if lengths else 0
            if min_len == 0:
                continue
            if len(set(lengths)) != 1:
                logger.warning(
                    f"{symbol} 数据长度不一致: {lengths}，将截断到 {min_len}"
                )
                data = {k: v[:min_len] for k, v in data.items()}

            df = pd.DataFrame(data)
            df["symbol"] = symbol
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Apply date filter
            if start_date:
                df = df[df["timestamp"] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df["timestamp"] <= pd.to_datetime(end_date)]

            all_data.append(df)

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        result = result.set_index(["timestamp", "symbol"])
        # Deduplicate to ensure (timestamp, symbol) uniqueness, avoid unstack error
        if result.index.has_duplicates:
            result = result[~result.index.duplicated(keep="last")]
        result = result.sort_index()

        return result

    def get_available_symbols(self) -> list[str]:
        """Get all available stock symbols"""
        if not self.features_dir.exists():
            return []
        return [d.name for d in self.features_dir.iterdir() if d.is_dir()]

    def get_date_range(self, symbol: str) -> Optional[tuple[datetime, datetime]]:
        """Get date range for a stock"""
        symbol_dir = self.features_dir / symbol
        dates_file = symbol_dir / "_dates.pkl"

        if not dates_file.exists():
            return None

        with open(dates_file, "rb") as f:
            dates = pickle.load(f)

        if not dates:
            return None

        return (pd.to_datetime(min(dates)), pd.to_datetime(max(dates)))

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
