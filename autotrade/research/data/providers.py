"""
数据提供者模块 - 从 Alpaca 获取美股历史数据

使用 alpaca-py 作为唯一数据源
"""

import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from alpaca.data import StockHistoricalDataClient
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from loguru import logger


class BaseDataProvider(ABC):
    """数据提供者基类"""

    @abstractmethod
    def fetch_data(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        获取历史数据

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            interval: 频率 ('1d' 或 '1h')

        Returns:
            包含 OHLCV 数据的 DataFrame，MultiIndex: (datetime, symbol)
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """检查数据源是否可用"""
        pass


class AlpacaDataProvider(BaseDataProvider):
    """
    Alpaca 数据提供者 - 主要数据源

    从 Alpaca API 获取美股历史 OHLCV 数据
    """

    def __init__(self, api_key: str | None = None, secret_key: str | None = None):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_API_SECRET")
        self._client: Optional[StockHistoricalDataClient] = None

    def _get_client(self) -> StockHistoricalDataClient:
        """获取或创建 Alpaca 客户端"""
        if self._client is None:
            self._client = StockHistoricalDataClient(self.api_key, self.secret_key)
        return self._client

    def is_available(self) -> bool:
        """检查 Alpaca API 是否可用"""
        if not self.api_key or not self.secret_key:
            return False
        try:
            client = self._get_client()
            # 尝试获取一小段数据来验证连接
            request = StockBarsRequest(
                symbol_or_symbols=["SPY"],
                timeframe=TimeFrame.Day,
                start=datetime.now() - timedelta(days=5),
                end=datetime.now() - timedelta(days=1),
                feed=DataFeed.IEX,  # 免费账户使用 IEX 数据源
            )
            client.get_stock_bars(request)
            return True
        except Exception as e:
            logger.warning(f"Alpaca API 不可用: {e}")
            return False

    def fetch_data(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        从 Alpaca 获取历史数据

        Returns:
            DataFrame with columns: open, high, low, close, volume
            MultiIndex: (datetime, symbol)
        """
        logger.info(f"从 Alpaca 获取数据: {symbols}, {start_date} - {end_date}, interval={interval}")

        client = self._get_client()
        
        # 映射 interval 到 Alpaca TimeFrame
        if interval == "1min":
            tf = TimeFrame.Minute
        elif interval == "1h":
            tf = TimeFrame.Hour
        else:
            tf = TimeFrame.Day

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=tf,
            start=start_date,
            end=end_date,
            feed=DataFeed.IEX,  # 免费账户使用 IEX 数据源
        )

        bars = client.get_stock_bars(request)
        df = bars.df

        if df.empty:
            logger.warning("Alpaca 返回空数据")
            return pd.DataFrame()

        # 重命名列以匹配标准格式
        df = df.rename(
            columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            }
        )

        # 只保留需要的列
        df = df[["open", "high", "low", "close", "volume"]]

        # 确保索引格式正确 (datetime, symbol)
        if isinstance(df.index, pd.MultiIndex):
            # Alpaca 返回的是 (symbol, datetime)，需要交换
            df = df.reset_index()
            if "symbol" in df.columns and "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
                df = df.set_index(["timestamp", "symbol"])
                df = df.sort_index()

        logger.info(f"从 Alpaca 获取了 {len(df)} 条记录")
        return df


class DataProviderFactory:
    """
    数据提供者工厂 - 返回 AlpacaDataProvider

    要求配置有效的 Alpaca API 密钥
    """

    @staticmethod
    def get_provider() -> BaseDataProvider:
        """获取数据提供者（仅支持 Alpaca）"""
        alpaca = AlpacaDataProvider()
        if alpaca.is_available():
            logger.info("使用 Alpaca 作为数据源")
            return alpaca

        raise RuntimeError(
            "Alpaca 数据源不可用。请确保已设置 ALPACA_API_KEY 和 ALPACA_API_SECRET 环境变量。"
        )

