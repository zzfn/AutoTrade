"""
Alpaca 回测补丁 - 修复 Lumibot Alpaca 多分钟 timestep 的 bug

Lumibot 原版的问题：
1. AlpacaData 的 TIMESTEP_MAPPING 使用错误的格式（f"5{TimeFrame.Minute}"）
2. AlpacaBacktesting 只支持 "day" 和 "minute"，硬编码检查限制了多分钟间隔

这个模块提供了修复后的类，支持 5 分钟、15 分钟、30 分钟等多分钟间隔。
"""
from datetime import timedelta

import pandas as pd
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from lumibot.data_sources import AlpacaData
from lumibot.data_sources.data_source import DataSource
from lumibot.backtesting import AlpacaBacktesting


class FixedAlpacaData(AlpacaData):
    """
    修复了多分钟 timestep 的 AlpacaData。

    Lumibot 原版的 TIMESTEP_MAPPING 中 "5 minutes" 等多分钟间隔的映射存在 bug，
    导致时间步被错误解析（如 "5 minutes" 被解析为 "51Min" 而不是 "5Min"）。

    这个类重新定义了 TIMESTEP_MAPPING，使用正确的 TimeFrame 对象。
    """

    # 重新定义 TIMESTEP_MAPPING
    TIMESTEP_MAPPING = [
        {
            "timestep": "minute",
            "representations": [TimeFrame.Minute, "minute"],
        },
        {
            "timestep": "5 minutes",
            "representations": [TimeFrame(5, TimeFrameUnit.Minute), "5 minutes"],
        },
        {
            "timestep": "15 minutes",
            "representations": [TimeFrame(15, TimeFrameUnit.Minute), "15 minutes"],
        },
        {
            "timestep": "30 minutes",
            "representations": [TimeFrame(30, TimeFrameUnit.Minute), "30 minutes"],
        },
        {
            "timestep": "hour",
            "representations": [TimeFrame.Hour, "hour"],
        },
        {
            "timestep": "day",
            "representations": [TimeFrame.Day, "day"],
        },
    ]


def patch_alpaca_timeframe_mapping():
    """
    修复 AlpacaData 的 TIMESTEP_MAPPING 与解析逻辑，避免传入错误的 timeframe 类型。
    """
    AlpacaData.TIMESTEP_MAPPING = [
        {"timestep": "minute", "representations": [TimeFrame.Minute, "minute"]},
        {"timestep": "5 minutes", "representations": [TimeFrame(5, TimeFrameUnit.Minute), "5 minutes"]},
        {"timestep": "10 minutes", "representations": [TimeFrame(10, TimeFrameUnit.Minute), "10 minutes"]},
        {"timestep": "15 minutes", "representations": [TimeFrame(15, TimeFrameUnit.Minute), "15 minutes"]},
        {"timestep": "30 minutes", "representations": [TimeFrame(30, TimeFrameUnit.Minute), "30 minutes"]},
        {"timestep": "hour", "representations": [TimeFrame.Hour, "hour"]},
        {"timestep": "1 hour", "representations": [TimeFrame.Hour, "1 hour"]},
        {"timestep": "2 hours", "representations": [TimeFrame(2, TimeFrameUnit.Hour), "2 hours"]},
        {"timestep": "4 hours", "representations": [TimeFrame(4, TimeFrameUnit.Hour), "4 hours"]},
        {"timestep": "day", "representations": [TimeFrame.Day, "day"]},
    ]

    def _parse_source_timestep(self, timestep, reverse=False):
        result = DataSource._parse_source_timestep(self, timestep, reverse=reverse)
        if isinstance(result, (list, tuple)):
            for item in result:
                if isinstance(item, TimeFrame):
                    return item
            return result[0]
        return result

    AlpacaData._parse_source_timestep = _parse_source_timestep


class MyAlpacaBacktesting(AlpacaBacktesting):
    """
    修复了多分钟 timestep 支持的 AlpacaBacktesting。

    Lumibot 原版的 AlpacaBacktesting 只支持 "day" 和 "minute" 两种 timestep。
    这个类扩展了 TIMESTEP_MAPPING，支持 5 分钟、15 分钟、30 分钟等间隔。

    主要修复：
    1. 扩展 TIMESTEP_MAPPING 以支持多分钟间隔
    2. 移除硬编码的 timestep 检查限制
    3. 修复 _parse_source_timestep 方法的映射逻辑
    """

    # 扩展 TIMESTEP_MAPPING 以支持多分钟间隔
    TIMESTEP_MAPPING = [
        {"timestep": "day", "representations": [TimeFrame.Day]},
        {"timestep": "hour", "representations": [TimeFrame.Hour]},
        {"timestep": "30 minutes", "representations": [TimeFrame(30, TimeFrameUnit.Minute)]},
        {"timestep": "15 minutes", "representations": [TimeFrame(15, TimeFrameUnit.Minute)]},
        {"timestep": "5 minutes", "representations": [TimeFrame(5, TimeFrameUnit.Minute)]},
        {"timestep": "minute", "representations": [TimeFrame.Minute]},
    ]

    def __init__(self, *args, **kwargs):
        # 捕获 config 参数供后续使用
        self._config_kwargs = kwargs
        super().__init__(*args, **kwargs)

    def _download_and_cache_ohlcv_data(
            self,
            *,
            base_asset=None,
            quote_asset=None,
            timestep=None,
            market=None,
            tzinfo=None,
            data_datetime_start=None,
            data_datetime_end=None,
            auto_adjust=None,
    ):
        """
        重写此方法以移除硬编码的 timestep 检查限制。

        原方法中有 `if timestep not in ['day', 'minute']` 的检查，
        这里移除该检查，允许使用多分钟间隔。
        """
        if base_asset is None:
            raise ValueError("The parameter 'base_asset' cannot be None.")
        if quote_asset is None:
            quote_asset = self.LUMIBOT_DEFAULT_QUOTE_ASSET
        if timestep is None:
            timestep = self._timestep
        if market is None:
            market = self.market
        if tzinfo is None:
            tzinfo = self.tzinfo
        if data_datetime_start is None:
            data_datetime_start = self._data_datetime_start
        if data_datetime_end is None:
            data_datetime_end = self._data_datetime_end
        if auto_adjust is None:
            auto_adjust = self._auto_adjust

        key = self._get_asset_key(
            base_asset=base_asset,
            quote_asset=quote_asset,
            timestep=timestep,
            market=market,
            tzinfo=tzinfo,
            data_datetime_start=data_datetime_start,
            data_datetime_end=data_datetime_end,
            auto_adjust=auto_adjust,
        )

        # Directory to save cached data.
        import os
        from lumibot.constants import LUMIBOT_CACHE_FOLDER

        cache_dir = os.path.join(LUMIBOT_CACHE_FOLDER, self.CACHE_SUBFOLDER)
        os.makedirs(cache_dir, exist_ok=True)

        # File path based on the unique key
        filename = f"{key}.csv"
        filepath = os.path.join(cache_dir, filename)

        from lumibot.tools.lumibot_logger import get_logger
        logger = get_logger(__name__)
        logger.info(f"Fetching and caching data for {key}")

        if base_asset.asset_type == 'crypto':
            client = self._crypto_client
            symbol = base_asset.symbol + '/' + quote_asset.symbol
            from alpaca.data.requests import CryptoBarsRequest
            request_params = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=self._parse_source_timestep(timestep, reverse=True),
                start=data_datetime_start,
                end=data_datetime_end + timedelta(days=1),
            )
        else:
            client = self._stock_client
            adjustment = 'all' if auto_adjust else 'split'
            from alpaca.data.requests import StockBarsRequest
            request_params = StockBarsRequest(
                symbol_or_symbols=base_asset.symbol,
                timeframe=self._parse_source_timestep(timestep, reverse=True),
                start=data_datetime_start,
                end=data_datetime_end + timedelta(days=1),
                adjustment=adjustment,
            )

        try:
            if isinstance(request_params, CryptoBarsRequest):
                bars = client.get_crypto_bars(request_params)
            else:
                bars = client.get_stock_bars(request_params)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch data for {key}: {e}")

        df = bars.df.reset_index()
        if df.empty:
            raise RuntimeError(f"No data fetched for {key}.")

        # Ensure 'timestamp' is a pandas timestamp object
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize(tzinfo)
        else:
            df['timestamp'] = df['timestamp'].dt.tz_convert(tzinfo)

        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        from lumibot.tools.helpers import get_trading_times
        trading_times = get_trading_times(
            pcal=self._trading_days,
            timestep=timestep,
        )

        # Reindex the dataframe with a row for each bar
        df = self._reindex_and_fill(df=df, trading_times=trading_times, timestep=timestep)

        # Filter data to include only rows between data_datetime_start and data_datetime_end
        df = df[(df['timestamp'] >= data_datetime_start) & (df['timestamp'] <= data_datetime_end)]

        # Save to cache
        df.to_csv(filepath, index=False)

        # Store in _data_store
        df.set_index('timestamp', inplace=True)
        self._data_store[key] = df
        logger.info(f"Finished fetching and caching data for {key}")
        return df
