"""
Qlib 数据适配器 - 将数据转换为 Qlib 格式

任务 1.1 & 1.4: 实现 QlibDataAdapter 和 QlibDataConverter
"""

import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from .providers import BaseDataProvider, DataProviderFactory


class QlibDataAdapter:
    """
    Qlib 数据适配器

    负责：
    1. 从数据提供者获取原始数据
    2. 转换为 Qlib 格式
    3. 存储到 data/qlib/ 目录
    4. 支持增量数据更新
    """

    def __init__(
        self,
        data_dir: str | Path = "data/qlib",
        provider: BaseDataProvider | None = None,
        interval: str = "1d",
    ):
        self.interval = interval
        self.base_dir = Path(data_dir)
        self.data_dir = self.base_dir / interval
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 子目录结构
        self.instruments_dir = self.data_dir / "instruments"
        self.features_dir = self.data_dir / "features"
        self.calendars_dir = self.data_dir / "calendars"

        for dir_path in [self.instruments_dir, self.features_dir, self.calendars_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # 数据提供者
        self._provider = provider

    @property
    def provider(self) -> BaseDataProvider:
        """获取数据提供者（懒加载）"""
        if self._provider is None:
            self._provider = DataProviderFactory.get_provider()
        return self._provider

    def fetch_and_store(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        update_mode: str = "replace",
    ) -> dict:
        """
        获取数据并存储为 Qlib 格式

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            update_mode: 更新模式 - 'replace' 替换 / 'append' 追加

        Returns:
            包含处理结果的字典
        """
        logger.info(f"开始获取数据 ({self.interval}): {symbols}, {start_date} - {end_date}")

        # 1. 获取原始数据
        df = self.provider.fetch_data(symbols, start_date, end_date, interval=self.interval)

        if df.empty:
            return {"status": "error", "message": "未获取到数据"}

        # 2. 转换为 Qlib 格式
        result = self._convert_and_store(df, update_mode)

        # 3. 更新日历和工具列表
        self._update_calendar(df)
        self._update_instruments(symbols, start_date, end_date)

        return result

    def _convert_and_store(self, df: pd.DataFrame, update_mode: str) -> dict:
        """
        将 DataFrame 转换为 Qlib 格式并存储

        Qlib 格式要求：
        - 每个股票一个目录
        - 每个特征一个 .bin 文件
        - 数据按时间排序
        """
        processed_symbols = []

        # 获取所有 symbol
        if isinstance(df.index, pd.MultiIndex):
            symbols = df.index.get_level_values("symbol").unique()
        else:
            symbols = df["symbol"].unique() if "symbol" in df.columns else []

        for symbol in symbols:
            try:
                # 提取单个股票的数据
                if isinstance(df.index, pd.MultiIndex):
                    symbol_df = df.xs(symbol, level="symbol")
                else:
                    symbol_df = df[df["symbol"] == symbol].copy()
                    symbol_df = symbol_df.set_index("timestamp")

                # 确保时间排序
                symbol_df = symbol_df.sort_index()

                # 存储到 Qlib 格式
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
        存储单个股票的数据

        Qlib 格式：
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

        # 追加模式：加载现有数据并合并
        if update_mode == "append" and dates_file.exists():
            # 加载现有数据为 DataFrame
            with open(dates_file, "rb") as f:
                existing_dates = pickle.load(f)

            existing_data = {"timestamp": pd.to_datetime(existing_dates)}
            for col in feature_cols:
                feature_file = symbol_dir / f"${col}.bin"
                if feature_file.exists():
                    existing_data[col] = np.fromfile(feature_file, dtype=np.float32)

            # 构建现有数据 DataFrame
            if len(existing_data.get("open", [])) == len(existing_dates):
                old_df = pd.DataFrame(existing_data).set_index("timestamp")
                # 准备新数据
                new_df = df[feature_cols].copy()
                new_df.index = pd.to_datetime(new_df.index)
                # 合并：concat + 去重，保留新数据 (keep='last')
                combined = pd.concat([old_df, new_df]).sort_index()
                combined = combined[~combined.index.duplicated(keep="last")]
                df = combined

        # 保存日期
        with open(dates_file, "wb") as f:
            pickle.dump(df.index.tolist(), f)

        # 保存每个特征
        for col in feature_cols:
            if col in df.columns:
                feature_file = symbol_dir / f"${col}.bin"
                df[col].values.astype(np.float32).tofile(feature_file)

        logger.debug(f"存储 {symbol} 数据完成")

    def _update_calendar(self, df: pd.DataFrame) -> None:
        """更新交易日历"""
        if isinstance(df.index, pd.MultiIndex):
            dates = df.index.get_level_values(0).unique()
        else:
            dates = df.index.unique()

        # 根据频率选择日历文件名
        cal_name = "day.txt" if self.interval == "1d" else "hour.txt"
        calendar_file = self.calendars_dir / cal_name

        # 加载现有日历
        existing_dates = set()
        if calendar_file.exists():
            with open(calendar_file, "r") as f:
                existing_dates = set(line.strip() for line in f)

        # 合并并排序
        format_str = "%Y-%m-%d" if self.interval == "1d" else "%Y-%m-%d %H:%M:%S"
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
        """更新股票列表"""
        instruments_file = self.instruments_dir / "all.txt"

        # 加载现有列表
        existing_instruments = {}
        if instruments_file.exists():
            with open(instruments_file, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        existing_instruments[parts[0]] = (parts[1], parts[2])

        # 更新
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        for symbol in symbols:
            if symbol in existing_instruments:
                # 扩展日期范围
                old_start, old_end = existing_instruments[symbol]
                new_start = min(old_start, start_str)
                new_end = max(old_end, end_str)
                existing_instruments[symbol] = (new_start, new_end)
            else:
                existing_instruments[symbol] = (start_str, end_str)

        # 写入
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
        加载 Qlib 格式的数据

        Args:
            symbols: 股票代码列表
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）

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

            # 加载日期索引
            dates_file = symbol_dir / "_dates.pkl"
            if not dates_file.exists():
                continue

            with open(dates_file, "rb") as f:
                dates = pickle.load(f)

            # 加载特征
            data = {"timestamp": dates}
            lengths = [len(dates)]
            for col in ["open", "high", "low", "close", "volume"]:
                feature_file = symbol_dir / f"${col}.bin"
                if feature_file.exists():
                    values = np.fromfile(feature_file, dtype=np.float32)
                    data[col] = values
                    lengths.append(len(values))

            # 防御性处理：对齐长度，避免 DataFrame 构造失败
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

            # 应用日期过滤
            if start_date:
                df = df[df["timestamp"] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df["timestamp"] <= pd.to_datetime(end_date)]

            all_data.append(df)

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        result = result.set_index(["timestamp", "symbol"])
        # 去重以保证 (timestamp, symbol) 唯一，避免 unstack 报错
        if result.index.has_duplicates:
            result = result[~result.index.duplicated(keep="last")]
        result = result.sort_index()

        return result

    def get_available_symbols(self) -> list[str]:
        """获取所有可用的股票代码"""
        if not self.features_dir.exists():
            return []
        return [d.name for d in self.features_dir.iterdir() if d.is_dir()]

    def get_date_range(self, symbol: str) -> tuple[datetime, datetime] | None:
        """获取某只股票的数据日期范围"""
        symbol_dir = self.features_dir / symbol
        dates_file = symbol_dir / "_dates.pkl"

        if not dates_file.exists():
            return None

        with open(dates_file, "rb") as f:
            dates = pickle.load(f)

        if not dates:
            return None

        return (pd.to_datetime(min(dates)), pd.to_datetime(max(dates)))
