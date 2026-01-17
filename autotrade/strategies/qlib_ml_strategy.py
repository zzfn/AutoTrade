"""
Qlib ML 策略 - 基于机器学习模型驱动的交易策略

任务 4.1 - 4.5: 实现 QlibMLStrategy

此策略与 MomentumStrategy 类似，但使用 ML 模型预测收益率，
而不是传统技术指标（如 SMA）来驱动交易决策。
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from lumibot.strategies.strategy import Strategy

from autotrade.research.features import QlibFeatureGenerator
from autotrade.research.models import LightGBMTrainer, ModelManager


class QlibMLStrategy(Strategy):
    """
    Qlib ML 策略

    完全由 ML 模型预测驱动的交易策略：
    1. 获取候选股票的历史数据
    2. 生成特征
    3. 使用 ML 模型预测收益率
    4. 选择 Top-K 预测分数最高的股票
    5. 等权重分配资金
    6. 定期再平衡

    策略参数:
        symbols: 候选股票列表
        model_name: 使用的模型名称（如为空则使用当前模型）
        top_k: 持仓股票数量
        rebalance_period: 再平衡周期（天数）
        lookback_period: 获取历史数据的天数
        sleeptime: 交易迭代间隔
    """

    parameters = {
        "symbols": ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN"],
        "model_name": None,  # None 表示使用 ModelManager 的当前模型
        "top_k": 3,
        "rebalance_period": 1,  # 每天再平衡
        "lookback_period": 60,  # 获取 60 天历史数据
        "sleeptime": "1D",
        "models_dir": "models",
    }

    def initialize(self):
        """策略初始化"""
        # 解析参数
        self._parse_parameters()

        # 初始化组件
        self.feature_generator = QlibFeatureGenerator(normalize=True)
        self.model_manager = ModelManager(self.models_dir)
        self.trainer: Optional[LightGBMTrainer] = None

        # 状态跟踪
        self.last_rebalance_date: Optional[datetime] = None
        self.current_predictions: dict = {}

        # 加载模型
        self._load_model()

        self.log_message(f"QlibMLStrategy 初始化完成")
        self.log_message(f"股票池: {self.symbols}")
        self.log_message(f"Top-K: {self.top_k}")
        self.log_message(f"再平衡周期: {self.rebalance_period} 天")

    def _parse_parameters(self):
        """解析策略参数"""
        # 股票列表
        if "symbols" in self.parameters:
            self.symbols = self.parameters["symbols"]
        elif "symbol" in self.parameters:
            self.symbols = [self.parameters["symbol"]]
        else:
            self.symbols = ["SPY"]

        # 确保格式统一为列表
        if isinstance(self.symbols, str):
            if "," in self.symbols:
                self.symbols = [s.strip() for s in self.symbols.split(",")]
            else:
                self.symbols = [self.symbols]

        # 其他参数
        self.model_name = self.parameters.get("model_name")
        self.top_k = self.parameters.get("top_k", 3)
        self.rebalance_period = self.parameters.get("rebalance_period", 1)
        self.lookback_period = self.parameters.get("lookback_period", 60)
        self.sleeptime = self.parameters.get("sleeptime", "1D")
        self.models_dir = self.parameters.get("models_dir", "models")

        # 确保 top_k 不超过股票数量
        self.top_k = min(self.top_k, len(self.symbols))

    def _load_model(self):
        """加载 ML 模型"""
        try:
            # 确定模型路径
            if self.model_name:
                model_path = Path(self.models_dir) / self.model_name
            else:
                model_path = self.model_manager.get_current_model_path()

            if not model_path or not model_path.exists():
                self.log_message("警告: 未找到 ML 模型，将使用随机预测")
                return

            # 加载模型
            self.trainer = LightGBMTrainer(model_dir=self.models_dir)
            self.trainer.load(model_path)

            self.log_message(f"已加载模型: {model_path.name}")

        except Exception as e:
            self.log_message(f"加载模型失败: {e}")
            self.trainer = None

    def on_trading_iteration(self):
        """
        交易迭代 - 核心逻辑

        每次迭代：
        1. 检查是否需要再平衡
        2. 获取数据和生成预测
        3. 选择 Top-K 股票
        4. 调整持仓
        """
        try:
            current_date = self.get_datetime().date()

            # 检查是否需要再平衡
            if self._should_rebalance(current_date):
                self.log_message(f"开始再平衡 ({current_date})")

                # 获取预测
                predictions = self._get_predictions()

                if not predictions:
                    self.log_message("无法获取预测，跳过本次交易")
                    return

                # 选择 Top-K
                top_symbols = self._select_top_k(predictions)
                self.log_message(f"Top-{self.top_k} 股票: {top_symbols}")

                # 执行交易
                self._rebalance_portfolio(top_symbols)

                # 更新状态
                self.last_rebalance_date = current_date
                self.current_predictions = predictions

        except Exception as e:
            import traceback

            self.log_message(f"交易迭代错误: {e}")
            traceback.print_exc()

    def _should_rebalance(self, current_date) -> bool:
        """检查是否应该再平衡"""
        if self.last_rebalance_date is None:
            return True

        days_since_last = (current_date - self.last_rebalance_date).days
        return days_since_last >= self.rebalance_period

    def _get_predictions(self) -> dict:
        """
        获取所有候选股票的预测分数

        Returns:
            {symbol: predicted_return} 字典
        """
        predictions = {}

        for symbol in self.symbols:
            try:
                # 获取历史数据
                history = self.get_historical_prices(
                    symbol, self.lookback_period, "day"
                )

                if history is None or history.df.empty:
                    self.log_message(f"无法获取 {symbol} 历史数据")
                    continue

                df = history.df
                if len(df) < 30:  # 需要足够的数据生成特征
                    continue

                # 生成特征
                features = self._generate_features(df)

                if features is None or features.empty:
                    continue

                # 获取最新一行特征
                latest_features = features.iloc[[-1]]

                # 预测
                if self.trainer is not None:
                    # 使用 ML 模型预测
                    pred = self.trainer.predict(latest_features)[0]
                else:
                    # 无模型时使用动量作为代理
                    pred = df["close"].pct_change(5).iloc[-1]

                predictions[symbol] = float(pred)

            except Exception as e:
                self.log_message(f"预测 {symbol} 失败: {e}")

        return predictions

    def _generate_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """为单个股票生成特征"""
        try:
            # 确保列名小写
            df = df.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )

            # 使用特征生成器
            features = self.feature_generator._generate_single_symbol(df)

            return features

        except Exception as e:
            self.log_message(f"生成特征失败: {e}")
            return None

    def _select_top_k(self, predictions: dict) -> list:
        """
        选择预测分数最高的 Top-K 股票

        Args:
            predictions: {symbol: score} 字典

        Returns:
            Top-K 股票代码列表
        """
        if not predictions:
            return []

        # 按分数排序
        sorted_symbols = sorted(
            predictions.keys(), key=lambda x: predictions[x], reverse=True
        )

        return sorted_symbols[: self.top_k]

    def _rebalance_portfolio(self, target_symbols: list):
        """
        再平衡投资组合

        Args:
            target_symbols: 目标持仓股票列表
        """
        # 获取当前持仓
        current_positions = {}
        for symbol in self.symbols:
            pos = self.get_position(symbol)
            if pos and float(pos.quantity) > 0:
                current_positions[symbol] = float(pos.quantity)

        # 确定需要卖出的股票
        to_sell = set(current_positions.keys()) - set(target_symbols)

        # 卖出
        for symbol in to_sell:
            qty = current_positions[symbol]
            self.log_message(f"卖出 {symbol}: {qty} 股")
            order = self.create_order(symbol, qty, "sell")
            self.submit_order(order)

        # 计算目标金额
        total_value = self.portfolio_value or self.get_cash()
        if total_value <= 0:
            return

        # 预留 5% 现金缓冲
        available_cash = total_value * 0.95
        target_per_stock = available_cash / len(target_symbols) if target_symbols else 0

        # 买入/调整
        for symbol in target_symbols:
            try:
                price = self.get_last_price(symbol)
                if price is None or price <= 0:
                    continue

                # 计算目标数量
                target_qty = int(target_per_stock / price)

                # 获取当前持仓
                current_qty = current_positions.get(symbol, 0)

                # 计算差额
                diff = target_qty - current_qty

                if diff > 0:
                    # 需要买入
                    cash = self.get_cash()
                    max_buyable = int(cash / price)
                    buy_qty = min(diff, max_buyable)

                    if buy_qty > 0:
                        self.log_message(
                            f"买入 {symbol}: {buy_qty} 股 @ ${price:.2f}"
                        )
                        order = self.create_order(symbol, buy_qty, "buy")
                        self.submit_order(order)

                elif diff < 0:
                    # 需要卖出部分
                    sell_qty = abs(diff)
                    if sell_qty > 0:
                        self.log_message(
                            f"减仓 {symbol}: {sell_qty} 股 @ ${price:.2f}"
                        )
                        order = self.create_order(symbol, sell_qty, "sell")
                        self.submit_order(order)

            except Exception as e:
                self.log_message(f"调整 {symbol} 仓位失败: {e}")

    def get_prediction_summary(self) -> dict:
        """获取当前预测摘要（用于前端显示）"""
        return {
            "predictions": self.current_predictions,
            "top_k": self.top_k,
            "last_rebalance": (
                self.last_rebalance_date.isoformat()
                if self.last_rebalance_date
                else None
            ),
            "model_loaded": self.trainer is not None,
        }
