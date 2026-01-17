import os
import threading
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any

from lumibot.backtesting import YahooDataBacktesting
from lumibot.brokers import Alpaca
from lumibot.traders import Trader

from autotrade.execution.strategies import (
    MomentumStrategy,
    QlibMLStrategy,
    get_strategy_class,
)
from autotrade.research.models import ModelManager


class TradeManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self.active_strategy = None
        self.strategy_thread: threading.Thread | None = None
        self.is_running = False

        # State storage
        self.state: dict[str, Any] = {
            "status": "stopped",
            "logs": [],
            "orders": [],
            "portfolio": {"cash": 0.0, "value": 0.0, "positions": []},
            "market_status": "unknown",
            "last_update": None,
        }

        # ML 策略配置
        self.strategy_type = "momentum"  # 'momentum' 或 'qlib_ml'
        self.ml_config: dict[str, Any] = {
            "model_name": None,  # None 表示使用当前模型
            "top_k": 3,
            "rebalance_period": 1,
        }
        self.model_manager = ModelManager()

        # Rolling 更新状态
        self.rolling_update_status = {
            "in_progress": False,
            "progress": 0,
            "message": "",
        }

        self._initialized = True

    def set_strategy(self, strategy_instance):
        """Set the strategy instance to be managed."""
        self.active_strategy = strategy_instance

    def initialize_and_start(self):
        """Initialize the broker, strategy, and trader, then start the thread."""
        if self.is_running:
            return {"status": "already_running"}

        # 1. Load credentials
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_API_SECRET")
        paper_trading = os.getenv("ALPACA_PAPER", "True").lower() == "true"

        if not api_key or not secret_key:
            self.log(
                "错误: 环境变量中未找到 Alpaca 凭证 (ALPACA_API_KEY, ALPACA_API_SECRET)。"
            )
            return {"status": "error", "message": "缺少凭证"}

        try:
            # 2. Setup Broker
            broker = Alpaca(
                {"API_KEY": api_key, "API_SECRET": secret_key, "PAPER": paper_trading}
            )

            # 3. Load symbols from config
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, "../configs/universe.yaml")
            symbols = ["SPY"]  # Default

            try:
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        config = yaml.safe_load(f)
                        if config and "symbols" in config:
                            symbols = config["symbols"]
                            self.log(f"Loaded symbols from config: {symbols}")
                        else:
                            self.log(
                                "Config file found but no 'symbols' key. Using default."
                            )
                else:
                    self.log(
                        f"Config file not found at {config_path}. Using default symbols."
                    )
            except Exception as e:
                self.log(f"Error reading config file: {e}. Using default symbols.")

            self.log(f"Starting strategy for symbols: {symbols}")
            self.log(f"Strategy type: {self.strategy_type}")

            # 4. Create Strategy based on type
            strategy_params = {
                "symbols": symbols,
                "sleeptime": "10S",
                "lookback_period": 60,
            }

            if self.strategy_type == "qlib_ml":
                # ML 策略特定参数
                strategy_params.update(
                    {
                        "model_name": self.ml_config.get("model_name"),
                        "top_k": self.ml_config.get("top_k", 3),
                        "rebalance_period": self.ml_config.get("rebalance_period", 1),
                        "sleeptime": "1D",  # ML 策略通常是日频
                    }
                )
                strategy = QlibMLStrategy(broker=broker, parameters=strategy_params)
                self.log(
                    f"Using QlibMLStrategy with model: {self.ml_config.get('model_name', 'current')}"
                )
            else:
                # 默认动量策略
                strategy = MomentumStrategy(broker=broker, parameters=strategy_params)
                self.log("Using MomentumStrategy")

            # 5. Create Trader and register
            self.trader = Trader()
            self.trader.add_strategy(strategy)

            self.set_strategy(strategy)

            # 6. Start the logic
            started = self.start_strategy(runner=self.trader.run_all)
            return {"status": "started" if started else "failed"}

        except Exception as e:
            self.log(f"Failed to setup strategy: {e}")
            return {"status": "error", "message": str(e)}

    def run_backtest(self, params: dict):
        """Run a backtest in a separate thread."""

        def _backtest_task():
            try:
                self.log("Starting backtest...")

                # 1. Parse dates
                backtesting_start = datetime.strptime(
                    params.get("start_date", "2023-01-01"), "%Y-%m-%d"
                )
                backtesting_end = datetime.strptime(
                    params.get("end_date", "2023-01-31"), "%Y-%m-%d"
                )

                # 2. Parse symbols (clean up quotes and spaces)
                symbol_input = params.get("symbol", "SPY")
                symbols = [
                    s.strip().replace('"', "").replace("'", "")
                    for s in symbol_input.split(",")
                    if s.strip()
                ]
                if not symbols:
                    symbols = ["SPY"]

                # 3. Parse interval
                interval = params.get("interval", "1d")
                if interval not in ["1d", "1h"]:
                    interval = "1d"

                self.log(
                    f"Backtesting {symbols} from {backtesting_start} to {backtesting_end} (Interval: {interval})"
                )

                # 4. Get strategy class and execute backtest
                strategy_type = params.get("strategy_type", self.strategy_type)
                strategy_class = get_strategy_class(strategy_type)

                try:
                    # Map interval to LumiBot frequency
                    # LumiBot usually uses 'day', 'hour', 'minute', etc.
                    lumibot_interval = "hour" if interval == "1h" else "day"

                    # Strategy.backtest is a blocking call
                    backtest_params = {
                        "symbols": symbols,
                        "lookback_period": 60,
                        "sleeptime": "0S",
                        "timestep": "1H" if interval == "1h" else "1D",
                    }

                    # Add ML-specific params if needed
                    if strategy_type == "qlib_ml":
                        backtest_params.update(
                            {
                                "model_name": params.get(
                                    "model_name", self.ml_config.get("model_name")
                                ),
                                "top_k": params.get(
                                    "top_k", self.ml_config.get("top_k", 3)
                                ),
                                "rebalance_period": params.get("rebalance_period", 1),
                            }
                        )

                    # If multiple symbols, use SPY as benchmark for better clarity
                    benchmark = (
                        "SPY" if len(symbols) > 1 or symbols[0] != "SPY" else symbols[0]
                    )

                    # Start time to identify new files
                    start_time = datetime.now()

                    strategy_class.backtest(
                        YahooDataBacktesting,
                        backtesting_start,
                        backtesting_end,
                        benchmark_asset=benchmark,
                        parameters=backtest_params,
                        # LumiBot uses time_unit for interval
                        time_unit=lumibot_interval,
                    )

                    # Find newly generated reports in logs/
                    import glob

                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    logs_dir = os.path.join(os.path.dirname(base_dir), "logs")

                    # Look for html files generated recently
                    html_files = glob.glob(os.path.join(logs_dir, "*.html"))
                    new_reports = []
                    for f in html_files:
                        if (
                            os.path.getmtime(f) >= start_time.timestamp() - 5
                        ):  # 5s buffer
                            new_reports.append(os.path.basename(f))

                    tearsheet = next((f for f in new_reports if "tearsheet" in f), None)
                    trades_report = next(
                        (f for f in new_reports if "trades" in f), None
                    )

                    if tearsheet or trades_report:
                        self.state["last_backtest"] = {
                            "tearsheet": f"/reports/{tearsheet}" if tearsheet else None,
                            "trades": f"/reports/{trades_report}"
                            if trades_report
                            else None,
                            "timestamp": datetime.now().isoformat(),
                        }
                        self.log(f"Backtest reports generated: {new_reports}")

                    self.log("Backtest finished successfully.")
                except Exception as e:
                    import traceback

                    self.log(f"Backtest execution failed: {e}")
                    print(traceback.format_exc())

            except Exception as e:
                self.log(f"Backtest error: {e}")

        # Start backtest in background thread
        thread = threading.Thread(target=_backtest_task, daemon=True)
        thread.start()
        return {"status": "backtest_started"}

    def start_strategy(self, runner=None):
        """Start the strategy in a separate thread and begin monitoring."""
        if self.is_running:
            return False

        def run_target():
            try:
                self.log("Starting strategy...")
                if runner:
                    runner()
                elif self.active_strategy:
                    if hasattr(self.active_strategy, "run_all"):
                        self.active_strategy.run_all()
                    elif hasattr(self.active_strategy, "run"):
                        self.active_strategy.run()
                    else:
                        raise AttributeError(
                            "Strategy has no run() or run_all() method and no runner provided."
                        )
                else:
                    raise ValueError("No strategy set and no runner provided.")
            except Exception as e:
                self.log(f"Strategy error: {str(e)}")
            finally:
                self.is_running = False
                self.update_status("stopped")
                self.log("Strategy stopped.")

        def monitor_target():
            """Polls the strategy for state updates while it is running."""
            import time

            while self.is_running:
                try:
                    if self.active_strategy and hasattr(
                        self.active_strategy, "get_datetime"
                    ):
                        # Only update if the strategy is actually initialized and running
                        # Note: We use a try-except because LumiBot methods might fail if not ready
                        try:
                            cash = float(self.active_strategy.get_cash())
                            value = float(self.active_strategy.portfolio_value)

                            # Get symbols from the strategy
                            symbols = getattr(self.active_strategy, "symbols", [])
                            positions_data = []
                            for symbol in symbols:
                                pos = self.active_strategy.get_position(symbol)
                                if pos:
                                    positions_data.append(
                                        {
                                            "symbol": symbol,
                                            "quantity": float(pos.quantity),
                                            "average_price": float(pos.average_price),
                                            "current_price": float(
                                                self.active_strategy.get_last_price(
                                                    symbol
                                                )
                                            )
                                            if hasattr(
                                                self.active_strategy, "get_last_price"
                                            )
                                            else 0.0,
                                            "unrealized_pl": float(pos.unrealized_pl)
                                            if hasattr(pos, "unrealized_pl")
                                            else 0.0,
                                            "unrealized_plpc": float(
                                                pos.unrealized_plpc
                                            )
                                            if hasattr(pos, "unrealized_plpc")
                                            else 0.0,
                                        }
                                    )

                            # Sync orders
                            lumi_orders = self.active_strategy.get_orders()
                            current_order_ids = [o["id"] for o in self.state["orders"]]

                            for o in lumi_orders:
                                order_id = str(o.identifier)
                                if order_id not in current_order_ids:
                                    order_info = {
                                        "id": order_id,
                                        "symbol": o.asset.symbol,
                                        "action": str(o.side).upper(),
                                        "quantity": float(o.quantity),
                                        "price": float(o.price) if o.price else 0.0,
                                        "status": str(o.status),
                                        "timestamp": self.active_strategy.get_datetime().isoformat(),
                                    }
                                    self.add_order(order_info)
                                    self.log(
                                        f"New Order: {order_info['action']} {order_info['quantity']} {order_info['symbol']} @ {order_info['price']}"
                                    )
                                else:
                                    # Update status of existing orders if they changed
                                    for existing in self.state["orders"]:
                                        if existing["id"] == order_id and existing[
                                            "status"
                                        ] != str(o.status):
                                            existing["status"] = str(o.status)
                                            self.log(
                                                f"Order Update: {order_id} is now {str(o.status)}"
                                            )

                            market_status = (
                                "open"
                                if self.active_strategy.get_datetime().weekday() < 5
                                else "closed"
                            )
                            self.update_portfolio(
                                cash, value, positions_data, market_status=market_status
                            )
                        except:
                            pass  # Might not be fully initialized yet
                except Exception as e:
                    print(f"Monitor error: {e}")
                time.sleep(1)  # Poll every 1 second

        self.is_running = True
        self.update_status("running")

        # Start both strategy and monitor
        self.strategy_thread = threading.Thread(target=run_target, daemon=True)
        self.monitor_thread = threading.Thread(target=monitor_target, daemon=True)

        self.strategy_thread.start()
        self.monitor_thread.start()
        return True

    def stop_strategy(self):
        """Stop the running strategy."""
        self.log("Stopping strategy...")
        self.update_status("stopping")

        if hasattr(self, "trader") and self.trader:
            try:
                # In newer LumiBot versions, we can stop the trader
                # If not available, we at least mark it as not running
                if hasattr(self.trader, "stop_all"):
                    self.trader.stop_all()
            except Exception as e:
                self.log(f"Error stopping trader: {e}")

        # Force thread to end if possible and mark as stopped
        self.is_running = False
        self.update_status("stopped")
        return {"status": "success", "message": "策略已停止"}

    def update_status(self, status: str):
        self.state["status"] = status
        self.state["last_update"] = datetime.now().isoformat()

    def log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.state["logs"].append(f"[{timestamp}] {message}")
        print(f"[TM LOG] {message}")
        if len(self.state["logs"]) > 100:
            self.state["logs"].pop(0)

    def update_portfolio(self, cash, value, positions, market_status="unknown"):
        # print(f"DEBUG: Updating portfolio: Cash={cash}, Val={value}, Pos={len(positions)}, Market={market_status}")
        self.state["portfolio"] = {"cash": cash, "value": value, "positions": positions}
        self.state["market_status"] = market_status
        self.state["last_update"] = datetime.now().isoformat()

    def add_order(self, order_info):
        self.state["orders"].insert(0, order_info)
        if len(self.state["orders"]) > 50:
            self.state["orders"].pop()

    def get_state(self):
        return self.state

    # ==================== ML 策略相关 API ====================

    def set_strategy_type(self, strategy_type: str) -> dict:
        """
        设置策略类型

        Args:
            strategy_type: 'momentum' 或 'qlib_ml'


        Returns:
            操作结果
        """
        if strategy_type not in ["momentum", "qlib_ml"]:
            return {"status": "error", "message": f"未知策略类型: {strategy_type}"}

        if self.is_running:
            return {"status": "error", "message": "策略运行中，请先停止"}

        self.strategy_type = strategy_type
        self.log(f"策略类型设置为: {strategy_type}")
        return {"status": "success", "strategy_type": strategy_type}

    def set_ml_config(self, config: dict) -> dict:
        """
        设置 ML 策略配置

        Args:
            config: 包含 model_name, top_k, rebalance_period 等参数

        Returns:
            操作结果
        """
        if self.is_running:
            return {"status": "error", "message": "策略运行中，请先停止"}

        # 更新配置
        if "model_name" in config:
            self.ml_config["model_name"] = config["model_name"]
        if "top_k" in config:
            self.ml_config["top_k"] = int(config["top_k"])
        if "rebalance_period" in config:
            self.ml_config["rebalance_period"] = int(config["rebalance_period"])

        self.log(f"ML 配置更新: {self.ml_config}")
        return {"status": "success", "config": self.ml_config}

    def get_strategy_config(self) -> dict:
        """获取当前策略配置"""
        return {
            "strategy_type": self.strategy_type,
            "ml_config": self.ml_config,
            "available_strategies": ["momentum", "qlib_ml"],
            "is_running": self.is_running,
            "status": self.state["status"],
        }

    def list_models(self) -> list:
        """列出所有可用的 ML 模型"""
        return self.model_manager.list_models()

    def get_current_model(self) -> dict:
        """获取当前选择的模型信息"""
        model_name = self.model_manager.get_current_model()
        if model_name:
            info = self.model_manager.get_model_info(model_name)
            return {"status": "success", "model": info}
        return {"status": "success", "model": None, "message": "未选择模型"}

    def select_model(self, model_name: str) -> dict:
        """
        选择要使用的模型

        Args:
            model_name: 模型名称

        Returns:
            操作结果
        """
        success = self.model_manager.set_current_model(model_name)
        if success:
            # 同时更新 ML 配置
            self.ml_config["model_name"] = model_name
            self.log(f"模型选择: {model_name}")
            return {"status": "success", "model_name": model_name}
        return {"status": "error", "message": f"模型不存在: {model_name}"}

    def start_rolling_update(self, config: dict = None) -> dict:
        """
        启动 Rolling 模型更新

        Args:
            config: 可选的训练配置

        Returns:
            操作结果
        """
        if self.rolling_update_status["in_progress"]:
            return {"status": "error", "message": "Rolling 更新已在进行中"}

        def _rolling_update_task():
            try:
                from autotrade.research.data import QlibDataAdapter
                from autotrade.research.features import QlibFeatureGenerator
                from autotrade.research.models import LightGBMTrainer
                from datetime import timedelta

                self.rolling_update_status["in_progress"] = True
                self.rolling_update_status["progress"] = 0
                self.rolling_update_status["message"] = "开始 Rolling 更新..."
                self.log("开始 Rolling 更新")

                # 默认配置
                train_config = config or {}
                symbols = train_config.get("symbols", ["SPY", "AAPL", "MSFT"])
                train_days = train_config.get("train_days", 252)
                target_horizon = train_config.get("target_horizon", 5)
                interval = train_config.get("interval", "1d")

                # 1. 加载数据 (20%)
                self.rolling_update_status["progress"] = 10
                self.rolling_update_status["message"] = f"加载数据 ({interval})..."

                adapter = QlibDataAdapter(interval=interval)
                end_date = datetime.now()
                # 增加一点缓冲，确保覆盖足够的数据，特别是对于小时数据
                start_date = end_date - timedelta(days=train_days + 60)

                # 尝试获取新数据
                try:
                    adapter.fetch_and_store(
                        symbols, start_date, end_date, update_mode="append"
                    )
                except Exception as e:
                    self.log(f"获取新数据失败（将使用现有数据）: {e}")

                df = adapter.load_data(symbols, start_date, end_date)
                self.rolling_update_status["progress"] = 20

                if df.empty:
                    raise ValueError("没有可用的数据")

                # 2. 生成特征 (40%)
                self.rolling_update_status["message"] = "生成特征..."
                feature_gen = QlibFeatureGenerator(normalize=True)
                features = feature_gen.generate(df)
                self.rolling_update_status["progress"] = 40

                # 3. 生成目标变量
                self.rolling_update_status["message"] = "准备训练数据..."
                import pandas as pd

                if isinstance(df.index, pd.MultiIndex):
                    close_prices = df["close"].unstack("symbol")
                    future_returns = close_prices.pct_change(target_horizon).shift(
                        -target_horizon
                    )
                    target = future_returns.stack().reindex(features.index)
                else:
                    target = (
                        df["close"].pct_change(target_horizon).shift(-target_horizon)
                    )
                    target = target.reindex(features.index)

                # 移除 NaN
                import numpy as np

                valid_mask = ~(features.isna().any(axis=1) | target.isna())
                features = features[valid_mask]
                target = target[valid_mask]
                self.rolling_update_status["progress"] = 50

                # 4. 训练模型 (80%)
                self.rolling_update_status["message"] = "训练模型..."

                # 分割训练/验证集 (80/20)
                split_idx = int(len(features) * 0.8)
                X_train, X_valid = features.iloc[:split_idx], features.iloc[split_idx:]
                y_train, y_valid = target.iloc[:split_idx], target.iloc[split_idx:]

                trainer = LightGBMTrainer(
                    model_name="lightgbm_rolling",
                    num_boost_round=300,
                )
                trainer.train(X_train, y_train, X_valid, y_valid)
                self.rolling_update_status["progress"] = 80

                # 5. 评估并保存 (100%)
                self.rolling_update_status["message"] = "保存模型..."
                metrics = trainer.evaluate(X_valid, y_valid)
                trainer.metadata.update(
                    {
                        "symbols": symbols,
                        "train_days": train_days,
                        "interval": interval,
                        "ic": metrics["ic"],
                        "icir": metrics["icir"],
                        "rolling_update": True,
                        "updated_at": datetime.now().isoformat(),
                    }
                )

                model_path = trainer.save()
                self.rolling_update_status["progress"] = 100
                self.rolling_update_status["message"] = (
                    f"完成！模型: {model_path.name}, IC: {metrics['ic']:.4f}"
                )

                self.log(f"Rolling 更新完成: {model_path.name}, IC={metrics['ic']:.4f}")

            except Exception as e:
                import traceback

                self.rolling_update_status["message"] = f"错误: {e}"
                self.log(f"Rolling 更新失败: {e}")
                traceback.print_exc()
            finally:
                self.rolling_update_status["in_progress"] = False

        # 启动后台任务
        thread = threading.Thread(target=_rolling_update_task, daemon=True)
        thread.start()

        return {"status": "started", "message": "Rolling 更新已启动"}

    def get_rolling_update_status(self) -> dict:
        """获取 Rolling 更新状态"""
        return self.rolling_update_status
