import os
import threading
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any

from lumibot.backtesting import YahooDataBacktesting
from lumibot.brokers import Alpaca
from lumibot.traders import Trader

from autotrade.execution.strategies import QlibMLStrategy
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
            "signals": [],  # 实时预测信号
        }
        
        # 市场状态检查缓存
        self._market_clock_cache = None  # 存储 Alpaca Clock 对象
        self._next_api_check_time = 0


        # ML 策略配置
        self.ml_config: dict[str, Any] = {
            "model_name": None,  # None 表示使用最优模型（由 ModelManager 自动选择）
            "top_k": 3,
        }
        self.model_manager = ModelManager()

        # 模型训练状态
        self.training_status = {
            "in_progress": False,
            "progress": 0,
            "message": "",
        }

        # 数据同步状态
        self.data_sync_status = {
            "in_progress": False,
            "progress": 0,
            "message": "",
            "last_sync": None,
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

            # 4. Create ML Strategy
            # 如果 model_name 为 None，使用 ModelManager 的当前模型（最优模型）
            model_name = self.ml_config.get("model_name")
            if model_name is None:
                # 自动选择最优模型
                model_name = self.model_manager.get_current_model()
                if model_name:
                    self.log(f"自动选择最优模型: {model_name}")
                else:
                    self.log("未找到训练好的模型，将使用默认模型")

            strategy_params = {
                "symbols": symbols,
                "model_name": model_name,
                "top_k": self.ml_config.get("top_k", 3),
                "sleeptime": "1D",  # ML 策略通常是日频
            }

            strategy = QlibMLStrategy(broker=broker, parameters=strategy_params)
            self.log(f"使用 QlibMLStrategy，模型: {model_name or '默认'}")

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

                # 4. Execute backtest with ML strategy
                strategy_class = QlibMLStrategy

                try:
                    # Map interval to LumiBot frequency
                    # LumiBot usually uses 'day', 'hour', 'minute', etc.
                    lumibot_interval = "hour" if interval == "1h" else "day"

                    # Strategy.backtest is a blocking call
                    # 如果未指定模型，使用当前最优模型
                    model_name = params.get("model_name", self.ml_config.get("model_name"))
                    if model_name is None:
                        model_name = self.model_manager.get_current_model()

                    backtest_params = {
                        "symbols": symbols,
                        "model_name": model_name,
                        "top_k": params.get("top_k", self.ml_config.get("top_k", 3)),
                        "sleeptime": "0S",
                        "timestep": "1H" if interval == "1h" else "1D",
                    }

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
                        try:
                            # 1. Update Portfolio (Cash and Value)
                            try:
                                cash = float(self.active_strategy.get_cash())
                                value = float(self.active_strategy.portfolio_value)
                            except Exception as e:
                                # Fallback if get_cash or portfolio_value fails
                                cash = 0.0
                                value = 0.0
                                # self.log(f"Debug: Failed to get cash/value: {e}")

                            # 2. Update Positions
                            positions_data = []
                            try:
                                # use get_positions() to get all actual positions from lumibot
                                all_positions = self.active_strategy.get_positions()
                                for pos in all_positions:
                                    if float(pos.quantity) == 0:
                                        continue
                                        
                                    symbol = pos.asset.symbol
                                    # Try to get the last price for the symbol
                                    last_price = 0.0
                                    try:
                                        if hasattr(self.active_strategy, "get_last_price"):
                                            last_price = float(self.active_strategy.get_last_price(symbol))
                                    except:
                                        pass

                                    # Lumibot attributes can vary by version
                                    avg_price = float(getattr(pos, "avg_fill_price", getattr(pos, "average_price", 0.0)))
                                    upl = float(getattr(pos, "unrealized_pl", getattr(pos, "pnl", 0.0)))
                                    uplpc = float(getattr(pos, "unrealized_plpc", getattr(pos, "pnl_percent", 0.0)))

                                    positions_data.append(
                                        {
                                            "symbol": symbol,
                                            "quantity": float(pos.quantity),
                                            "average_price": avg_price,
                                            "current_price": last_price,
                                            "unrealized_pl": upl,
                                            "unrealized_plpc": uplpc,
                                            "asset_class": getattr(pos.asset, "asset_class", "stock"),
                                        }
                                    )
                            except Exception as e:
                                self.log(f"Debug: Error updating positions: {e}")

                            # 3. Sync orders
                            try:
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
                            except Exception as e:
                                # self.log(f"Debug: Error updating orders: {e}")
                                pass

                            # 4. Update overall status
                            try:
                                import time
                                import pytz
                                from datetime import datetime, timedelta

                                now_ts = time.time()
                                should_update_api = False
                                
                                # 检查是否需要更新 API 数据
                                if self._market_clock_cache is None or now_ts >= self._next_api_check_time:
                                    should_update_api = True

                                if should_update_api:
                                    if hasattr(self.active_strategy.broker, "api"):
                                        try:
                                            # 调用 API
                                            clock = self.active_strategy.broker.api.get_clock()
                                            self._market_clock_cache = clock
                                            
                                            # 计算下次更新时间 (智能缓存)
                                            # 默认至少 1 分钟后才更新，防止 API 滥用
                                            next_check = now_ts + 60
                                            
                                            current_time = clock.timestamp
                                            
                                            if clock.is_open:
                                                # 如果开盘中，下次关键时间是收盘
                                                time_to_close = (clock.next_close - current_time).total_seconds()
                                                # 设置为收盘前一点，或最长 15 分钟缓存
                                                wait_seconds = min(time_to_close + 5, 15 * 60)
                                                next_check = max(next_check, now_ts + wait_seconds)
                                            else:
                                                # 如果未开盘，下次关键时间是开盘
                                                time_to_open = (clock.next_open - current_time).total_seconds()
                                                
                                                # 但我们要考虑盘前 (04:00) 和 盘后结束 (20:00)
                                                # 这些时间点 Alpaca Clock 不直接给，需要自己算距离
                                                ny_tz = pytz.timezone('America/New_York')
                                                now_ny = datetime.now(ny_tz)
                                                
                                                # 关键时间点列表 (今天和明天的 04:00, 20:00)
                                                check_points = []
                                                for day_offset in [0, 1]:
                                                    date_ref = now_ny.date() + timedelta(days=day_offset)
                                                    check_points.append(ny_tz.localize(datetime.combine(date_ref, datetime.min.time()) + timedelta(hours=4)))  # 04:00
                                                    check_points.append(ny_tz.localize(datetime.combine(date_ref, datetime.min.time()) + timedelta(hours=20))) # 20:00
                                                
                                                # 找到最近的一个还未到的时间点
                                                next_point_wait = 15 * 60 # 默认 15 min
                                                for point in check_points:
                                                    wait = (point - now_ny).total_seconds()
                                                    if wait > 0:
                                                        next_point_wait = wait
                                                        break
                                                
                                                # 取 min(距离开盘, 距离盘前/盘后切换点, 15分钟)
                                                wait_seconds = min(time_to_open + 5, next_point_wait + 5, 15 * 60)
                                                next_check = max(next_check, now_ts + wait_seconds)

                                            self._next_api_check_time = next_check
                                            # self.log(f"Debug: Market Clock updated. Next check in {(self._next_api_check_time - now_ts):.0f}s")
                                            
                                        except Exception as e:
                                            # self.log(f"Debug: API call failed: {e}")
                                            # 出错后 1 分钟重试
                                            self._next_api_check_time = now_ts + 60
                                    else:
                                        # Fallback logic if no API
                                        self._market_clock_cache = None
                                        self._next_api_check_time = now_ts + 60

                                # ----------------------------------------------------
                                # 基于缓存的 Clock 数据计算当前状态
                                # ----------------------------------------------------
                                market_status = "unknown"
                                
                                if self._market_clock_cache:
                                    clock = self._market_clock_cache
                                    if clock.is_open:
                                        market_status = "open"
                                    else:
                                        # 计算详细状态 (pre_market, after_hours, closed)
                                        # 使用本地当前时间 (因为 API timestamp 是快照)
                                        ny_tz = pytz.timezone('America/New_York')
                                        now_ny = datetime.now(ny_tz)
                                        
                                        # 周末判断
                                        if now_ny.weekday() >= 5:
                                            market_status = "closed"
                                        else:
                                            current_hour = now_ny.hour
                                            current_minute = now_ny.minute
                                            t_val = current_hour * 100 + current_minute
                                            
                                            # 04:00 - 09:30 Pre-Market
                                            # 09:30 - 16:00 Market (Should be handled by is_open, but simple fallback)
                                            # 16:00 - 20:00 After-Hours
                                            
                                            if 400 <= t_val < 930:
                                                # 只有当今天(或next_open那一天)是交易日才算 Pre-Market
                                                # Check if next_open is today
                                                if clock.next_open.astimezone(ny_tz).date() == now_ny.date():
                                                    market_status = "pre_market"
                                                else:
                                                    market_status = "closed" # Holiday morning
                                            elif 1600 <= t_val < 2000:
                                                market_status = "after_hours"
                                            else:
                                                market_status = "closed"
                                                
                                elif not hasattr(self.active_strategy.broker, "api"):
                                    # Fallback simple logic
                                    try:
                                        dt = self.active_strategy.get_datetime()
                                        if dt and dt.weekday() < 5 and 9 <= dt.hour < 16:
                                            market_status = "open"
                                        else:
                                            market_status = "closed"
                                    except:
                                        market_status = "unknown"
                                else:
                                    # API available but cache fetch failed specific status usually kept from previous
                                    pass

                            except Exception as e:
                                # self.log(f"Debug: Error checking market status: {e}")
                                # import traceback
                                # traceback.print_exc()
                                pass

                            # 5. Get prediction signals from strategy
                            signals_data = []
                            try:
                                if hasattr(self.active_strategy, "get_prediction_summary"):
                                    summary = self.active_strategy.get_prediction_summary()
                                    predictions = summary.get("predictions", {})
                                    top_k = summary.get("top_k", 3)
                                    model_loaded = summary.get("model_loaded", False)
                                    
                                    # Convert predictions to sorted list
                                    if predictions:
                                        sorted_preds = sorted(
                                            predictions.items(),
                                            key=lambda x: x[1],
                                            reverse=True
                                        )
                                        for rank, (symbol, score) in enumerate(sorted_preds, 1):
                                            signals_data.append({
                                                "symbol": symbol,
                                                "score": float(score),
                                                "rank": rank,
                                                "is_top_k": rank <= top_k,
                                            })
                                    
                                    self.state["model_loaded"] = model_loaded
                            except Exception as e:
                                # self.log(f"Debug: Error getting signals: {e}")
                                pass
                            
                            self.state["signals"] = signals_data

                            self.update_portfolio(
                                cash, value, positions_data, market_status=market_status
                            )
                        except Exception as e:
                            # self.log(f"Inner monitor error: {e}")
                            pass
                except Exception as e:
                    print(f"Outer Monitor error: {e}")
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
                    # This might block, but we call it from a thread or with timeout
                    self.trader.stop_all()
            except Exception as e:
                self.log(f"Error stopping trader: {e}")

        # Force thread to end if possible and mark as stopped
        self.is_running = False
        self.update_status("stopped")
        self.log("Strategy stopped manually.")
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

    def set_ml_config(self, config: dict) -> dict:
        """
        设置 ML 策略配置

        Args:
            config: 包含 model_name, top_k 等参数

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

        self.log(f"ML 配置更新: {self.ml_config}")
        return {"status": "success", "config": self.ml_config}

    def get_strategy_config(self) -> dict:
        """获取当前策略配置"""
        return {
            "strategy_type": "qlib_ml",  # 固定使用 ML 策略
            "ml_config": self.ml_config.copy(),
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

    def delete_model(self, model_name: str) -> dict:
        """
        删除模型

        Args:
            model_name: 模型名称

        Returns:
            操作结果
        """
        success = self.model_manager.delete_model(model_name)
        if success:
            self.log(f"删除模型: {model_name}")
            return {"status": "success", "model_name": model_name}
        return {"status": "error", "message": f"删除模型失败: {model_name}"}

    def start_model_training(self, config: dict = None) -> dict:
        """
        启动模型训练

        Args:
            config: 可选的训练配置

        Returns:
            操作结果
        """
        if self.training_status["in_progress"]:
            return {"status": "error", "message": "模型训练已在进行中"}

        def _training_task():
            try:
                from autotrade.research.data import QlibDataAdapter
                from autotrade.research.features import QlibFeatureGenerator
                from autotrade.research.models import LightGBMTrainer
                from datetime import timedelta

                self.training_status["in_progress"] = True
                self.training_status["progress"] = 0
                self.training_status["message"] = "开始模型训练..."
                self.log("开始模型训练")

                # 默认配置
                train_config = config or {}
                symbols = train_config.get("symbols", ["SPY", "AAPL", "MSFT"])
                train_days = train_config.get("train_days", 252)
                target_horizon = train_config.get("target_horizon", 5)
                interval = train_config.get("interval", "1d")

                # 1. 加载数据 (20%)
                self.training_status["progress"] = 10
                self.training_status["message"] = f"加载数据 ({interval})..."

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
                self.training_status["progress"] = 20

                if df.empty:
                    raise ValueError("没有可用的数据")

                # 2. 生成特征 (40%)
                self.training_status["message"] = "生成特征..."
                feature_gen = QlibFeatureGenerator(normalize=True)
                features = feature_gen.generate(df)
                self.training_status["progress"] = 40

                # 3. 生成目标变量
                self.training_status["message"] = "准备训练数据..."
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
                self.training_status["progress"] = 50

                # 4. 训练模型 (80%)
                self.training_status["message"] = "训练模型..."

                # 分割训练/验证集 (80/20)
                split_idx = int(len(features) * 0.8)
                X_train, X_valid = features.iloc[:split_idx], features.iloc[split_idx:]
                y_train, y_valid = target.iloc[:split_idx], target.iloc[split_idx:]

                trainer = LightGBMTrainer(
                    model_name="lightgbm_rolling",
                    num_boost_round=300,
                )
                trainer.train(X_train, y_train, X_valid, y_valid)
                self.training_status["progress"] = 80

                # 5. 评估并保存 (100%)
                self.training_status["message"] = "保存模型..."
                metrics = trainer.evaluate(X_valid, y_valid)
                trainer.metadata.update(
                    {
                        "symbols": symbols,
                        "train_days": train_days,
                        "interval": interval,
                        "ic": metrics["ic"],
                        "icir": metrics["icir"],
                        "trained_via_ui": True,
                        "updated_at": datetime.now().isoformat(),
                    }
                )

                model_path = trainer.save()
                self.training_status["progress"] = 100
                self.training_status["message"] = (
                    f"完成！模型: {model_path.name}, IC: {metrics['ic']:.4f}"
                )

                self.log(f"模型训练完成: {model_path.name}, IC={metrics['ic']:.4f}")

            except Exception as e:
                import traceback

                self.training_status["message"] = f"错误: {e}"
                self.log(f"模型训练失败: {e}")
                traceback.print_exc()
            finally:
                self.training_status["in_progress"] = False

        # 启动后台任务
        thread = threading.Thread(target=_training_task, daemon=True)
        thread.start()

        return {"status": "started", "message": "模型训练已启动"}

    def get_training_status(self) -> dict:
        """获取模型训练状态"""
        return self.training_status

    def start_data_sync(self, config: dict = None) -> dict:
        """
        启动数据同步

        Args:
            config: 包含 symbols, days, interval, update_mode 等参数
        """
        if self.data_sync_status["in_progress"]:
            return {"status": "error", "message": "数据同步已在进行中"}

        def _data_sync_task():
            try:
                from autotrade.research.data import QlibDataAdapter
                from datetime import timedelta

                self.data_sync_status["in_progress"] = True
                self.data_sync_status["progress"] = 0
                self.data_sync_status["message"] = "准备同步数据..."
                self.log("开始数据同步")

                sync_config = config or {}
                symbols = sync_config.get("symbols", ["SPY", "AAPL", "MSFT"])
                days = sync_config.get("days", 365)
                interval = sync_config.get("interval", "1d")
                update_mode = sync_config.get("update_mode", "append")

                adapter = QlibDataAdapter(interval=interval)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)

                self.data_sync_status["message"] = (
                    f"正在从 Alpaca 获取 {len(symbols)} 只股票的数据..."
                )
                self.data_sync_status["progress"] = 10

                adapter.fetch_and_store(
                    symbols, start_date, end_date, update_mode=update_mode
                )

                self.data_sync_status["progress"] = 100
                self.data_sync_status["last_sync"] = datetime.now().isoformat()
                self.data_sync_status["message"] = (
                    f"成功同步 {len(symbols)} 只股票的数据 ({interval})"
                )
                self.log(f"数据同步完成: {len(symbols)} symbols")

            except Exception as e:
                self.data_sync_status["message"] = f"同步失败: {e}"
                self.log(f"数据同步失败: {e}")
            finally:
                self.data_sync_status["in_progress"] = False

        thread = threading.Thread(target=_data_sync_task, daemon=True)
        thread.start()

        return {"status": "started", "message": "数据同步已启动"}

    def get_data_sync_status(self) -> dict:
        """获取数据同步状态"""
        return self.data_sync_status
