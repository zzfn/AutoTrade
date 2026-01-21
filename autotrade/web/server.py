"""
Web server for AutoTrade.

FastAPI application with REST API and WebSocket endpoints.
All trading strategy logic is integrated directly in this module.
"""

import asyncio
import logging
import os
import threading
import yaml
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from lumibot.backtesting import YahooDataBacktesting
from lumibot.brokers import Alpaca
from lumibot.traders import Trader

from autotrade.strategies import QlibMLStrategy
from autotrade.ml import ModelManager


load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ==============================================================================
# REGION: State Management (formerly TradeManager)
# ==============================================================================

_state_lock = threading.Lock()

# Global state - previously in TradeManager
state: dict[str, Any] = {
    "status": "stopped",
    "logs": [],
    "orders": [],
    "portfolio": {"cash": 0.0, "value": 0.0, "positions": []},
    "market_status": "unknown",
    "last_update": None,
    "signals": [],  # 实时预测信号
    "model_loaded": False,
}

# Strategy execution state
active_strategy = None
strategy_thread: threading.Thread | None = None
monitor_thread: threading.Thread | None = None
is_running = False
trader_instance = None

# Track known orders to prevent duplicate logs
_known_order_ids: set = set()

# 市场状态检查缓存
_market_clock_cache = None
_next_api_check_time = 0

# ML 策略配置
ml_config: dict[str, Any] = {
    "model_name": None,  # None 表示使用最优模型（由 ModelManager 自动选择）
    "top_k": 3,
}

# Model manager instance
model_manager = ModelManager()

# 模型训练状态
training_status = {
    "in_progress": False,
    "progress": 0,
    "message": "",
}

# 数据同步状态
data_sync_status = {
    "in_progress": False,
    "progress": 0,
    "message": "",
    "last_sync": None,
}


# ==============================================================================
# REGION: Core Trading Logic Functions (formerly TradeManager methods)
# ==============================================================================


def log_message(message: str):
    """Add a timestamped message to the logs."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    state["logs"].append(f"[{timestamp}] {message}")
    print(f"[LOG] {message}")
    if len(state["logs"]) > 100:
        state["logs"].pop(0)


def update_status(new_status: str):
    """Update the current status."""
    state["status"] = new_status
    state["last_update"] = datetime.now().isoformat()


def update_portfolio(cash: float, value: float, positions: list, market_status: str = "unknown"):
    """Update portfolio state."""
    state["portfolio"] = {"cash": cash, "value": value, "positions": positions}
    state["market_status"] = market_status
    state["last_update"] = datetime.now().isoformat()


def initialize_and_start() -> dict:
    """Initialize the broker, strategy, and trader, then start the thread."""
    global active_strategy, trader_instance, is_running

    if is_running:
        return {"status": "already_running"}

    # 1. Load credentials
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_API_SECRET")
    paper_trading = os.getenv("ALPACA_PAPER", "True").lower() == "true"

    if not api_key or not secret_key:
        log_message(
            "错误: 环境变量中未找到 Alpaca 凭证 (ALPACA_API_KEY, ALPACA_API_SECRET)。"
        )
        return {"status": "error", "message": "缺少凭证"}

    try:
        # 2. Setup Broker
        broker = Alpaca(
            {"API_KEY": api_key, "API_SECRET": secret_key, "PAPER": paper_trading}
        )

        # 3. Load symbols and interval from config
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "../../configs/qlib_ml_config.yaml")
        symbols = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]  # Default
        interval = "1min"  # Default
        lookback_period = 2  # Default
        top_k = 3  # Default
        sleeptime = "1M"  # Default

        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                    # 支持新的嵌套结构
                    if config:
                        if "data" in config:
                            data_conf = config["data"]
                            if "symbols" in data_conf:
                                symbols = data_conf["symbols"]
                            if "interval" in data_conf:
                                interval = data_conf["interval"]
                            if "lookback_period" in data_conf:
                                lookback_period = data_conf["lookback_period"]

                        # 从 strategy.ml 读取策略参数
                        if "strategy" in config and "ml" in config["strategy"]:
                            ml_conf = config["strategy"]["ml"]
                            if "top_k" in ml_conf:
                                top_k = ml_conf["top_k"]
                            if "sleeptime" in ml_conf:
                                sleeptime = ml_conf["sleeptime"]
                            if "interval" in ml_conf:
                                interval = ml_conf["interval"]

                        log_message(f"从配置文件加载: symbols={len(symbols)}, interval={interval}, lookback={lookback_period}d")
            else:
                log_message(
                    f"Config file not found at {config_path}. Using default config."
                )
        except Exception as e:
            log_message(f"Error reading config file: {e}. Using default config.")

        log_message(f"Starting strategy for symbols: {symbols}")

        # 4. Create ML Strategy
        # 如果 model_name 为 None，使用 ModelManager 的当前模型（最优模型）
        model_name = ml_config.get("model_name")
        if model_name is None:
            # 自动选择最优模型
            model_name = model_manager.get_current_model()
            if model_name:
                log_message(f"自动选择最优模型: {model_name}")
            else:
                log_message("未找到训练好的模型，将使用默认模型")

        strategy_params = {
            "symbols": symbols,
            "model_name": model_name,
            "top_k": top_k,
            "sleeptime": sleeptime,
            "interval": interval,
            "lookback_period": lookback_period,
        }

        strategy = QlibMLStrategy(broker=broker, parameters=strategy_params)
        log_message(f"使用 QlibMLStrategy，模型: {model_name or '默认'}")

        # 5. Create Trader and register
        trader_instance = Trader()
        trader_instance.add_strategy(strategy)

        active_strategy = strategy

        # 6. Start the logic
        started = start_strategy(runner=trader_instance.run_all)
        return {"status": "started" if started else "failed"}

    except Exception as e:
        log_message(f"Failed to setup strategy: {e}")
        return {"status": "error", "message": str(e)}


def run_backtest(params: dict) -> dict:
    """Run a backtest in a separate thread."""

    def _backtest_task():
        try:
            log_message("Starting backtest...")

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

            # 3. Parse interval (优先从配置文件读取)
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, "../../configs/qlib_ml_config.yaml")
            interval = params.get("interval", None)

            if interval is None and os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        config = yaml.safe_load(f)
                        if config:
                            if "data" in config and "interval" in config["data"]:
                                interval = config["data"]["interval"]
                            elif "strategy" in config and "ml" in config["strategy"] and "interval" in config["strategy"]["ml"]:
                                interval = config["strategy"]["ml"]["interval"]
                except Exception as e:
                    log_message(f"Error reading config for interval: {e}")

            if interval is None:
                interval = "1min"
            if interval not in ["1d", "1h", "1min"]:
                interval = "1min"

            log_message(
                f"Backtesting {symbols} from {backtesting_start} to {backtesting_end} (Interval: {interval})"
            )

            # 4. Execute backtest with ML strategy
            strategy_class = QlibMLStrategy

            try:
                # Map interval to LumiBot frequency
                if interval == "1min":
                    lumibot_interval = "minute"
                elif interval == "1h":
                    lumibot_interval = "hour"
                else:
                    lumibot_interval = "day"

                # 如果未指定模型，使用当前最优模型
                model_name = params.get("model_name", ml_config.get("model_name"))
                if model_name is None:
                    model_name = model_manager.get_current_model()

                backtest_params = {
                    "symbols": symbols,
                    "model_name": model_name,
                    "top_k": params.get("top_k", ml_config.get("top_k", 3)),
                    "sleeptime": "0S",
                    "timestep": "1M" if interval == "1min" else ("1H" if interval == "1h" else "1D"),
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
                    time_unit=lumibot_interval,
                )

                # Find newly generated reports in logs/
                import glob

                base_dir = os.path.dirname(os.path.abspath(__file__))
                logs_dir = os.path.join(os.path.dirname(os.path.dirname(base_dir)), "logs")

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
                    state["last_backtest"] = {
                        "tearsheet": f"/reports/{tearsheet}" if tearsheet else None,
                        "trades": f"/reports/{trades_report}"
                        if trades_report
                        else None,
                        "timestamp": datetime.now().isoformat(),
                    }
                    log_message(f"Backtest reports generated: {new_reports}")

                log_message("Backtest finished successfully.")
            except Exception as e:
                import traceback

                log_message(f"Backtest execution failed: {e}")
                print(traceback.format_exc())

        except Exception as e:
            log_message(f"Backtest error: {e}")

    # Start backtest in background thread
    thread = threading.Thread(target=_backtest_task, daemon=True)
    thread.start()
    return {"status": "backtest_started"}


def start_strategy(runner=None) -> bool:
    """Start the strategy in a separate thread and begin monitoring."""
    global is_running, strategy_thread, monitor_thread

    if is_running:
        return False

    def run_target():
        global is_running
        try:
            log_message("Starting strategy...")
            if runner:
                runner()
            elif active_strategy:
                if hasattr(active_strategy, "run_all"):
                    active_strategy.run_all()
                elif hasattr(active_strategy, "run"):
                    active_strategy.run()
                else:
                    raise AttributeError(
                        "Strategy has no run() or run_all() method and no runner provided."
                    )
            else:
                raise ValueError("No strategy set and no runner provided.")
        except Exception as e:
            log_message(f"Strategy error: {str(e)}")
        finally:
            is_running = False
            update_status("stopped")
            log_message("Strategy stopped.")

    def monitor_target():
        """Polls the strategy for state updates while it is running."""
        global _market_clock_cache, _next_api_check_time
        import time
        import pytz

        while is_running:
            try:
                if active_strategy and hasattr(active_strategy, "get_datetime"):
                    try:
                        # 1. Update Portfolio (Cash and Value)
                        try:
                            cash = float(active_strategy.get_cash())
                            value = float(active_strategy.portfolio_value)
                        except Exception:
                            cash = 0.0
                            value = 0.0

                        # 2. Update Positions
                        positions_data = []
                        try:
                            all_positions = active_strategy.get_positions()
                            for pos in all_positions:
                                if float(pos.quantity) == 0:
                                    continue

                                symbol = pos.asset.symbol
                                last_price = 0.0
                                try:
                                    if hasattr(active_strategy, "get_last_price"):
                                        last_price = float(active_strategy.get_last_price(symbol))
                                except:
                                    pass

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
                            log_message(f"Debug: Error updating positions: {e}")

                        # 3. Sync orders
                        try:
                            if hasattr(active_strategy.broker, "get_all_orders"):
                                lumi_orders = active_strategy.broker.get_all_orders()
                            else:
                                lumi_orders = active_strategy.get_orders()

                            new_orders_list = []
                            current_orders_map = {o["id"]: o for o in state["orders"]}

                            for o in lumi_orders:
                                order_id = str(o.identifier)

                                # --- Timestamp Extraction Logic ---
                                ts_str = None
                                raw_data = getattr(o, "_raw", {})
                                if not isinstance(raw_data, dict):
                                    raw_data = {}

                                val = raw_data.get("filled_at")
                                if not val:
                                    val = raw_data.get("submitted_at") or raw_data.get("created_at")

                                if not val:
                                    candidates = [
                                        "filled_at",
                                        "submitted_at",
                                        "created_at",
                                        "timestamp",
                                        "broker_create_date",
                                        "_date_created"
                                    ]
                                    for attr in candidates:
                                        v = getattr(o, attr, None)
                                        if v:
                                            val = v
                                            break

                                if val:
                                    try:
                                        if isinstance(val, str):
                                            try:
                                                val = datetime.fromisoformat(val)
                                            except:
                                                pass

                                        if hasattr(val, "astimezone"):
                                            local_val = val.astimezone()
                                            ts_str = local_val.isoformat()
                                        elif hasattr(val, "isoformat"):
                                            ts_str = val.isoformat()
                                        else:
                                            ts_str = str(val)
                                    except Exception:
                                        ts_str = str(val)

                                if not ts_str and order_id in current_orders_map:
                                    ts_str = current_orders_map[order_id]["timestamp"]

                                if not ts_str:
                                    ts_str = active_strategy.get_datetime().astimezone().isoformat()

                                # Price logic - Prioritize Average Filled Price
                                price = 0.0
                                raw_price = raw_data.get("filled_avg_price")
                                if raw_price:
                                    try:
                                        price = float(raw_price)
                                    except:
                                        pass

                                if price == 0.0:
                                    if hasattr(o, "avg_fill_price") and o.avg_fill_price:
                                        price = float(o.avg_fill_price)
                                    elif hasattr(o, "filled_avg_price") and o.filled_avg_price:
                                        price = float(o.filled_avg_price)
                                    elif hasattr(o, "average_price") and o.average_price:
                                        price = float(o.average_price)
                                    elif hasattr(o, "price") and o.price:
                                        price = float(o.price)

                                # Intent logic
                                intent = "unknown"
                                raw_intent = raw_data.get("position_intent")

                                if raw_intent is not None:
                                    if hasattr(raw_intent, "value"):
                                        intent = str(raw_intent.value)
                                    else:
                                        intent = str(raw_intent)

                                if intent == "unknown" and hasattr(o, "position_intent"):
                                    obj_intent = o.position_intent
                                    if obj_intent:
                                        if hasattr(obj_intent, "value"):
                                            intent = str(obj_intent.value)
                                        else:
                                            intent = str(obj_intent)

                                if "." in intent:
                                    intent = intent.split(".")[-1]
                                intent = intent.lower()

                                order_info = {
                                    "id": order_id,
                                    "symbol": o.asset.symbol,
                                    "action": str(o.side).upper(),
                                    "quantity": float(o.quantity),
                                    "price": price,
                                    "status": str(o.status),
                                    "timestamp": ts_str,
                                    "intent": intent,
                                }
                                new_orders_list.append(order_info)

                                # Check for new orders to log
                                if order_id not in _known_order_ids:
                                    is_recent = False
                                    try:
                                        if ts_str:
                                            odt = datetime.fromisoformat(ts_str)
                                            if (datetime.now() - odt).total_seconds() < 300:
                                                is_recent = True
                                    except:
                                        is_recent = True

                                    if is_recent:
                                        log_message(f"New Order Found [ID:{order_id}]: {order_info['action']} {order_info['quantity']} {order_info['symbol']} @ {order_info['price']}")

                                    _known_order_ids.add(order_id)

                            try:
                                new_orders_list.sort(key=lambda x: x["timestamp"], reverse=True)
                            except:
                                pass

                            state["orders"] = new_orders_list[:50]

                        except Exception:
                            pass

                        # 4. Update overall status
                        try:
                            now_ts = time.time()
                            should_update_api = False

                            if _market_clock_cache is None or now_ts >= _next_api_check_time:
                                should_update_api = True

                            if should_update_api:
                                if hasattr(active_strategy.broker, "api"):
                                    try:
                                        clock = active_strategy.broker.api.get_clock()
                                        _market_clock_cache = clock

                                        next_check = now_ts + 60
                                        current_time = clock.timestamp

                                        if clock.is_open:
                                            time_to_close = (clock.next_close - current_time).total_seconds()
                                            wait_seconds = min(time_to_close + 5, 15 * 60)
                                            next_check = max(next_check, now_ts + wait_seconds)
                                        else:
                                            time_to_open = (clock.next_open - current_time).total_seconds()

                                            ny_tz = pytz.timezone('America/New_York')
                                            now_ny = datetime.now(ny_tz)

                                            check_points = []
                                            for day_offset in [0, 1]:
                                                date_ref = now_ny.date() + timedelta(days=day_offset)
                                                check_points.append(ny_tz.localize(datetime.combine(date_ref, datetime.min.time()) + timedelta(hours=4)))
                                                check_points.append(ny_tz.localize(datetime.combine(date_ref, datetime.min.time()) + timedelta(hours=20)))

                                            next_point_wait = 15 * 60
                                            for point in check_points:
                                                wait = (point - now_ny).total_seconds()
                                                if wait > 0:
                                                    next_point_wait = wait
                                                    break

                                            wait_seconds = min(time_to_open + 5, next_point_wait + 5, 15 * 60)
                                            next_check = max(next_check, now_ts + wait_seconds)

                                        _next_api_check_time = next_check

                                    except Exception:
                                        _next_api_check_time = now_ts + 60
                                else:
                                    _market_clock_cache = None
                                    _next_api_check_time = now_ts + 60

                            market_status = "unknown"

                            if _market_clock_cache:
                                clock = _market_clock_cache
                                if clock.is_open:
                                    market_status = "open"
                                else:
                                    ny_tz = pytz.timezone('America/New_York')
                                    now_ny = datetime.now(ny_tz)

                                    if now_ny.weekday() >= 5:
                                        market_status = "closed"
                                    else:
                                        current_hour = now_ny.hour
                                        current_minute = now_ny.minute
                                        t_val = current_hour * 100 + current_minute

                                        if 400 <= t_val < 930:
                                            if clock.next_open.astimezone(ny_tz).date() == now_ny.date():
                                                market_status = "pre_market"
                                            else:
                                                market_status = "closed"
                                        elif 1600 <= t_val < 2000:
                                            market_status = "after_hours"
                                        else:
                                            market_status = "closed"

                            elif not hasattr(active_strategy.broker, "api"):
                                try:
                                    dt = active_strategy.get_datetime()
                                    if dt and dt.weekday() < 5 and 9 <= dt.hour < 16:
                                        market_status = "open"
                                    else:
                                        market_status = "closed"
                                except:
                                    market_status = "unknown"

                        except Exception:
                            pass

                        # 5. Get prediction signals from strategy
                        signals_data = []
                        try:
                            if hasattr(active_strategy, "get_prediction_summary"):
                                summary = active_strategy.get_prediction_summary()
                                predictions = summary.get("predictions", {})
                                top_k_val = summary.get("top_k", 3)
                                model_loaded = summary.get("model_loaded", False)

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
                                            "is_top_k": rank <= top_k_val,
                                        })

                                state["model_loaded"] = model_loaded
                        except Exception:
                            pass

                        state["signals"] = signals_data

                        update_portfolio(cash, value, positions_data, market_status=market_status)
                    except Exception:
                        pass
            except Exception as e:
                print(f"Outer Monitor error: {e}")
            time.sleep(1)

    is_running = True
    update_status("running")

    strategy_thread = threading.Thread(target=run_target, daemon=True)
    monitor_thread = threading.Thread(target=monitor_target, daemon=True)

    strategy_thread.start()
    monitor_thread.start()
    return True


def stop_strategy() -> dict:
    """Stop the running strategy."""
    global is_running

    log_message("Stopping strategy...")
    update_status("stopping")

    if trader_instance:
        try:
            if hasattr(trader_instance, "stop_all"):
                trader_instance.stop_all()
        except Exception as e:
            log_message(f"Error stopping trader: {e}")

    is_running = False
    update_status("stopped")
    log_message("Strategy stopped manually.")
    return {"status": "success", "message": "策略已停止"}


# ==============================================================================
# REGION: ML Strategy Configuration Functions
# ==============================================================================


def set_ml_config_internal(config: dict) -> dict:
    """设置 ML 策略配置"""
    if is_running:
        return {"status": "error", "message": "策略运行中，请先停止"}

    if "model_name" in config:
        ml_config["model_name"] = config["model_name"]
    if "top_k" in config:
        ml_config["top_k"] = int(config["top_k"])

    log_message(f"ML 配置更新: {ml_config}")
    return {"status": "success", "config": ml_config}


def get_strategy_config_internal() -> dict:
    """获取当前策略配置"""
    return {
        "strategy_type": "qlib_ml",
        "ml_config": ml_config.copy(),
        "is_running": is_running,
        "status": state["status"],
    }


def list_models_internal() -> list:
    """列出所有可用的 ML 模型"""
    return model_manager.list_models()


def get_current_model_internal() -> dict:
    """获取当前选择的模型信息"""
    model_name = model_manager.get_current_model()
    if model_name:
        info = model_manager.get_model_info(model_name)
        return {"status": "success", "model": info}
    return {"status": "success", "model": None, "message": "未选择模型"}


def select_model_internal(model_name: str) -> dict:
    """选择要使用的模型"""
    success = model_manager.set_current_model(model_name)
    if success:
        ml_config["model_name"] = model_name
        log_message(f"模型选择: {model_name}")
        return {"status": "success", "model_name": model_name}
    return {"status": "error", "message": f"模型不存在: {model_name}"}


def delete_model_internal(model_name: str) -> dict:
    """删除模型"""
    success = model_manager.delete_model(model_name)
    if success:
        log_message(f"删除模型: {model_name}")
        return {"status": "success", "model_name": model_name}
    return {"status": "error", "message": f"删除模型失败: {model_name}"}


def start_model_training_internal(config: dict = None) -> dict:
    """启动模型训练"""
    if training_status["in_progress"]:
        return {"status": "error", "message": "模型训练已在进行中"}

    def _training_task():
        try:
            from autotrade.research.data import QlibDataAdapter
            from autotrade.ml import QlibFeatureGenerator, LightGBMTrainer
            import pandas as pd
            import numpy as np

            training_status["in_progress"] = True
            training_status["progress"] = 0
            training_status["message"] = "开始模型训练..."
            log_message("开始模型训练")

            # 默认配置（优先从 YAML 配置文件读取）
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, "../../configs/qlib_ml_config.yaml")

            yaml_interval = None
            yaml_train_days = 30
            yaml_target_horizon = 60

            if os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        yaml_config = yaml.safe_load(f)
                        if yaml_config:
                            if "data" in yaml_config:
                                if "interval" in yaml_config["data"]:
                                    yaml_interval = yaml_config["data"]["interval"]
                                if "train_days" in yaml_config["data"]:
                                    yaml_train_days = yaml_config["data"]["train_days"]
                            if "model" in yaml_config and "target_horizon" in yaml_config["model"]:
                                yaml_target_horizon = yaml_config["model"]["target_horizon"]
                            if "rolling_update" in yaml_config:
                                if "interval" in yaml_config["rolling_update"]:
                                    yaml_interval = yaml_config["rolling_update"].get("interval", yaml_interval)
                                if "train_days" in yaml_config["rolling_update"]:
                                    yaml_train_days = yaml_config["rolling_update"]["train_days"]
                                if "target_horizon" in yaml_config["rolling_update"]:
                                    yaml_target_horizon = yaml_config["rolling_update"]["target_horizon"]
                except Exception as e:
                    log_message(f"Error reading YAML config: {e}")

            train_config = config or {}
            symbols = train_config.get("symbols", ["SPY", "AAPL", "MSFT"])
            train_days = train_config.get("train_days", yaml_train_days)
            target_horizon = train_config.get("target_horizon", yaml_target_horizon)
            interval = train_config.get("interval", yaml_interval or "1min")

            # 1. 加载数据 (20%)
            training_status["progress"] = 10
            training_status["message"] = f"加载数据 ({interval})..."

            adapter = QlibDataAdapter(interval=interval)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=train_days + 60)

            try:
                adapter.fetch_and_store(
                    symbols, start_date, end_date, update_mode="append"
                )
            except Exception as e:
                log_message(f"获取新数据失败（将使用现有数据）: {e}")

            df = adapter.load_data(symbols, start_date, end_date)
            training_status["progress"] = 20

            if df.empty:
                raise ValueError("没有可用的数据")

            # 2. 生成特征 (40%)
            training_status["message"] = "生成特征..."
            feature_gen = QlibFeatureGenerator(normalize=True)
            features = feature_gen.generate(df)
            training_status["progress"] = 40

            # 3. 生成目标变量
            training_status["message"] = "准备训练数据..."

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

            valid_mask = ~(features.isna().any(axis=1) | target.isna())
            features = features[valid_mask]
            target = target[valid_mask]
            training_status["progress"] = 50

            # 4. 训练模型 (80%)
            training_status["message"] = "训练模型..."

            split_idx = int(len(features) * 0.8)
            X_train, X_valid = features.iloc[:split_idx], features.iloc[split_idx:]
            y_train, y_valid = target.iloc[:split_idx], target.iloc[split_idx:]

            trainer = LightGBMTrainer(
                model_name="deepalaph",
                num_boost_round=300,
            )
            trainer.train(X_train, y_train, X_valid, y_valid)
            training_status["progress"] = 80

            # 5. 评估并保存 (100%)
            training_status["message"] = "保存模型..."
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
            training_status["progress"] = 100
            training_status["message"] = (
                f"完成！模型: {model_path.name}, IC: {metrics['ic']:.4f}"
            )

            log_message(f"模型训练完成: {model_path.name}, IC={metrics['ic']:.4f}")

        except Exception as e:
            import traceback

            training_status["message"] = f"错误: {e}"
            log_message(f"模型训练失败: {e}")
            traceback.print_exc()
        finally:
            training_status["in_progress"] = False

    thread = threading.Thread(target=_training_task, daemon=True)
    thread.start()

    return {"status": "started", "message": "模型训练已启动"}


def get_training_status_internal() -> dict:
    """获取模型训练状态"""
    return training_status


def start_data_sync_internal(config: dict = None) -> dict:
    """启动数据同步"""
    if data_sync_status["in_progress"]:
        return {"status": "error", "message": "数据同步已在进行中"}

    def _data_sync_task():
        try:
            from autotrade.research.data import QlibDataAdapter

            data_sync_status["in_progress"] = True
            data_sync_status["progress"] = 0
            data_sync_status["message"] = "准备同步数据..."
            log_message("开始数据同步")

            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, "../../configs/qlib_ml_config.yaml")

            yaml_interval = None
            yaml_days = 30

            if os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        yaml_config = yaml.safe_load(f)
                        if yaml_config:
                            if "data" in yaml_config:
                                if "interval" in yaml_config["data"]:
                                    yaml_interval = yaml_config["data"]["interval"]
                                if "train_days" in yaml_config["data"]:
                                    yaml_days = yaml_config["data"]["train_days"]
                            if "rolling_update" in yaml_config:
                                if "interval" in yaml_config["rolling_update"]:
                                    yaml_interval = yaml_config["rolling_update"].get("interval", yaml_interval)
                                if "train_days" in yaml_config["rolling_update"]:
                                    yaml_days = yaml_config["rolling_update"]["train_days"]
                except Exception as e:
                    log_message(f"Error reading YAML config: {e}")

            sync_config = config or {}
            symbols = sync_config.get("symbols", ["SPY", "AAPL", "MSFT"])
            days = sync_config.get("days", yaml_days)
            interval = sync_config.get("interval", yaml_interval or "1min")
            update_mode = sync_config.get("update_mode", "append")

            adapter = QlibDataAdapter(interval=interval)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            data_sync_status["message"] = (
                f"正在从 Alpaca 获取 {len(symbols)} 只股票的数据..."
            )
            data_sync_status["progress"] = 10

            adapter.fetch_and_store(
                symbols, start_date, end_date, update_mode=update_mode
            )

            data_sync_status["progress"] = 100
            data_sync_status["last_sync"] = datetime.now().isoformat()
            data_sync_status["message"] = (
                f"成功同步 {len(symbols)} 只股票的数据 ({interval})"
            )
            log_message(f"数据同步完成: {len(symbols)} symbols")

        except Exception as e:
            data_sync_status["message"] = f"同步失败: {e}"
            log_message(f"数据同步失败: {e}")
        finally:
            data_sync_status["in_progress"] = False

    thread = threading.Thread(target=_data_sync_task, daemon=True)
    thread.start()

    return {"status": "started", "message": "数据同步已启动"}


def get_data_sync_status_internal() -> dict:
    """获取数据同步状态"""
    return data_sync_status


# ==============================================================================
# REGION: FastAPI Application
# ==============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理 FastAPI 应用生命周期的上下文管理器。
    
    注意：策略不再在此处自动启动。
    策略由 run_strategy_main() 在主线程中运行。
    此 lifespan 仅处理清理逻辑。
    """
    logger.info("FastAPI 服务器启动...（仅作为 UI 显示）")
    
    yield
    
    logger.info("FastAPI 服务器关闭...")
    # 如果策略仍在运行，尝试清理
    if is_running:
        try:
            stop_strategy()
        except Exception as e:
            logger.error(f"清理策略时发生错误: {e}")


app = FastAPI(lifespan=lifespan)


# Paths - Updated for new location (web/server.py)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # autotrade/
UI_DIR = os.path.join(BASE_DIR, "ui")
TEMPLATES_DIR = os.path.join(UI_DIR, "templates")
STATIC_DIR = os.path.join(UI_DIR, "static")

# Mounts
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount(
    "/reports",
    StaticFiles(directory=os.path.join(os.path.dirname(BASE_DIR), "reports")),
    name="reports",
)
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/backtest", response_class=HTMLResponse)
async def read_backtest(request: Request):
    return templates.TemplateResponse(request, "backtest.html")


@app.post("/api/run_backtest")
async def api_run_backtest(request: Request):
    params = await request.json()
    return run_backtest(params)


# ==================== ML 策略相关 API ====================


@app.get("/api/strategy/config")
async def get_strategy_config():
    """获取当前策略配置"""
    return get_strategy_config_internal()


@app.post("/api/strategy/start")
async def api_start_strategy():
    """手动启动策略。
    
    注意：在 UI-Strategy 分离架构下，策略应在主线程运行。
    此 API 仅在策略未运行时提供状态信息。
    如需启动策略，请使用 `python autotrade/web/server.py` 直接运行。
    """
    if is_running:
        return {"status": "already_running", "message": "策略已在主线程运行中"}
    return {
        "status": "info",
        "message": "策略应通过 'python autotrade/web/server.py' 在主线程启动",
    }


@app.post("/api/strategy/stop")
async def api_stop_strategy():
    """手动停止策略。
    
    注意：此操作会设置停止标志，策略将在下一个周期结束后停止。
    """
    if not is_running:
        return {"status": "not_running", "message": "策略未运行"}
    return stop_strategy()


@app.post("/api/strategy/ml_config")
async def set_ml_config(request: Request):
    """设置 ML 策略配置"""
    config = await request.json()
    return set_ml_config_internal(config)


@app.get("/api/models")
async def list_models():
    """列出所有可用的 ML 模型"""
    return {"status": "success", "models": list_models_internal()}


@app.get("/api/models/current")
async def get_current_model():
    """获取当前选择的模型"""
    return get_current_model_internal()


@app.post("/api/models/select")
async def select_model(request: Request):
    """选择要使用的模型"""
    data = await request.json()
    model_name = data.get("model_name")
    if not model_name:
        return {"status": "error", "message": "缺少 model_name 参数"}
    return select_model_internal(model_name)


@app.post("/api/models/delete")
async def delete_model(request: Request):
    """删除模型"""
    data = await request.json()
    model_name = data.get("model_name")
    if not model_name:
        return {"status": "error", "message": "缺少 model_name 参数"}
    return delete_model_internal(model_name)


@app.post("/api/models/train")
async def start_model_training(request: Request):
    """启动模型训练"""
    try:
        config = await request.json()
    except Exception:
        config = None
    return start_model_training_internal(config)


@app.get("/api/models/train/status")
async def get_training_status():
    """获取模型训练状态"""
    return get_training_status_internal()


@app.post("/api/data/sync")
async def start_data_sync(request: Request):
    """启动数据同步"""
    try:
        config = await request.json()
    except Exception:
        config = None
    return start_data_sync_internal(config)


@app.get("/api/data/sync/status")
async def get_data_sync_status():
    """获取数据同步状态"""
    return get_data_sync_status_internal()


# ==================== 模型管理页面 ====================


@app.get("/models", response_class=HTMLResponse)
async def models_page(request: Request):
    """模型管理页面"""
    return templates.TemplateResponse(request, "models.html")


@app.get("/data", response_class=HTMLResponse)
async def data_page(request: Request):
    """数据中心页面"""
    return templates.TemplateResponse(request, "data.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected")
    try:
        while True:
            try:
                ws_state = {
                    "status": state.get("status", "unknown"),
                    "logs": list(state.get("logs", [])),
                    "orders": [dict(o) for o in state.get("orders", [])],
                    "portfolio": {
                        "cash": state.get("portfolio", {}).get("cash", 0.0),
                        "value": state.get("portfolio", {}).get("value", 0.0),
                        "positions": [dict(p) for p in state.get("portfolio", {}).get("positions", [])]
                    },
                    "market_status": state.get("market_status", "unknown"),
                    "last_update": state.get("last_update"),
                    "signals": [dict(s) for s in state.get("signals", [])],
                    "model_loaded": state.get("model_loaded", False),
                    "strategy_config": get_strategy_config_internal(),
                    "training_status": get_training_status_internal().copy() if isinstance(get_training_status_internal(), dict) else get_training_status_internal(),
                    "data_sync_status": get_data_sync_status_internal().copy() if isinstance(get_data_sync_status_internal(), dict) else get_data_sync_status_internal()
                }

                await websocket.send_json(ws_state)
            except (WebSocketDisconnect, RuntimeError):
                logger.info("WebSocket connection closed")
                break
            except Exception as e:
                logger.error(f"Error preparing or sending WS data: {e}")

            await asyncio.sleep(1)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WS Connection Error: {e}")


# ==============================================================================
# REGION: Background Server & Main Thread Strategy
# ==============================================================================

# 用于控制服务器关闭的事件
_server_shutdown_event = threading.Event()
_uvicorn_server = None


def start_server_background(host: str = "0.0.0.0", port: int = 8000) -> threading.Thread:
    """在后台线程中启动 FastAPI 服务器。
    
    Args:
        host: 服务器监听地址
        port: 服务器监听端口
        
    Returns:
        启动服务器的线程对象
    """
    import uvicorn
    
    global _uvicorn_server
    
    def run_server():
        global _uvicorn_server
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="info",
            # 关键：在后台线程运行时必须禁用信号处理器
            # 否则会与主线程的信号处理冲突
        )
        _uvicorn_server = uvicorn.Server(config)
        # 禁用信号处理器（只有主线程才能处理信号）
        _uvicorn_server.install_signal_handlers = lambda: None
        
        logger.info(f"启动 UI 服务器: http://{host}:{port}")
        _uvicorn_server.run()
        logger.info("UI 服务器已停止")
    
    server_thread = threading.Thread(target=run_server, daemon=True, name="UIServer")
    server_thread.start()
    
    # 等待服务器启动
    import time
    time.sleep(1)
    
    return server_thread


def stop_server_background():
    """停止后台运行的 FastAPI 服务器。"""
    global _uvicorn_server
    if _uvicorn_server:
        _uvicorn_server.should_exit = True
        logger.info("已发送服务器停止信号")


def run_strategy_main() -> dict:
    """在主线程中运行交易策略（阻塞调用）。
    
    此函数会初始化 broker、策略和 trader，然后运行策略直到完成或被中断。
    
    Returns:
        策略运行结果
    """
    global active_strategy, trader_instance, is_running, strategy_thread, monitor_thread
    
    if is_running:
        return {"status": "already_running"}
    
    # 1. 加载凭证
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_API_SECRET")
    paper_trading = os.getenv("ALPACA_PAPER", "True").lower() == "true"
    
    if not api_key or not secret_key:
        log_message(
            "错误: 环境变量中未找到 Alpaca 凭证 (ALPACA_API_KEY, ALPACA_API_SECRET)。"
        )
        return {"status": "error", "message": "缺少凭证"}
    
    try:
        # 2. 设置 Broker
        broker = Alpaca(
            {"API_KEY": api_key, "API_SECRET": secret_key, "PAPER": paper_trading}
        )
        
        # 3. 从配置加载 symbols 和 interval
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "../../configs/qlib_ml_config.yaml")
        symbols = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
        interval = "1min"
        lookback_period = 2
        top_k = 3
        sleeptime = "1M"
        
        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                    if config:
                        if "data" in config:
                            data_conf = config["data"]
                            if "symbols" in data_conf:
                                symbols = data_conf["symbols"]
                            if "interval" in data_conf:
                                interval = data_conf["interval"]
                            if "lookback_period" in data_conf:
                                lookback_period = data_conf["lookback_period"]
                        
                        if "strategy" in config and "ml" in config["strategy"]:
                            ml_conf = config["strategy"]["ml"]
                            if "top_k" in ml_conf:
                                top_k = ml_conf["top_k"]
                            if "sleeptime" in ml_conf:
                                sleeptime = ml_conf["sleeptime"]
                            if "interval" in ml_conf:
                                interval = ml_conf["interval"]
                        
                        log_message(f"从配置文件加载: symbols={len(symbols)}, interval={interval}, lookback={lookback_period}d")
            else:
                log_message(f"配置文件未找到: {config_path}，使用默认配置。")
        except Exception as e:
            log_message(f"读取配置文件时出错: {e}，使用默认配置。")
        
        log_message(f"启动策略: symbols={symbols}")
        
        # 4. 创建 ML 策略
        model_name = ml_config.get("model_name")
        if model_name is None:
            model_name = model_manager.get_current_model()
            if model_name:
                log_message(f"自动选择最优模型: {model_name}")
            else:
                log_message("未找到训练好的模型，将使用默认模型")
        
        strategy_params = {
            "symbols": symbols,
            "model_name": model_name,
            "top_k": top_k,
            "sleeptime": sleeptime,
            "interval": interval,
            "lookback_period": lookback_period,
        }
        
        strategy = QlibMLStrategy(broker=broker, parameters=strategy_params)
        log_message(f"使用 QlibMLStrategy，模型: {model_name or '默认'}")
        
        # 5. 创建 Trader 并注册
        trader_instance = Trader()
        trader_instance.add_strategy(strategy)
        
        active_strategy = strategy
        is_running = True
        update_status("running")
        
        # 6. 启动监控线程（后台）
        def monitor_target():
            """轮询策略状态更新。"""
            import time
            import pytz
            global _market_clock_cache, _next_api_check_time
            
            while is_running:
                try:
                    if active_strategy and hasattr(active_strategy, "get_datetime"):
                        try:
                            # 更新投资组合
                            try:
                                cash = float(active_strategy.get_cash())
                                value = float(active_strategy.portfolio_value)
                            except Exception:
                                cash = 0.0
                                value = 0.0
                            
                            # 更新持仓
                            positions_data = []
                            try:
                                all_positions = active_strategy.get_positions()
                                for pos in all_positions:
                                    if float(pos.quantity) == 0:
                                        continue
                                    
                                    symbol = pos.asset.symbol
                                    last_price = 0.0
                                    try:
                                        if hasattr(active_strategy, "get_last_price"):
                                            last_price = float(active_strategy.get_last_price(symbol))
                                    except:
                                        pass
                                    
                                    avg_price = float(getattr(pos, "avg_fill_price", getattr(pos, "average_price", 0.0)))
                                    upl = float(getattr(pos, "unrealized_pl", getattr(pos, "pnl", 0.0)))
                                    uplpc = float(getattr(pos, "unrealized_plpc", getattr(pos, "pnl_percent", 0.0)))
                                    
                                    positions_data.append({
                                        "symbol": symbol,
                                        "quantity": float(pos.quantity),
                                        "average_price": avg_price,
                                        "current_price": last_price,
                                        "unrealized_pl": upl,
                                        "unrealized_plpc": uplpc,
                                        "asset_class": getattr(pos.asset, "asset_class", "stock"),
                                    })
                            except Exception:
                                pass
                            
                            # 获取市场状态
                            market_status = "unknown"
                            try:
                                if hasattr(active_strategy.broker, "api"):
                                    clock = active_strategy.broker.api.get_clock()
                                    if clock.is_open:
                                        market_status = "open"
                                    else:
                                        market_status = "closed"
                            except Exception:
                                pass
                            
                            # 获取预测信号
                            signals_data = []
                            try:
                                if hasattr(active_strategy, "get_prediction_summary"):
                                    summary = active_strategy.get_prediction_summary()
                                    predictions = summary.get("predictions", {})
                                    top_k_val = summary.get("top_k", 3)
                                    model_loaded = summary.get("model_loaded", False)
                                    
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
                                                "is_top_k": rank <= top_k_val,
                                            })
                                    
                                    state["model_loaded"] = model_loaded
                            except Exception:
                                pass
                            
                            state["signals"] = signals_data
                            update_portfolio(cash, value, positions_data, market_status=market_status)
                        except Exception:
                            pass
                except Exception as e:
                    print(f"Monitor error: {e}")
                time.sleep(1)
        
        monitor_thread = threading.Thread(target=monitor_target, daemon=True, name="StrategyMonitor")
        monitor_thread.start()
        
        # 7. 在主线程运行策略（阻塞）
        log_message("策略开始在主线程运行...")
        try:
            trader_instance.run_all()
        except KeyboardInterrupt:
            log_message("收到中断信号，正在停止策略...")
        except Exception as e:
            log_message(f"策略运行出错: {e}")
        finally:
            is_running = False
            update_status("stopped")
            log_message("策略已停止。")
        
        return {"status": "completed"}
        
    except Exception as e:
        log_message(f"设置策略失败: {e}")
        is_running = False
        update_status("stopped")
        return {"status": "error", "message": str(e)}


# ==============================================================================
# REGION: Main Entry Point
# ==============================================================================


if __name__ == "__main__":
    """主入口点：UI 在后台线程，策略在主线程。
    
    使用方式：
        python autotrade/web/server.py
        
    或使用 uv：
        uv run python autotrade/web/server.py
        
    架构：
        - Thread 1 (Background): FastAPI + Uvicorn (UI 服务器)
        - Thread 0 (Main): LumiBot 策略执行
        - 共享状态: state 字典作为 UI 和策略之间的桥梁
    """
    import signal
    import sys
    
    # 设置信号处理器（仅主线程）
    def signal_handler(sig, frame):
        logger.info("收到终止信号，正在清理...")
        global is_running
        is_running = False
        stop_server_background()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 1. 启动 UI 服务器（后台线程）
    logger.info("=" * 60)
    logger.info("AutoTrade - UI/Strategy 分离模式")
    logger.info("=" * 60)
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    server_thread = start_server_background(host=host, port=port)
    
    logger.info(f"UI 服务器已在后台启动: http://{host}:{port}")
    logger.info("-" * 60)
    
    # 2. 在主线程运行策略（阻塞）
    logger.info("正在主线程启动交易策略...")
    result = run_strategy_main()
    logger.info(f"策略运行结果: {result}")
    
    # 3. 清理
    stop_server_background()
    logger.info("所有服务已停止。")
