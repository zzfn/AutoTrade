"""
策略运行器 - 在主线程中执行交易策略。

此模块负责策略的生命周期管理，包括初始化、启动和停止。
与 UI 服务器完全分离，通过网络 API 通信。
"""

import logging
import os
import threading
import yaml
from datetime import datetime, timedelta

from lumibot.brokers import Alpaca
from lumibot.traders import Trader

from autotrade.strategies import QlibMLStrategy
from autotrade.ml import ModelManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ==============================================================================
# REGION: State Management
# ==============================================================================

# Strategy execution state
active_strategy = None
trader_instance = None
is_running = False

# 监控线程
monitor_thread: threading.Thread | None = None

# 市场状态检查缓存
_market_clock_cache = None
_next_api_check_time = 0

# ML 策略配置
ml_config: dict = {
    "model_name": None,  # None 表示使用最优模型
    "top_k": 3,
}

# Model manager instance
model_manager = ModelManager()

# 状态字典（用于 UI 更新）
state: dict = {
    "status": "stopped",
    "logs": [],
    "orders": [],
    "portfolio": {"cash": 0.0, "value": 0.0, "positions": []},
    "market_status": "unknown",
    "last_update": None,
    "signals": [],
    "model_loaded": False,
}

# Track known orders to prevent duplicate logs
_known_order_ids: set = set()


# ==============================================================================
# REGION: Core Functions
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


# ==============================================================================
# REGION: Strategy Execution
# ==============================================================================


def initialize_strategy() -> dict:
    """初始化 broker、策略和 trader，但不启动。"""
    global active_strategy, trader_instance

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

        return {"status": "initialized"}

    except Exception as e:
        log_message(f"设置策略失败: {e}")
        return {"status": "error", "message": str(e)}


def start_monitoring():
    """启动监控线程，定期更新策略状态。"""
    global monitor_thread

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


def run_strategy_main() -> dict:
    """在主线程中运行交易策略（阻塞调用）。

    此函数会初始化 broker、策略和 trader，然后运行策略直到完成或被中断。

    Returns:
        策略运行结果
    """
    global active_strategy, trader_instance, is_running, monitor_thread

    if is_running:
        return {"status": "already_running"}

    # 初始化策略
    init_result = initialize_strategy()
    if init_result.get("status") == "error":
        return init_result

    # 设置运行状态
    is_running = True
    update_status("running")

    # 启动监控线程
    start_monitoring()

    # 在主线程运行策略（阻塞）
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


def stop_strategy() -> dict:
    """停止运行中的策略。"""
    global is_running

    log_message("停止策略...")
    update_status("stopping")

    if trader_instance:
        try:
            if hasattr(trader_instance, "stop_all"):
                trader_instance.stop_all()
        except Exception as e:
            log_message(f"停止 trader 时出错: {e}")

    is_running = False
    update_status("stopped")
    log_message("策略已手动停止。")
    return {"status": "success", "message": "策略已停止"}


def get_state() -> dict:
    """获取当前策略状态（供 UI 服务器使用）。"""
    return state.copy()
