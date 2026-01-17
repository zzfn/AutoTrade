import asyncio
import logging
import os
import signal
import threading
import traceback
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from lumibot.backtesting import YahooDataBacktesting

# LumiBot Imports
from lumibot.brokers import Alpaca
from lumibot.traders import Trader

from autotrade.strategies.momentum_strategy import MomentumStrategy
from autotrade.trade_manager import TradeManager


load_dotenv()

# Monkey patch signal.signal to prevent ValueError in non-main threads
_original_signal = signal.signal


def _thread_safe_signal(signum, handler):
    if threading.current_thread() is not threading.main_thread():
        logging.warning(
            f"Ignored signal registration for {signum} from non-main thread."
        )
        return
    return _original_signal(signum, handler)


signal.signal = _thread_safe_signal


app = FastAPI()

import yaml

# ... existing imports ...

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
CONFIG_PATH = os.path.join(BASE_DIR, "../configs/universe.yaml")

# Mounts
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Manager
tm = TradeManager()


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/backtest", response_class=HTMLResponse)
async def read_backtest(request: Request):
    return templates.TemplateResponse("backtest.html", {"request": request})


@app.post("/api/start")
async def start_trading():
    if tm.is_running:
        return {"status": "already_running"}

    # Setup Broker and Strategy
    # Using Environment Variables
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_API_SECRET")
    paper_trading = os.getenv("ALPACA_PAPER", "True").lower() == "true"

    if not api_key or not secret_key:
        tm.log(
            "错误: 环境变量中未找到 Alpaca 凭证 (ALPACA_API_KEY, ALPACA_API_SECRET)。"
        )
        return {"status": "error", "message": "缺少凭证"}

    try:
        broker = Alpaca(
            {"API_KEY": api_key, "API_SECRET": secret_key, "PAPER": paper_trading}
        )

        # 统一使用 MomentumStrategy
        # 从 configs/universe.yaml 读取交易标的列表
        symbols = ["SPY"] # Default
        try:
            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH, "r") as f:
                    config = yaml.safe_load(f)
                    if config and "symbols" in config:
                        symbols = config["symbols"]
                        tm.log(f"Loaded symbols from config: {symbols}")
                    else:
                        tm.log("Config file found but no 'symbols' key. Using default.")
            else:
                tm.log(f"Config file not found at {CONFIG_PATH}. Using default symbols.")
        except Exception as e:
            tm.log(f"Error reading config file: {e}. Using default symbols.")

        tm.log(f"Starting strategy for symbols: {symbols}")

        strategy_params = {
            "symbols": symbols, 
            "sleeptime": "10S", # 实盘轮询间隔
            "lookback_period": 60
        }
        
        strategy = MomentumStrategy(broker=broker, parameters=strategy_params)

        # LumiBot: Create a Trader to manage the strategy
        trader = Trader()
        trader.add_strategy(strategy)

        tm.set_strategy(strategy)  # Store strategy ref

        # Run using the trader
        started = tm.start_strategy(runner=trader.run_all)
        return {"status": "started" if started else "failed"}
    except Exception as e:
        tm.log(f"Failed to setup strategy: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/api/stop")
async def stop_trading():
    tm.stop_strategy()
    return {"status": "stopping"}


@app.post("/api/run_backtest")
async def run_backtest(request: Request):
    # This might block, so we should run in thread again or use Celery (overkill).
    # MVP: Run in simple thread.
    params = await request.json()

    def _backtest_task():
        try:
            tm.log("Starting backtest...")
            # Setup Backtest

            backtesting_start = datetime.strptime(
                params.get("start_date", "2023-01-01"), "%Y-%m-%d"
            )
            backtesting_end = datetime.strptime(
                params.get("end_date", "2023-01-31"), "%Y-%m-%d"
            )
            
            # 支持多标的回测，前端传入逗号分隔的 symbol
            symbol_input = params.get("symbol", "SPY")
            symbols = [s.strip() for s in symbol_input.split(",") if s.strip()]
            if not symbols:
                symbols = ["SPY"]

            tm.log(f"Backtesting {symbols} from {backtesting_start} to {backtesting_end}")

            # Run Backtest
            try:
                # 使用统一的策略类
                # YahooDataBacktesting 会按需获取 symbols 的数据
                # benchmark_asset 设为第一个 symbol
                
                MomentumStrategy.backtest(
                    YahooDataBacktesting,
                    backtesting_start,
                    backtesting_end,
                    benchmark_asset=symbols[0], 
                    parameters={
                        "symbols": symbols,
                        "lookback_period": 60,
                        "sleeptime": "0S"
                    },
                )
                tm.log("Backtest finished successfully.")
            except Exception as e:
                tm.log(f"Backtest execution failed: {e}")
                traceback.print_exc()

        except Exception as e:
            tm.log(f"Backtest error: {e}")

    thread = threading.Thread(target=_backtest_task)
    thread.start()

    return {"status": "backtest_started"}


@app.on_event("startup")
async def startup_event():
    """Automatically start the strategy when the application starts."""
    logging.info("Initiating auto-start of trading strategy...")
    try:
        # Call the start_trading function directly
        result = await start_trading()
        logging.info(f"Auto-start strategy result: {result}")
    except Exception as e:
        logging.error(f"Failed to auto-start strategy: {e}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Poll state from TM
            state = tm.state
            await websocket.send_json(state)
            await asyncio.sleep(1)  # 1Hz update
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WS Error: {e}")
