import asyncio
import logging
import os
import signal
import threading
from contextlib import asynccontextmanager
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理交易策略生命周期的上下文管理器"""
    logging.info("正在执行交易策略生命周期启动...")
    try:
        # 启动策略
        result = tm.initialize_and_start()
        logging.info(f"策略启动结果: {result}")
    except Exception as e:
        logging.error(f"策略启动失败: {e}")
    
    yield  # 这里是应用运行期间
    
    # 应用关闭时的清理逻辑
    logging.info("正在执行策略生命周期关闭清理...")
    tm.stop_strategy()


app = FastAPI(lifespan=lifespan)


# ... existing imports ...

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(BASE_DIR, "ui")
TEMPLATES_DIR = os.path.join(UI_DIR, "templates")
STATIC_DIR = os.path.join(UI_DIR, "static")

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


@app.post("/api/run_backtest")
async def run_backtest(request: Request):
    params = await request.json()
    return tm.run_backtest(params)


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
