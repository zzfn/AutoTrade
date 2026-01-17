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
from autotrade.execution.strategies.momentum_strategy import MomentumStrategy
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
app.mount(
    "/reports",
    StaticFiles(directory=os.path.join(os.path.dirname(BASE_DIR), "logs")),
    name="reports",
)
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


# ==================== ML 策略相关 API ====================


@app.get("/api/strategy/config")
async def get_strategy_config():
    """获取当前策略配置"""
    return tm.get_strategy_config()


@app.post("/api/strategy/type")
async def set_strategy_type(request: Request):
    """设置策略类型"""
    data = await request.json()
    strategy_type = data.get("strategy_type", "momentum")
    return tm.set_strategy_type(strategy_type)


@app.post("/api/strategy/start")
async def start_strategy():
    """手动启动策略"""
    return tm.initialize_and_start()


@app.post("/api/strategy/stop")
async def stop_strategy():
    """手动停止策略"""
    return tm.stop_strategy()


@app.post("/api/strategy/ml_config")
async def set_ml_config(request: Request):
    """设置 ML 策略配置"""
    config = await request.json()
    return tm.set_ml_config(config)


@app.get("/api/models")
async def list_models():
    """列出所有可用的 ML 模型"""
    return {"status": "success", "models": tm.list_models()}


@app.get("/api/models/current")
async def get_current_model():
    """获取当前选择的模型"""
    return tm.get_current_model()


@app.post("/api/models/select")
async def select_model(request: Request):
    """选择要使用的模型"""
    data = await request.json()
    model_name = data.get("model_name")
    if not model_name:
        return {"status": "error", "message": "缺少 model_name 参数"}
    return tm.select_model(model_name)


@app.post("/api/models/rolling_update")
async def start_rolling_update(request: Request):
    """启动 Rolling 模型更新"""
    try:
        config = await request.json()
    except Exception:
        config = None
    return tm.start_rolling_update(config)


@app.get("/api/models/rolling_update/status")
async def get_rolling_update_status():
    """获取 Rolling 更新状态"""
    return tm.get_rolling_update_status()


# ==================== 模型管理页面 ====================


@app.get("/models", response_class=HTMLResponse)
async def models_page(request: Request):
    """模型管理页面"""
    return templates.TemplateResponse("models.html", {"request": request})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Poll state from TM
            state = tm.state
            # 添加策略配置信息
            state["strategy_config"] = tm.get_strategy_config()
            state["rolling_update_status"] = tm.get_rolling_update_status()
            await websocket.send_json(state)
            await asyncio.sleep(1)  # 1Hz update
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WS Error: {e}")
