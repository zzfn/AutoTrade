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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

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

# Monkey patch Alpaca broker to avoid AttributeError: 'Alpaca' object has no attribute 'process_pending_orders'
# this is a known issue in some version of lumibot during shutdown.
from lumibot.brokers import Alpaca
if not hasattr(Alpaca, "process_pending_orders"):
    def _process_pending_orders_patch(self, *args, **kwargs):
        pass
    Alpaca.process_pending_orders = _process_pending_orders_patch


@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理交易策略生命周期的上下文管理器"""
    logger.info("正在执行交易策略生命周期启动...")
    
    # 将初始化放在后台任务中，以免阻塞服务器启动
    async def startup_task():
        try:
            # 等待一小会儿，确保服务器已经开始监听
            await asyncio.sleep(1)
            logger.info("后台线程初始化交易策略...")
            # 使用 run_in_executor 避免同步初始化代码阻塞 asyncio 事件循环
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, tm.initialize_and_start)
            logger.info(f"策略启动结果: {result}")
        except Exception as e:
            logger.error(f"后台策略启动失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # 启动后台初始化任务
    init_task = asyncio.create_task(startup_task())
    
    yield  # 这里是应用运行期间

    # 应用关闭时的清理逻辑
    logger.info("正在执行策略生命周期关闭清理...")
    init_task.cancel()  # 如果还在初始化则取消
    try:
        # 使用 to_thread 在同步代码执行期间不阻塞事件循环，并设置 5 秒超时
        await asyncio.wait_for(asyncio.to_thread(tm.stop_strategy), timeout=5.0)
    except asyncio.TimeoutError:
        logger.warning("策略清理超时，强制关闭...")
    except Exception as e:
        logger.error(f"清理过程中发生错误: {e}")


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
    return templates.TemplateResponse(request, "index.html")


@app.get("/backtest", response_class=HTMLResponse)
async def read_backtest(request: Request):
    return templates.TemplateResponse(request, "backtest.html")


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


@app.post("/api/models/delete")
async def delete_model(request: Request):
    """删除模型"""
    data = await request.json()
    model_name = data.get("model_name")
    if not model_name:
        return {"status": "error", "message": "缺少 model_name 参数"}
    return tm.delete_model(model_name)


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


@app.post("/api/data/sync")
async def start_data_sync(request: Request):
    """启动数据同步"""
    try:
        config = await request.json()
    except Exception:
        config = None
    return tm.start_data_sync(config)


@app.get("/api/data/sync/status")
async def get_data_sync_status():
    """获取数据同步状态"""
    return tm.get_data_sync_status()


# ==================== 模型管理页面 ====================


@app.get("/models", response_class=HTMLResponse)
async def models_page(request: Request):
    """模型管理页面"""
    return templates.TemplateResponse(request, "models.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected")
    try:
        while True:
            # Poll state and create a combined dictionary for the frontend
            # We copy tm.state members to avoid modification while serializing
            try:
                # Granular copy of state members
                state = {
                    "status": tm.state.get("status", "unknown"),
                    "logs": list(tm.state.get("logs", [])),
                    "orders": [dict(o) for o in tm.state.get("orders", [])],
                    "portfolio": {
                        "cash": tm.state.get("portfolio", {}).get("cash", 0.0),
                        "value": tm.state.get("portfolio", {}).get("value", 0.0),
                        "positions": [dict(p) for p in tm.state.get("portfolio", {}).get("positions", [])]
                    },
                    "market_status": tm.state.get("market_status", "unknown"),
                    "last_update": tm.state.get("last_update"),
                    "strategy_config": tm.get_strategy_config(),
                    "rolling_update_status": tm.get_rolling_update_status().copy() if isinstance(tm.get_rolling_update_status(), dict) else tm.get_rolling_update_status(),
                    "data_sync_status": tm.get_data_sync_status().copy() if isinstance(tm.get_data_sync_status(), dict) else tm.get_data_sync_status()
                }
                
                # Use send_json which handles the serialization
                await websocket.send_json(state)
            except (WebSocketDisconnect, RuntimeError):
                logger.info("WebSocket connection closed")
                break
            except Exception as e:
                logger.error(f"Error preparing or sending WS data: {e}")
                # Don't break here unless it's a critical one, but log it
            
            await asyncio.sleep(1)  # 1Hz update
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WS Connection Error: {e}")
