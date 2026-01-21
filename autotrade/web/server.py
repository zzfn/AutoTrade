"""
Web server for AutoTrade UI.

FastAPI application with WebSocket endpoint for real-time updates.
Strategy execution is handled separately from this UI server.
"""

import asyncio
import logging
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ==============================================================================
# REGION: State Management
# ==============================================================================

_state_lock = threading.Lock()

# Global state for UI updates
state: dict[str, Any] = {
    "status": "disconnected",
    "logs": [],
    "last_update": None,
}

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


# ==============================================================================
# REGION: FastAPI Application
# ==============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage FastAPI application lifecycle."""
    logger.info("FastAPI 服务器启动...")
    yield
    logger.info("FastAPI 服务器关闭...")


app = FastAPI(lifespan=lifespan)


# Paths
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


# ==============================================================================
# REGION: Routes
# ==============================================================================


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """主页 - 仪表板"""
    return templates.TemplateResponse(request, "index.html")


@app.get("/backtest", response_class=HTMLResponse)
async def read_backtest(request: Request):
    """回测页面"""
    return templates.TemplateResponse(request, "backtest.html")


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
    """WebSocket端点 - 实时状态更新"""
    await websocket.accept()
    logger.info("WebSocket client connected")
    try:
        while True:
            try:
                ws_state = {
                    "status": state.get("status", "unknown"),
                    "logs": list(state.get("logs", [])),
                    "last_update": state.get("last_update"),
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
# REGION: Server Control
# ==============================================================================

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


# ==============================================================================
# REGION: Main Entry Point
# ==============================================================================


if __name__ == "__main__":
    """主入口点：仅启动 UI 服务器。

    使用方式：
        python autotrade/web/server.py

    或使用 uv：
        uv run python autotrade/web/server.py

    注意：
        此文件仅提供 UI 服务器功能。
        策略执行应该在独立的进程或服务中运行。
    """
    import signal
    import sys

    # 设置信号处理器
    def signal_handler(sig, frame):
        logger.info("收到终止信号，正在清理...")
        stop_server_background()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 启动 UI 服务器
    logger.info("=" * 60)
    logger.info("AutoTrade - UI Server")
    logger.info("=" * 60)

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    server_thread = start_server_background(host=host, port=port)

    logger.info(f"UI 服务器已启动: http://{host}:{port}")
    logger.info("-" * 60)

    # 保持主线程运行
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("收到中断信号")
    finally:
        stop_server_background()
        logger.info("UI 服务器已停止。")
