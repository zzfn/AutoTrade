import os

import uvicorn


def is_running_in_docker() -> bool:
    """检测是否在 Docker 容器中运行"""
    return os.path.exists("/.dockerenv")


if __name__ == "__main__":
    print("Starting AutoTrade Web Server (FastAPI + React)...")
    reload = not is_running_in_docker()
    if reload:
        print("🔧 Development mode: hot reload enabled")
    else:
        print("🐳 Docker mode: hot reload disabled")
    uvicorn.run(
        "autotrade.web.server:app",
        host="0.0.0.0",
        port=8000,
        reload=reload,
    )
