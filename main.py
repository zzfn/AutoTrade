import os

import uvicorn


def is_running_in_docker() -> bool:
    """检测是否在 Docker/Kubernetes 容器中运行"""
    # 检测 /.dockerenv 文件
    if os.path.exists("/.dockerenv"):
        return True
    # 检测 Kubernetes 环境变量
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        return True
    # 检测环境变量显式设置
    if os.environ.get("AUTOTRADE_ENV", "").lower() in ("production", "docker", "kubernetes"):
        return True
    return False


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
