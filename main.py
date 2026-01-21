"""
AutoTrade 主入口点。

使用 UI/Strategy 分离架构：
- 后台线程：FastAPI + Uvicorn (UI 服务器)
- 主线程：LumiBot 策略执行
"""
import sys

# ==============================================================================
# 加速 matplotlib 初始化（必须在其他导入之前！）
# ==============================================================================
# 使用非交互式后端，跳过不必要的 GUI 初始化
import os
import matplotlib
matplotlib.use('Agg')
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache' # 确保有地方写缓存
os.environ.setdefault("MPLBACKEND", "Agg")
# 禁用字体管理器的自动扫描日志
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)


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
    print("=" * 60)
    print("AutoTrade - UI/Strategy 分离模式")
    print("=" * 60)
    
    in_docker = is_running_in_docker()
    if in_docker:
        print("🐳 Docker 模式")
    else:
        print("🔧 开发模式")
    
    # 导入 UI 服务器和策略运行器
    from autotrade.web.server import start_server_background, stop_server_background
    from autotrade.strategies.runner import run_strategy_main, logger
    import signal

    # 设置信号处理器
    def signal_handler(sig, frame):
        logger.info("收到终止信号，正在清理...")
        # 设置策略停止标志
        import autotrade.strategies.runner as runner_module
        runner_module.is_running = False
        stop_server_background()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 获取配置
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    # 1. 启动 UI 服务器（后台线程）
    server_thread = start_server_background(host=host, port=port)
    logger.info(f"UI 服务器已在后台启动: http://{host}:{port}")
    print("-" * 60)
    
    # 2. 在主线程运行策略（阻塞）
    logger.info("正在主线程启动交易策略...")
    result = run_strategy_main()
    logger.info(f"策略运行结果: {result}")
    
    # 3. 清理
    stop_server_background()
    logger.info("所有服务已停止。")
