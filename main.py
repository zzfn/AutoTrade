"""
AutoTrade 主入口点。

使用 UI/Strategy 分离架构：
- 后台线程：FastAPI + Uvicorn (UI 服务器)
- 主线程：LumiBot 策略执行
"""
import logging
import os
import sys
from datetime import datetime


class ETFormatter(logging.Formatter):
    """自定义日志格式化器，使用美东时间"""

    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        self.tz = None

    def formatTime(self, record, datefmt=None):
        # 创建 UTC 时间
        ct = self.converter(record.created)
        # 转换为 datetime 对象
        dt = datetime.utcfromtimestamp(record.created)

        try:
            # 转换为美东时间
            from autotrade.utils.timezone import utc_to_et
            et_dt = utc_to_et(dt)
            # 格式化时间
            return et_dt.strftime("%Y-%m-%d %H:%M:%S") + " ET"
        except Exception:
            # 如果转换失败，使用默认格式
            return dt.strftime("%Y-%m-%d %H:%M:%S")


def setup_main_logger() -> logging.Logger:
    """配置主程序日志记录器"""
    logger = logging.getLogger("autotrade.main")
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler
    if not logger.handlers:
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # 日志格式：包含时间戳、级别、模块名、消息
        formatter = ETFormatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def is_running_in_docker() -> bool:
    """检测是否在 Docker/Kubernetes 容器中运行"""
    if os.path.exists("/.dockerenv"):
        return True
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        return True
    if os.environ.get("AUTOTRADE_ENV", "").lower() in ("production", "docker", "kubernetes"):
        return True
    return False


if __name__ == "__main__":
    # 初始化日志
    logger = setup_main_logger()

    # 启动信息
    logger.info("AutoTrade - UI/Strategy 分离模式")

    # 检测运行环境
    in_docker = is_running_in_docker()
    if in_docker:
        logger.info("运行环境: Docker 容器")
    else:
        logger.info("运行环境: 本地开发环境")

    # 导入并运行分离架构
    from autotrade.web.server import (
        start_server_background,
        run_strategy_main,
        stop_server_background,
    )
    import signal

    # 设置信号处理器
    def signal_handler(sig, _frame):
        logger.info("收到终止信号 %s，正在清理资源...", sig)
        import autotrade.web.server as server_module
        server_module.is_running = False
        stop_server_background()
        logger.info("程序已退出")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 获取服务配置
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    try:
        # 1. 启动 UI 服务器（后台线程）
        logger.info("正在启动 UI 服务器...")
        server_thread = start_server_background(host=host, port=port)
        logger.info("UI 服务器已启动: http://%s:%s", host, port)

        # 2. 在主线程运行策略（阻塞）
        logger.info("正在主线程启动交易策略...")
        result = run_strategy_main()
        logger.info("策略运行结果: %s", result)

    except Exception as e:
        logger.exception("程序运行异常: %s", e)
        sys.exit(1)

    finally:
        # 3. 清理资源
        logger.info("正在停止所有服务...")
        stop_server_background()
        logger.info("所有服务已停止")
