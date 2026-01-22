#!/usr/bin/env python3
"""
AutoTrade 主入口 - 最简 LumiBot 测试案例

仅用于测试 LumiBot 框架是否能正常运行
"""
# =============================================================================
# 关键：禁用 Python 输出缓冲，确保日志立即显示
# =============================================================================
import sys
import os

# 强制无缓冲输出（容器环境必须）
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None

# 立即打印启动信息（在任何复杂 import 之前）
print("[BOOT] AutoTrade main.py 开始执行...", flush=True)
print(f"[BOOT] Python: {sys.version}", flush=True)
print(f"[BOOT] 工作目录: {os.getcwd()}", flush=True)

import logging


# =============================================================================
# 日志配置
# =============================================================================
def setup_logging():
    """配置日志系统"""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # 配置根日志记录器
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    # 设置第三方库的日志级别
    logging.getLogger("lumibot").setLevel(logging.WARNING)
    logging.getLogger("alpaca").setLevel(logging.WARNING)

    return logging.getLogger(__name__)


# =============================================================================
# 最简单的 LumiBot 策略
# =============================================================================

# =============================================================================
# Import 追踪器（增强版）：找出到底卡在哪个模块
# =============================================================================
import sys
import builtins
import time
import traceback

# 尝试获取内存使用
try:
    import resource
    def get_memory_mb():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # macOS: KB -> MB
except ImportError:
    def get_memory_mb():
        return -1

_original_import = builtins.__import__
_import_depth = 0
_import_times = {}  # 记录每个模块的导入时间
_import_stack = []  # 导入栈

def _tracing_import(name, *args, **kwargs):
    global _import_depth
    start_time = time.time()
    indent = "  " * _import_depth
    
    # 追踪所有顶层导入和关键模块
    should_trace = (
        _import_depth == 0 or 
        name.startswith(('lumibot', 'pandas', 'numpy', 'sklearn', 'torch', 'tensorflow', 'alpaca', 'matplotlib')) or
        _import_depth <= 2  # 追踪前2层嵌套
    )
    
    if should_trace:
        mem = get_memory_mb()
        mem_str = f" [MEM:{mem:.0f}MB]" if mem > 0 else ""
        print(f"[IMPORT] {indent}>>> {name}{mem_str}", flush=True)
    
    _import_stack.append(name)
    _import_depth += 1
    
    try:
        result = _original_import(name, *args, **kwargs)
        elapsed = time.time() - start_time
        _import_times[name] = elapsed
        
        if should_trace and elapsed > 0.5:  # 超过 0.5 秒的导入标记为慢
            print(f"[IMPORT] {indent}<<< {name} ⚠️ SLOW ({elapsed:.2f}s)", flush=True)
        elif should_trace:
            print(f"[IMPORT] {indent}<<< {name} ({elapsed:.2f}s)", flush=True)
        
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[IMPORT] {indent}!!! {name} FAILED after {elapsed:.2f}s: {e}", flush=True)
        print(f"[IMPORT] Import stack: {' -> '.join(_import_stack)}", flush=True)
        traceback.print_exc()
        raise
    finally:
        _import_depth -= 1
        _import_stack.pop()

# 启用 import 追踪
builtins.__import__ = _tracing_import
print("[BOOT] Import 追踪已启用（增强版：显示耗时和内存）", flush=True)
print(f"[BOOT] 初始内存: {get_memory_mb():.0f}MB", flush=True)

# 开始 import
try:
    print("\n[BOOT] ========== Step 1: import matplotlib ==========", flush=True)
    t0 = time.time()
    import matplotlib
    print(f"[BOOT] Step 1 done: matplotlib {matplotlib.__version__} ({time.time()-t0:.2f}s)", flush=True)
    
    print("\n[BOOT] ========== Step 2: import matplotlib.font_manager ==========", flush=True)
    t0 = time.time()
    import matplotlib.font_manager
    print(f"[BOOT] Step 2 done ({time.time()-t0:.2f}s)", flush=True)
    
    print("\n[BOOT] ========== Step 3: import lumibot.strategies.strategy ==========", flush=True)
    print(f"[BOOT] 当前内存: {get_memory_mb():.0f}MB", flush=True)
    t0 = time.time()
    from lumibot.strategies.strategy import Strategy
    print(f"[BOOT] Step 3 done ({time.time()-t0:.2f}s)", flush=True)
    
except Exception as e:
    print(f"\n[BOOT] ❌ Import 失败: {e}", flush=True)
    traceback.print_exc()
    # 打印最慢的导入
    if _import_times:
        print("\n[BOOT] 导入耗时排行（前10）:", flush=True)
        sorted_times = sorted(_import_times.items(), key=lambda x: x[1], reverse=True)[:10]
        for name, t in sorted_times:
            print(f"  {t:.2f}s - {name}", flush=True)
    sys.exit(1)

# 关闭追踪
builtins.__import__ = _original_import
print("\n[BOOT] Import 追踪已关闭", flush=True)

# 打印导入统计
if _import_times:
    print("\n[BOOT] 导入耗时排行（前10）:", flush=True)
    sorted_times = sorted(_import_times.items(), key=lambda x: x[1], reverse=True)[:10]
    for name, t in sorted_times:
        print(f"  {t:.2f}s - {name}", flush=True)

print(f"[BOOT] 当前内存: {get_memory_mb():.0f}MB", flush=True)

print("\n[BOOT] ========== Step 4: import lumibot.brokers ==========", flush=True)
t0 = time.time()
from lumibot.brokers import Alpaca
print(f"[BOOT] Step 4 done ({time.time()-t0:.2f}s)", flush=True)

print("\n[BOOT] ========== Step 5: import lumibot.traders ==========", flush=True)
t0 = time.time()
from lumibot.traders import Trader
print(f"[BOOT] Step 5 done: 所有 import 完成! ({time.time()-t0:.2f}s)", flush=True)
print(f"[BOOT] 最终内存: {get_memory_mb():.0f}MB", flush=True)


class SimpleTestStrategy(Strategy):
    """最简单的测试策略：只打印状态，不交易"""

    parameters = {
        "symbol": "SPY",
        "sleeptime": "1M",  # 每分钟检查一次
    }

    def initialize(self):
        self.sleeptime = self.parameters["sleeptime"]
        self.logger = logging.getLogger(f"{__name__}.SimpleTestStrategy")

        self.logger.info("=" * 50)
        self.logger.info("SimpleTestStrategy 启动成功！")
        self.logger.info(f"交易标的: {self.symbol}")
        self.logger.info(f"检查频率: {self.sleeptime}")
        self.logger.info("=" * 50)

    def on_trading_iteration(self):
        """每次交易迭代"""
        try:
            price = self.get_last_price(self.symbol)
            cash = self.get_cash()
            value = self.portfolio_value

            self.logger.info(
                f"[状态] {self.symbol}=${price:.2f} | 现金=${cash:.2f} | 总资产=${value:.2f}"
            )
        except Exception as e:
            self.logger.error(f"获取数据出错: {e}", exc_info=True)

    def before_market_opens(self):
        """市场开盘前"""
        self.logger.info("市场即将开盘...")

    def after_market_closes(self):
        """市场收盘后"""
        self.logger.info("市场已收盘")


def is_running_in_docker() -> bool:
    """检测是否在 Docker/Kubernetes 容器中运行"""
    if os.path.exists("/.dockerenv"):
        return True
    if os.environ.get("KUBERNAT_ES_SERVICE_HOST"):
        return True
    if os.environ.get("AUTOTRADE_ENV", "").lower() in ("production", "docker", "kubernetes"):
        return True
    return False


def main():
    """主函数"""
    # 初始化日志
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("AutoTrade - 最简 LumiBot 测试")
    logger.info("=" * 60)

    # 检测运行环境
    in_docker = is_running_in_docker()
    if in_docker:
        logger.info("🐳 Docker 模式")
    else:
        logger.info("🔧 开发模式")

    # 检查环境变量
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_API_SECRET")

    if not api_key or not secret_key:
        logger.error("未设置 Alpaca 凭证")
        logger.info("请设置以下环境变量:")
        logger.info("  export ALPACA_API_KEY=your_key")
        logger.info("  export ALPACA_API_SECRET=your_secret")
        logger.info("或在 .env 文件中配置:")
        logger.info("  ALPACA_API_KEY=your_key")
        logger.info("  ALPACA_API_SECRET=your_secret")
        sys.exit(1)

    logger.info("凭证已加载 (Paper Trading)")
    logger.debug(f"API Key: {api_key[:10]}...{api_key[-4:]}")

    # 设置信号处理
    import signal

    def signal_handler(sig, frame):
        logger.info("收到终止信号，正在停止...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # 创建 Broker
        logger.info("正在连接 Alpaca...")
        broker = Alpaca(
            {
                "API_KEY": api_key,
                "API_SECRET": secret_key,
                "PAPER": True,  # Paper Trading
            }
        )
        logger.info("Alpaca 连接成功")

        # 创建策略
        logger.info("正在创建策略...")
        strategy = SimpleTestStrategy(
            broker=broker,
            parameters={
                "symbol": "SPY",
                "sleeptime": "1M",
            },
        )
        logger.info(f"策略已创建: {strategy.__class__.__name__}")

        # 创建 Trader
        trader = Trader()
        trader.add_strategy(strategy)
        logger.info("策略已添加到 Trader")

        logger.info("-" * 60)
        logger.info("开始运行策略... (Ctrl+C 停止)")
        logger.info("=" * 60)

        # 运行策略（阻塞）
        trader.run_all()

    except KeyboardInterrupt:
        logger.warning("策略已手动停止")
    except Exception as e:
        logger.error(f"运行出错: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
