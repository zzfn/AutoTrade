#!/usr/bin/env python3
"""
AutoTrade 主入口 - 最简 LumiBot 测试案例

仅用于测试 LumiBot 框架是否能正常运行
"""
import os
import sys

# =============================================================================
# 加速 matplotlib 初始化（必须在其他导入之前！）
# =============================================================================
from unittest.mock import MagicMock
sys.modules["matplotlib.font_manager"] = MagicMock()
import matplotlib
matplotlib.font_manager = MagicMock()
matplotlib.use('Agg')
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
os.environ.setdefault("MPLBACKEND", "Agg")

import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)


# =============================================================================
# 最简单的 LumiBot 策略
# =============================================================================
from lumibot.strategies.strategy import Strategy
from lumibot.brokers import Alpaca
from lumibot.traders import Trader


class SimpleTestStrategy(Strategy):
    """最简单的测试策略：只打印状态，不交易"""

    parameters = {
        "symbol": "SPY",
        "sleeptime": "1M",  # 每分钟检查一次
    }

    def initialize(self):
        self.sleeptime = self.parameters["sleeptime"]
        self.log_message("=" * 50)
        self.log_message("SimpleTestStrategy 启动成功！")
        self.log_message(f"交易标的: {self.symbol}")
        self.log_message("=" * 50)

    def on_trading_iteration(self):
        """每次交易迭代"""
        try:
            price = self.get_last_price(self.symbol)
            cash = self.get_cash()
            value = self.portfolio_value

            self.log_message(f"[状态] {self.symbol}=${price:.2f} | 现金=${cash:.2f} | 总资产=${value:.2f}")
        except Exception as e:
            self.log_message(f"获取数据出错: {e}")

    def before_market_opens(self):
        """市场开盘前"""
        self.log_message("市场即将开盘...")

    def after_market_closes(self):
        """市场收盘后"""
        self.log_message("市场已收盘")


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
    print("=" * 60)
    print("AutoTrade - 最简 LumiBot 测试")
    print("=" * 60)

    # 检测运行环境
    in_docker = is_running_in_docker()
    if in_docker:
        print("🐳 Docker 模式")
    else:
        print("🔧 开发模式")

    # 检查环境变量
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_API_SECRET")

    if not api_key or not secret_key:
        print("\n❌ 错误: 未设置 Alpaca 凭证")
        print("\n请设置以下环境变量:")
        print("  export ALPACA_API_KEY=your_key")
        print("  export ALPACA_API_SECRET=your_secret")
        print("\n或在 .env 文件中配置:")
        print("  ALPACA_API_KEY=your_key")
        print("  ALPACA_API_SECRET=your_secret")
        sys.exit(1)

    print(f"\n✓ 凭证已加载 (Paper Trading)")
    print("-" * 60)

    # 设置信号处理
    import signal
    def signal_handler(sig, frame):
        print("\n\n收到终止信号，正在停止...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # 创建 Broker
        print("正在连接 Alpaca...")
        broker = Alpaca({
            "API_KEY": api_key,
            "API_SECRET": secret_key,
            "PAPER": True,  # Paper Trading
        })

        # 创建策略
        strategy = SimpleTestStrategy(
            broker=broker,
            parameters={
                "symbol": "SPY",
                "sleeptime": "1M",
            }
        )

        # 创建 Trader
        trader = Trader()
        trader.add_strategy(strategy)

        print("✓ 策略已加载")
        print("-" * 60)
        print("开始运行策略... (Ctrl+C 停止)")
        print("=" * 60)

        # 运行策略（阻塞）
        trader.run_all()

    except KeyboardInterrupt:
        print("\n策略已手动停止")
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
