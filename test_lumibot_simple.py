#!/usr/bin/env python3
"""
最简单的 LumiBot 测试案例

运行方式:
    python test_lumibot_simple.py

测试内容:
- 简单的买入持有策略
- 只依赖 Alpaca paper trading
- 无 ML 模型，无 qlib
"""

import os
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from lumibot.brokers import Alpaca


class SimpleStrategy(Strategy):
    """最简单的策略：启动后什么都不做，只打印日志"""

    parameters = {
        "symbol": "SPY",
    }

    def initialize(self):
        """策略初始化"""
        self.sleeptime = "1H"  # 每小时检查一次
        self.log_message("SimpleStrategy 初始化成功")

    def on_trading_iteration(self):
        """每次交易迭代"""
        symbol = self.symbol
        price = self.get_last_price(symbol)
        cash = self.get_cash()
        value = self.portfolio_value

        self.log_message(f"价格: {symbol} = ${price:.2f}, 现金: ${cash:.2f}, 总资产: ${value:.2f}")


def main():
    """主函数"""
    print("=" * 60)
    print("LumiBot 最简测试案例")
    print("=" * 60)

    # 1. 检查环境变量
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_API_SECRET")

    if not api_key or not secret_key:
        print("错误: 未设置环境变量 ALPACA_API_KEY 和 ALPACA_API_SECRET")
        print("请在 .env 文件中设置或运行:")
        print("  export ALPACA_API_KEY=your_key")
        print("  export ALPACA_API_SECRET=your_secret")
        return

    # 2. 创建 Broker
    print("正在连接 Alpaca Paper Trading...")
    broker = Alpaca({
        "API_KEY": api_key,
        "API_SECRET": secret_key,
        "PAPER": True,  # 使用模拟交易
    })

    # 3. 创建策略
    strategy = SimpleStrategy(
        broker=broker,
        parameters={"symbol": "SPY"}
    )

    # 4. 创建 Trader
    trader = Trader()
    trader.add_strategy(strategy)

    # 5. 运行
    print("开始运行策略 (Ctrl+C 停止)...")
    print("-" * 60)

    try:
        trader.run_all()
    except KeyboardInterrupt:
        print("\n收到中断信号，正在停止...")
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
