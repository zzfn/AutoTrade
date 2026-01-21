#!/usr/bin/env python3
"""
LumiBot 回测测试案例（不需要 API 密钥）

运行方式:
    python test_lumibot_backtest.py

测试内容:
- 使用历史数据回测
- 不需要 Alpaca API 密钥
- 最简单的买入持有策略
"""

from datetime import datetime, timedelta
from lumibot.strategies.strategy import Strategy
from lumibot.backtesting import YahooDataBacktesting
from lumibot.traders import Trader


class BuyHoldStrategy(Strategy):
    """最简单的买入持有策略"""

    parameters = {
        "symbol": "SPY",
    }

    def initialize(self):
        """初始化"""
        self.sleeptime = "1D"  # 每天检查一次
        self.log_message(f"策略初始化: 买入 {self.symbol}")

    def on_trading_iteration(self):
        """每次交易迭代"""
        # 如果没有持仓，就买入
        position = self.get_position(self.symbol)
        if position is None or position.quantity == 0:
            price = self.get_last_price(self.symbol)
            cash = self.get_cash()

            # 使用 95% 的现金买入
            if cash > 100:
                qty = int((cash * 0.95) / price)
                if qty > 0:
                    order = self.create_order(self.symbol, qty, "buy")
                    self.submit_order(order)
                    self.log_message(f"买入 {qty} 股 {self.symbol} @ ${price:.2f}")

        # 打印当前状态
        price = self.get_last_price(self.symbol)
        value = self.portfolio_value
        self.log_message(f"价格: ${price:.2f}, 总资产: ${value:.2f}")


def main():
    """主函数 - 运行回测"""
    print("=" * 60)
    print("LumiBot 回测测试")
    print("=" * 60)

    # 设置回测时间范围（最近 30 天）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    print(f"回测期间: {start_date.date()} 到 {end_date.date()}")

    # 创建策略
    strategy = BuyHoldStrategy(
        broker=None,  # 回测不需要 broker
        parameters={"symbol": "SPY"},
    )

    # 运行回测
    print("开始回测...")
    print("-" * 60)

    try:
        # 使用 Yahoo 数据进行回测
        result = strategy.backtest(
            YahooDataBacktesting,
            start_date=start_date,
            end_date=end_date,
        )
        print("-" * 60)
        print("回测完成！")
        print(f"最终资产: ${result['portfolio_value']:.2f}")
        print(f"总收益率: {result['total_return']:.2%}")
    except Exception as e:
        print(f"回测出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()