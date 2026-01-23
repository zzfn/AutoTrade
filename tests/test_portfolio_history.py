"""测试 Alpaca PortfolioHistory API 的 base_value 返回格式"""

import asyncio
import os
import sys
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from typing import List, Optional

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


@dataclass
class PortfolioHistoryMock:
    """模拟 Alpaca PortfolioHistory 对象"""
    timestamp: List[int]
    equity: List[float]
    profit_loss: List[float]
    profit_loss_pct: List[float]
    base_value: List[float] | float  # 可能是列表或单个值

    def model_dump(self):
        """模拟 Pydantic 模型的 model_dump 方法"""
        return {
            'timestamp': self.timestamp,
            'equity': self.equity,
            'profit_loss': self.profit_loss,
            'profit_loss_pct': self.profit_loss_pct,
            'base_value': self.base_value,
        }


def create_mock_broker(base_value_format: str = 'list'):
    """创建模拟的 broker 和 API

    Args:
        base_value_format: 'list' 或 'float'，决定 base_value 的返回格式
    """
    # 创建模拟数据
    timestamps = [1705900000, 1705900300, 1705900600]
    equity = [100000.0, 100100.0, 100200.0]
    profit_loss = [0.0, 100.0, 200.0]
    profit_loss_pct = [0.0, 0.1, 0.2]

    # 根据 base_value_format 创建不同格式的 base_value
    if base_value_format == 'list':
        # Alpaca API 可能返回列表格式
        base_value = [100000.0, 100000.0, 100000.0]
    elif base_value_format == 'float':
        # Alpaca API 可能返回单个 float（这就是 WARNING 出现的情况）
        base_value = 100000.0
    else:
        raise ValueError(f"不支持的 base_value_format: {base_value_format}")

    # 创建 PortfolioHistory 对象
    portfolio_history = PortfolioHistoryMock(
        timestamp=timestamps,
        equity=equity,
        profit_loss=profit_loss,
        profit_loss_pct=profit_loss_pct,
        base_value=base_value
    )

    # 创建模拟 API
    mock_api = Mock()
    mock_api.get_portfolio_history = Mock(return_value=portfolio_history)

    # 创建模拟 broker
    mock_broker = Mock()
    mock_broker.api = mock_api

    return mock_broker


async def test_portfolio_history_with_list():
    """测试 base_value 为列表的情况"""
    print("\n=== 测试 1: base_value 为列表 ===")
    mock_broker = create_mock_broker('list')

    # 导入 server 模块（需要先导入才能 patch）
    from autotrade.web import server

    # Patch active_strategy 和 is_running
    with patch.object(server, 'active_strategy') as mock_strategy:
        mock_strategy.broker = mock_broker

        with patch.object(server, 'is_running', True):
            # 调用函数
            result = await server.get_portfolio_history(period="1D", timeframe="5Min")

            # 检查结果
            print(f"状态: {result['status']}")
            print(f"数据条数: {len(result['data'])}")
            if result['data']:
                print(f"第一条数据: {result['data'][0]}")
                print(f"base_value 是列表: {isinstance(result['data'][0].get('base_value'), float)}")


async def test_portfolio_history_with_float():
    """测试 base_value 为单个 float 的情况（触发 WARNING）"""
    print("\n=== 测试 2: base_value 为单个 float (触发 WARNING) ===")
    mock_broker = create_mock_broker('float')

    from autotrade.web import server

    # Patch active_strategy 和 is_running
    with patch.object(server, 'active_strategy') as mock_strategy:
        mock_strategy.broker = mock_broker

        with patch.object(server, 'is_running', True):
            # 调用函数
            result = await server.get_portfolio_history(period="1D", timeframe="5Min")

            # 检查结果
            print(f"状态: {result['status']}")
            print(f"数据条数: {len(result['data'])}")
            if result['data']:
                print(f"第一条数据: {result['data'][0]}")
                print(f"base_value 值: {result['data'][0].get('base_value')}")
                print(f"注意: base_value 应该为 None，因为触发了 WARNING")


async def test_real_api_format():
    """测试真实 Alpaca API 的返回格式（需要配置 API 密钥）"""
    print("\n=== 测试 3: 真实 Alpaca API 格式 ===")

    # 检查是否配置了 API 密钥
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_API_SECRET')

    if not api_key or not api_secret:
        print("未配置 Alpaca API 密钥，跳过真实 API 测试")
        print("如需测试，请在 .env 文件中配置:")
        print("  ALPACA_API_KEY=your_key")
        print("  ALPACA_API_SECRET=your_secret")
        return

    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetPortfolioHistoryRequest

        # 创建真实客户端
        trading_client = TradingClient(
            api_key=api_key,
            secret_key=api_secret,
            paper=True  # 使用模拟交易环境
        )

        # 调用真实 API
        history_request = GetPortfolioHistoryRequest(
            period="1D",
            timeframe="5Min"
        )

        portfolio_history = trading_client.get_portfolio_history(
            history_filter=history_request
        )

        # 检查返回的数据结构
        print(f"PortfolioHistory 类型: {type(portfolio_history)}")
        print(f"属性列表: {dir(portfolio_history)}")

        # 尝试不同的方式获取数据
        if hasattr(portfolio_history, 'model_dump'):
            raw_data = portfolio_history.model_dump()
            print(f"\n使用 model_dump() 获取数据:")
        elif hasattr(portfolio_history, '_raw'):
            raw_data = portfolio_history._raw
            print(f"\n使用 _raw 属性获取数据:")
        elif hasattr(portfolio_history, '__dict__'):
            raw_data = portfolio_history.__dict__
            print(f"\n使用 __dict__ 属性获取数据:")
        else:
            print("\n无法获取原始数据")
            return

        # 检查各字段类型
        print(f"\n字段类型检查:")
        print(f"  timestamp: {type(raw_data.get('timestamp'))}, 长度: {len(raw_data.get('timestamp', []))}")
        print(f"  equity: {type(raw_data.get('equity'))}, 长度: {len(raw_data.get('equity', []))}")
        print(f"  profit_loss: {type(raw_data.get('profit_loss'))}, 长度: {len(raw_data.get('profit_loss', []))}")
        print(f"  profit_loss_pct: {type(raw_data.get('profit_loss_pct'))}, 长度: {len(raw_data.get('profit_loss_pct', []))}")
        print(f"  base_value: {type(raw_data.get('base_value'))}", end='')

        base_value = raw_data.get('base_value')
        if isinstance(base_value, list):
            print(f", 长度: {len(base_value)}")
            if base_value:
                print(f"  base_value[0]: {base_value[0]}")
        else:
            print(f", 值: {base_value}")

        # 显示前几条数据
        if raw_data.get('timestamp'):
            print(f"\n前 3 条时间戳:")
            for i, ts in enumerate(raw_data['timestamp'][:3]):
                dt = datetime.fromtimestamp(ts)
                print(f"  {i+1}. {dt} (timestamp: {ts})")

    except Exception as e:
        print(f"真实 API 测试失败: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """运行所有测试"""
    print("=" * 60)
    print("Alpaca PortfolioHistory API base_value 格式测试")
    print("=" * 60)

    # 测试 1: base_value 为列表
    await test_portfolio_history_with_list()

    # 测试 2: base_value 为单个 float
    await test_portfolio_history_with_float()

    # 测试 3: 真实 API
    await test_real_api_format()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == '__main__':
    asyncio.run(main())
