#!/usr/bin/env python
"""
初始化 Qlib 数据脚本

任务 1.5: 从 Alpaca/YFinance 获取美股数据并转换为 Qlib 格式

使用方法:
    uv run python scripts/init_qlib_data.py --symbols SPY,AAPL,MSFT --days 730
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 确保可以导入项目模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from autotrade.research.data import QlibDataAdapter
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(description="初始化 Qlib 数据")
    parser.add_argument(
        "--symbols",
        type=str,
        default="SPY,AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA,JPM,V",
        help="股票代码列表，用逗号分隔 (默认: 10只热门股票)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=730,
        help="获取的历史天数 (默认: 730 = 2年)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="开始日期 (格式: YYYY-MM-DD，优先于 --days)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="结束日期 (格式: YYYY-MM-DD，默认: 今天)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="datasets",
        help="数据存储目录 (默认: datasets)",
    )
    parser.add_argument(
        "--update-mode",
        type=str,
        choices=["replace", "append"],
        default="replace",
        help="更新模式: replace=替换, append=追加 (默认: replace)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        choices=["1d", "1h"],
        default="1d",
        help="数据频率: 1d 或 1h (默认: 1d)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 解析参数
    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    else:
        end_date = datetime.now()

    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    else:
        start_date = end_date - timedelta(days=args.days)

    logger.info("=" * 50)
    logger.info("Qlib 数据初始化")
    logger.info("=" * 50)
    logger.info(f"股票列表: {symbols}")
    logger.info(f"日期范围: {start_date.date()} - {end_date.date()}")
    logger.info(f"数据目录: {args.data_dir}")
    logger.info(f"更新模式: {args.update_mode}")
    logger.info(f"数据频率: {args.interval}")
    logger.info("=" * 50)

    # 创建适配器并获取数据
    adapter = QlibDataAdapter(data_dir=args.data_dir, interval=args.interval)

    try:
        result = adapter.fetch_and_store(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            update_mode=args.update_mode,
        )

        if result["status"] == "success":
            logger.info("=" * 50)
            logger.info("数据获取成功!")
            logger.info(f"处理的股票: {result['processed_symbols']}")
            logger.info(f"总记录数: {result['total_records']}")
            logger.info("=" * 50)

            # 验证数据
            logger.info("\n数据验证:")
            for symbol in result["processed_symbols"]:
                date_range = adapter.get_date_range(symbol)
                if date_range:
                    logger.info(
                        f"  {symbol}: {date_range[0].date()} - {date_range[1].date()}"
                    )
        else:
            logger.error(f"数据获取失败: {result.get('message', '未知错误')}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"初始化失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
