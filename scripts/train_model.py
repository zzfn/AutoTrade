#!/usr/bin/env python
"""
模型训练脚本

任务 3.5: 创建训练脚本，支持配置文件和命令行参数

使用方法:
    # 基础训练
    uv run python scripts/train_model.py --symbols SPY,AAPL,MSFT

    # 使用配置文件
    uv run python scripts/train_model.py --config configs/qlib_ml_config.yaml

    # Walk-Forward 验证
    uv run python scripts/train_model.py --symbols SPY --walk-forward
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import yaml

# 确保可以导入项目模块
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

from autotrade.research.data import QlibDataAdapter
from autotrade.research.features import QlibFeatureGenerator
from autotrade.research.models import LightGBMTrainer, ModelManager
from autotrade.research.models.trainer import WalkForwardValidator


def parse_args():
    parser = argparse.ArgumentParser(description="训练 ML 模型")

    # 数据参数
    parser.add_argument(
        "--symbols",
        type=str,
        default="SPY,AAPL,MSFT,GOOGL,AMZN",
        help="股票代码列表，用逗号分隔",
    )
    parser.add_argument(
        "--train-days",
        type=int,
        default=504,
        help="训练数据天数 (默认: 504 = 2年)",
    )
    parser.add_argument(
        "--valid-days",
        type=int,
        default=63,
        help="验证数据天数 (默认: 63 = 3个月)",
    )

    # 模型参数
    parser.add_argument(
        "--model-name",
        type=str,
        default="lightgbm",
        help="模型名称 (默认: lightgbm)",
    )
    parser.add_argument(
        "--num-boost-round",
        type=int,
        default=500,
        help="迭代次数 (默认: 500)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="学习率 (默认: 0.05)",
    )

    # 目标变量
    parser.add_argument(
        "--target-horizon",
        type=int,
        default=5,
        help="预测时间范围（天数）(默认: 5)",
    )

    # 验证模式
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="使用 Walk-Forward 验证",
    )

    # 配置文件
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径 (YAML)",
    )

    # 频率
    parser.add_argument(
        "--interval",
        type=str,
        default=None,
        help="数据频率 (1min, 1h 或 1d)，默认从配置文件读取",
    )

    # 其他
    parser.add_argument(
        "--data-dir",
        type=str,
        default="datasets",
        help="Qlib 数据目录 (默认: datasets)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="模型保存目录 (默认: models)",
    )
    parser.add_argument(
        "--set-current",
        action="store_true",
        help="训练完成后设置为当前模型",
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_data(
    symbols: list[str],
    data_dir: str,
    train_days: int,
    valid_days: int,
    target_horizon: int,
    interval: str = "1d",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    准备训练和验证数据

    Returns:
        X_train, X_valid, y_train, y_valid
    """
    logger.info(f"准备数据 (interval={interval})...")

    # 加载数据
    adapter = QlibDataAdapter(data_dir=data_dir, interval=interval)

    # 检查是否有数据
    available_symbols = adapter.get_available_symbols()
    missing_symbols = [s for s in symbols if s not in available_symbols]

    if missing_symbols:
        logger.warning(f"以下股票没有数据: {missing_symbols}")
        logger.info("尝试获取数据...")

        # 获取数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=train_days + valid_days + 100)

        adapter.fetch_and_store(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
        )

    # 加载数据
    df = adapter.load_data(symbols)

    if df.empty:
        raise ValueError("没有可用的数据")

    logger.info(f"加载了 {len(df)} 条数据")

    # 生成特征
    logger.info("生成特征...")
    feature_gen = QlibFeatureGenerator(normalize=True)
    features = feature_gen.generate(df)

    # 生成目标变量（未来 N 天收益率）
    logger.info(f"生成目标变量 (horizon={target_horizon})...")

    # 按股票计算未来收益
    if isinstance(df.index, pd.MultiIndex):
        # 多股票
        close_prices = df["close"].unstack("symbol")
        future_returns = close_prices.pct_change(target_horizon).shift(-target_horizon)
        target = future_returns.stack().reindex(features.index)
    else:
        # 单股票
        target = df["close"].pct_change(target_horizon).shift(-target_horizon)
        target = target.reindex(features.index)

    # 移除 NaN
    valid_mask = ~(features.isna().any(axis=1) | target.isna())
    features = features[valid_mask]
    target = target[valid_mask]

    logger.info(f"有效样本: {len(features)}")

    # 分割训练和验证集
    total_days = features.index.get_level_values(0).nunique() if isinstance(
        features.index, pd.MultiIndex
    ) else len(features.index.unique())

    train_end_idx = int(len(features) * (train_days / (train_days + valid_days)))

    X_train = features.iloc[:train_end_idx]
    X_valid = features.iloc[train_end_idx:]
    y_train = target.iloc[:train_end_idx]
    y_valid = target.iloc[train_end_idx:]

    logger.info(f"训练集: {len(X_train)} 样本, 验证集: {len(X_valid)} 样本")

    return X_train, X_valid, y_train, y_valid


def main():
    args = parse_args()

    # 如果提供了配置文件，加载并覆盖参数
    if args.config:
        config = load_config(args.config)
        
        # 处理嵌套配置映射
        # 1. data
        if "data" in config:
            data_conf = config["data"]
            if "symbols" in data_conf:
                args.symbols = ",".join(data_conf["symbols"])
            if "data_dir" in data_conf:
                args.data_dir = data_conf["data_dir"]
            if "train_days" in data_conf:
                args.train_days = data_conf["train_days"]
            if "valid_days" in data_conf:
                args.valid_days = data_conf["valid_days"]
            if "interval" in data_conf and args.interval is None:
                args.interval = data_conf["interval"]
                
        # 2. model
        if "model" in config:
            model_conf = config["model"]
            if "name" in model_conf:
                args.model_name = model_conf["name"]
            if "params" in model_conf:
                # 注意：params 目前在 argparse 里没有直接对应的字典参数，
                # 这里主要处理一些顶层暴露的参数
                params = model_conf["params"]
                if "learning_rate" in params:
                    args.learning_rate = params["learning_rate"]
                if "num_boost_round" in model_conf: # 注意这里是在 model 层级还是 params 层级
                    # qlib_ml_config 示例中 num_boost_round 在 model 层级
                   args.num_boost_round = model_conf.get("num_boost_round", args.num_boost_round)
            
            # 检查直接在 model 下的参数
            if "num_boost_round" in model_conf:
                args.num_boost_round = model_conf["num_boost_round"]
                
            if "target_horizon" in model_conf:
                 args.target_horizon = model_conf["target_horizon"]

        # 3. validation
        if "validation" in config:
             val_conf = config["validation"]
             # walk_forward 是个开关
             # 如果 config 里明确配置了 validation.enabled，则覆盖
             if "enabled" in val_conf and val_conf["enabled"]:
                 args.walk_forward = True

        # 保留原有的扁平映射作为 fallback
        for key, value in config.items():
            if hasattr(args, key.replace("-", "_")) and not isinstance(value, dict):
                setattr(args, key.replace("-", "_"), value)

    # 如果 interval 还是没有设置，使用默认值
    if args.interval is None:
        args.interval = "1min"

    # 解析参数
    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    logger.info("=" * 60)
    logger.info("ML 模型训练")
    logger.info("=" * 60)
    logger.info(f"股票列表: {symbols}")
    logger.info(f"训练天数: {args.train_days}")
    logger.info(f"验证天数: {args.valid_days}")
    logger.info(f"预测 horizon: {args.target_horizon}")
    logger.info(f"模型名称: {args.model_name}")
    logger.info(f"Walk-Forward 验证: {args.walk_forward}")
    logger.info(f"数据频率: {args.interval}")
    logger.info("=" * 60)

    try:
        # 准备数据
        X_train, X_valid, y_train, y_valid = prepare_data(
            symbols=symbols,
            data_dir=args.data_dir,
            train_days=args.train_days,
            valid_days=args.valid_days,
            target_horizon=args.target_horizon,
            interval=args.interval,
        )

        if args.walk_forward:
            # Walk-Forward 验证
            logger.info("\n开始 Walk-Forward 验证...")

            # 合并数据进行验证
            X_all = pd.concat([X_train, X_valid])
            y_all = pd.concat([y_train, y_valid])

            validator = WalkForwardValidator(
                trainer_class=LightGBMTrainer,
                train_window=252,  # 1 年
                test_window=21,  # 1 个月
                step_size=21,
                model_dir=args.models_dir,
                model_name=args.model_name,
                num_boost_round=args.num_boost_round,
                params={"learning_rate": args.learning_rate},
            )

            results = validator.validate(X_all, y_all)

            logger.info("\n" + "=" * 60)
            logger.info("Walk-Forward 验证结果")
            logger.info("=" * 60)
            logger.info(f"折数: {results['num_folds']}")
            logger.info(f"平均 IC: {results['mean_ic']:.4f}")
            logger.info(f"IC 标准差: {results['std_ic']:.4f}")
            logger.info(f"ICIR: {results['icir']:.4f}")
            logger.info(f"平均 MSE: {results['mean_mse']:.6f}")

        else:
            # 标准训练
            logger.info("\n开始模型训练...")

            trainer = LightGBMTrainer(
                model_dir=args.models_dir,
                model_name=args.model_name,
                num_boost_round=args.num_boost_round,
                params={"learning_rate": args.learning_rate},
            )

            trainer.train(X_train, y_train, X_valid, y_valid)

            # 评估
            train_metrics = trainer.evaluate(X_train, y_train)
            valid_metrics = trainer.evaluate(X_valid, y_valid)

            logger.info("\n" + "=" * 60)
            logger.info("训练结果")
            logger.info("=" * 60)
            logger.info(f"训练集 IC: {train_metrics['ic']:.4f}")
            logger.info(f"验证集 IC: {valid_metrics['ic']:.4f}")
            logger.info(f"验证集 MSE: {valid_metrics['mse']:.6f}")

            # 更新元数据
            trainer.metadata.update(
                {
                    "symbols": symbols,
                    "train_days": args.train_days,
                    "valid_days": args.valid_days,
                    "target_horizon": args.target_horizon,
                    "train_ic": train_metrics["ic"],
                    "valid_ic": valid_metrics["ic"],
                    "ic": valid_metrics["ic"],
                    "icir": valid_metrics["icir"],
                    "mse": valid_metrics["mse"],
                }
            )

            # 保存模型
            model_path = trainer.save()
            logger.info(f"\n模型已保存到: {model_path}")

            # 显示特征重要性
            logger.info("\nTop 10 特征重要性:")
            importance = trainer.get_feature_importance(10)
            for _, row in importance.iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.2f}")

            # 设置为当前模型
            if args.set_current:
                manager = ModelManager(args.models_dir)
                manager.set_current_model(model_path.name)
                logger.info(f"\n已设置为当前模型: {model_path.name}")

        logger.info("\n" + "=" * 60)
        logger.info("训练完成!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"训练失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
