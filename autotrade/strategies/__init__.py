from .momentum_strategy import MomentumStrategy
from .qlib_ml_strategy import QlibMLStrategy


__all__ = ["MomentumStrategy", "QlibMLStrategy"]

# 策略注册表 - 用于策略工厂
STRATEGY_REGISTRY = {
    "momentum": MomentumStrategy,
    "qlib_ml": QlibMLStrategy,
}


def get_strategy_class(strategy_type: str):
    """根据策略类型获取策略类"""
    if strategy_type not in STRATEGY_REGISTRY:
        raise ValueError(
            f"未知策略类型: {strategy_type}. 可用: {list(STRATEGY_REGISTRY.keys())}"
        )
    return STRATEGY_REGISTRY[strategy_type]
