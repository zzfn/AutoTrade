"""
Lumibot Patches - Lumibot 回测系统修复补丁

这个模块包含了对 Lumibot 回测系统的修复补丁。
"""
from autotrade.lumibot_patches.alpaca_patches import (
    FixedAlpacaData,
    MyAlpacaBacktesting,
    patch_alpaca_timeframe_mapping,
)
from autotrade.lumibot_patches.strategy_patches import (
    patch_strategy_to_disable_cloud,
    unpatch_strategy_to_disable_cloud,
)

__all__ = [
    "FixedAlpacaData",
    "MyAlpacaBacktesting",
    "patch_alpaca_timeframe_mapping",
    "patch_strategy_to_disable_cloud",
    "unpatch_strategy_to_disable_cloud",
]
