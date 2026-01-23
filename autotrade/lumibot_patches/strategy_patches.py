"""
Strategy 补丁 - 禁用 Lumiwealth 云功能

Lumibot 在每次交易迭代后会尝试向 Lumiwealth 云服务器发送更新。
如果未设置 LUMIWEALTH_API_KEY，会打印警告；如果设置了无效的 key，会报 403 错误。

这个补丁通过 monkey patch 的方式禁用云功能，避免警告和错误日志。
"""


def patch_strategy_to_disable_cloud():
    """
    禁用 Strategy 的云功能。

    通过替换 Strategy.send_update_to_cloud 方法，直接返回 None，
    跳过所有与 Lumiwealth 云服务器的通信。

    使用方式：
        from autotrade.lumibot_patches.strategy_patches import patch_strategy_to_disable_cloud
        patch_strategy_to_disable_cloud()
    """
    from lumibot.strategies.strategy import Strategy

    # 保存原始方法（如果需要的话）
    if not hasattr(Strategy, "_original_send_update_to_cloud"):
        Strategy._original_send_update_to_cloud = Strategy.send_update_to_cloud

    # 替换为空实现
    Strategy.send_update_to_cloud = lambda self: None


def unpatch_strategy_to_disable_cloud():
    """
    恢复 Strategy 的云功能（撤销补丁）。

    使用方式：
        from autotrade.lumibot_patches.strategy_patches import unpatch_strategy_to_disable_cloud
        unpatch_strategy_to_disable_cloud()
    """
    from lumibot.strategies.strategy import Strategy

    if hasattr(Strategy, "_original_send_update_to_cloud"):
        Strategy.send_update_to_cloud = Strategy._original_send_update_to_cloud
        delattr(Strategy, "_original_send_update_to_cloud")