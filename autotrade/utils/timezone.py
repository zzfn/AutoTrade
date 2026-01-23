"""
时区工具模块

提供美东时间（Eastern Time）格式化和转换功能
"""

from datetime import datetime, timezone
from typing import Optional


def format_et_time(timestamp: Optional[datetime]) -> str:
    """
    将时间戳格式化为美东时间字符串

    Args:
        timestamp: datetime 对象，如果为 None 返回 "-"

    Returns:
        格式化的美东时间字符串，格式：YYYY-MM-DD HH:MM:SS ET/EDT
        如果输入无效，返回 "-"

    Examples:
        >>> from datetime import datetime
        >>> dt = datetime(2025, 1, 15, 14, 30, 0)
        >>> format_et_time(dt)
        '2025-01-15 09:30:00 ET'
    """
    if timestamp is None:
        return "-"

    try:
        # 如果是 naive datetime，假设是 UTC
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        # 转换为美东时间
        import pytz
        et_tz = pytz.timezone('America/New_York')
        et_time = timestamp.astimezone(et_tz)

        # 判断是 EDT 还是 EST
        # EDT: UTC-4 (3月第二个周日 - 11月第一个周日)
        # EST: UTC-5 (其他时间)
        tz_name = "ET"  # 默认使用通用的 Eastern Time

        # 格式化时间
        time_str = et_time.strftime("%Y-%m-%d %H:%M:%S")

        return f"{time_str} {tz_name}"
    except Exception as e:
        # 如果转换失败，返回原始格式
        return timestamp.isoformat() if timestamp else "-"


def format_et_time_short(timestamp: Optional[datetime]) -> str:
    """
    将时间戳格式化为美东时间字符串（仅时分秒）

    Args:
        timestamp: datetime 对象，如果为 None 返回 "-"

    Returns:
        格式化的美东时间字符串，格式：HH:MM:SS ET/EDT
        如果输入无效，返回 "-"
    """
    if timestamp is None:
        return "-"

    try:
        # 如果是 naive datetime，假设是 UTC
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        # 转换为美东时间
        import pytz
        et_tz = pytz.timezone('America/New_York')
        et_time = timestamp.astimezone(et_tz)

        time_str = et_time.strftime("%H:%M:%S")

        return f"{time_str} ET"
    except Exception as e:
        return "-"


def get_et_now() -> datetime:
    """
    获取当前美东时间的 datetime 对象

    Returns:
        当前美东时间的 datetime 对象（带时区信息）
    """
    import pytz
    et_tz = pytz.timezone('America/New_York')
    return datetime.now(et_tz)


def utc_to_et(timestamp: datetime) -> datetime:
    """
    将 UTC 时间转换为美东时间

    Args:
        timestamp: UTC 时间（可以是 naive 或 aware）

    Returns:
        美东时间的 datetime 对象（带时区信息）
    """
    import pytz
    et_tz = pytz.timezone('America/New_York')

    # 如果是 naive datetime，假设是 UTC
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    return timestamp.astimezone(et_tz)
