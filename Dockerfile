# 尝试使用完整版镜像（排查 slim 缺库问题）
FROM python:3.11-bookworm

# 设置时区为中国标准时间
ENV TZ=Asia/Shanghai
ENV UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 关键：禁用 Python 输出缓冲，确保容器日志实时显示
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# matplotlib 配置：使用 Agg 后端，避免 GUI 依赖
ENV MPLBACKEND=Agg

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install system dependencies
# libgomp1 is required for lightgbm
# fontconfig + fonts-dejavu-core 是 matplotlib 需要的基础字体
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    fontconfig \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -fv

# 复制 uv 二进制文件
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装依赖
COPY pyproject.toml uv.lock ./
RUN uv sync --no-install-project

# 复制项目代码并安装项目
COPY . .
RUN uv sync

# 修复 polars 二进制不兼容问题：
# polars 是 Rust 编译的，可能存在架构不匹配问题
# 强制在容器内重新安装，确保二进制匹配当前架构
RUN uv pip install --force-reinstall polars

# 治本：在构建时预热并验证所有依赖
# 如果任何导入失败，构建会失败（而不是运行时失败）
RUN uv run python -c "\
import sys; \
print('验证依赖...', flush=True); \
import matplotlib.font_manager; print('✓ matplotlib', flush=True); \
import pandas; print('✓ pandas', flush=True); \
import numpy; print('✓ numpy', flush=True); \
import polars; print('✓ polars', flush=True); \
from lumibot.strategies.strategy import Strategy; print('✓ lumibot.strategies', flush=True); \
from lumibot.brokers import Alpaca; print('✓ lumibot.brokers', flush=True); \
from lumibot.traders import Trader; print('✓ lumibot.traders', flush=True); \
print('所有依赖验证通过!', flush=True); \
"


EXPOSE 8000

# 设置默认命令
CMD ["uv", "run","main.py"]