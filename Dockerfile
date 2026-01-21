FROM python:3.11-slim

ENV TZ=Asia/Shanghai
ENV UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PYTHONUNBUFFERED=1
# 关键：强制指定 Matplotlib 配置和缓存路径到固定位置
ENV MPLCONFIGDIR=/app/.matplotlib_cache

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 先安装依赖（利用缓存层）
COPY pyproject.toml uv.lock ./
RUN uv sync --no-install-project

# 复制项目
COPY . .

# 创建固定缓存目录并赋予权限
RUN mkdir -p $MPLCONFIGDIR && chmod -R 777 $MPLCONFIGDIR

# 重点：使用 uv run 执行预热，确保环境一致
# 并且通过环境变量告知 matplotlib 缓存位置
RUN uv run python -c "import matplotlib.pyplot; import matplotlib.font_manager; matplotlib.font_manager._get_fontmanager()"

RUN mkdir -p /app/logs /app/reports && chmod -R 777 /app/logs /app/reports

EXPOSE 8000

CMD ["uv", "run", "main.py"]