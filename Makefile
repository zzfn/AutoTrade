# AutoTrade Makefile
# ==================

.PHONY: help install dev run test lint format clean check all

# 默认目标：显示帮助信息
help:
	@echo "AutoTrade 开发命令"
	@echo "=================="
	@echo ""
	@echo "环境管理:"
	@echo "  make install     - 安装生产依赖"
	@echo "  make dev         - 安装开发依赖"
	@echo "  make sync        - 同步依赖（uv sync）"
	@echo ""
	@echo "运行:"
	@echo "  make run         - 运行主程序"
	@echo ""
	@echo "代码质量:"
	@echo "  make lint        - 运行 Ruff 检查"
	@echo "  make format      - 格式化代码（Ruff）"
	@echo "  make check       - 检查代码（lint + format 检查）"
	@echo ""
	@echo "测试:"
	@echo "  make test        - 运行测试"
	@echo ""
	@echo "清理:"
	@echo "  make clean       - 清理缓存文件"
	@echo ""
	@echo "组合命令:"
	@echo "  make all         - 格式化 + 检查 + 测试"

# ==================
# 环境管理
# ==================

# 安装生产依赖
install:
	uv sync --frozen

# 安装开发依赖
dev:
	uv sync --frozen --group dev

# 同步依赖
sync:
	uv sync

# ==================
# 运行
# ==================

# 运行帮助
run:
	uv run python main.py

# 运行回测
backtest:
	uv run python main.py backtest

# 运行模拟盘
paper:
	uv run python main.py paper

# 运行实盘（谨慎使用！）
live:
	@echo "⚠️  警告：即将启动实盘交易！"
	@read -p "确认继续？[y/N] " confirm && [ "$$confirm" = "y" ] && uv run python main.py live || echo "已取消"

# ==================
# 代码质量
# ==================

# Ruff 代码检查
lint:
	uv run ruff check .

# 格式化代码
format:
	uv run ruff format .
	uv run ruff check --fix .

# 检查代码（不修改）
check:
	uv run ruff format --check .
	uv run ruff check .

# ==================
# 测试
# ==================

# 运行测试
test:
	uv run pytest

# ==================
# 清理
# ==================

# 清理缓存文件
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true

# ==================
# 组合命令
# ==================

# 格式化 + 检查 + 测试
all: format lint test
