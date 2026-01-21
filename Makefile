# AutoTrade Makefile
# ==================

.PHONY: help install dev run test lint format clean check all docker-build docker-run k8s-deploy k8s-delete k8s-logs k8s-status k8s-run k8s-delete-pod

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
	@echo "Docker:"
	@echo "  make docker-build    - 构建 Docker 镜像"
	@echo "  make docker-run      - 运行 Docker 容器"
	@echo ""
	@echo "Kubernetes:"
	@echo "  make k8s-run         - 快速运行 Pod (推荐)"
	@echo "  make k8s-delete-pod  - 删除 Pod"
	@echo "  make k8s-deploy      - 完整部署 (Deployment + Service)"
	@echo "  make k8s-delete      - 删除完整部署"
	@echo "  make k8s-logs        - 查看日志"
	@echo "  make k8s-status      - 查看状态"
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

# ==================
# Docker
# ==================

# 构建 Docker 镜像
docker-build:
	@echo "📦 构建 Docker 镜像..."
	./scripts/docker-build.sh

# 运行 Docker 容器
docker-run:
	@echo "🚀 运行 Docker 容器..."
	docker run -it --rm \
		-p 8000:8000 \
		--env-file .env \
		-v $(PWD)/logs:/app/logs \
		-v $(PWD)/reports:/app/reports \
		autotrade:latest

# ==================
# Kubernetes
# ==================

# 快速运行 Pod（仅 Pod，最简单）
k8s-run:
	@echo "🚀 快速部署 Pod..."
	./scripts/k8s-run.sh

# 删除 Pod
k8s-delete-pod:
	@echo "🗑️  删除 Pod..."
	./scripts/k8s-delete-pod.sh

# 完整部署（Deployment + Service）
k8s-deploy:
	@echo "🚀 部署到 Kubernetes..."
	./scripts/k8s-deploy.sh

# 删除 Kubernetes 部署
k8s-delete:
	@echo "🗑️  删除 Kubernetes 部署..."
	./scripts/k8s-delete.sh

# 查看 Kubernetes 日志
k8s-logs:
	kubectl logs -f deployment/autotrade

# 查看 Kubernetes 状态
k8s-status:
	@echo "📊 Kubernetes 资源状态:"
	@echo ""
	kubectl get pods

# 查看特定 Pod 的状态
k8s-pod-status:
	kubectl get pod autotrade -o wide
