#!/bin/bash
# Docker 镜像构建脚本

set -e

# 配置变量
IMAGE_NAME="${IMAGE_NAME:-autotrade}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
REGISTRY="${REGISTRY:-}"  # 如果使用远程仓库，设置为 registry.example.com

FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"
if [ -n "$REGISTRY" ]; then
    FULL_IMAGE_NAME="${REGISTRY}/${FULL_IMAGE_NAME}"
fi

echo "========================================"
echo "AutoTrade Docker 镜像构建"
echo "========================================"
echo "镜像名称: $FULL_IMAGE_NAME"
echo "----------------------------------------"

# 构建镜像
echo "📦 开始构建 Docker 镜像..."
docker build -t "$FULL_IMAGE_NAME" .

if [ $? -eq 0 ]; then
    echo "✅ 镜像构建成功！"
    echo "📝 镜像信息："
    docker images "$IMAGE_NAME" | tail -n +2

    # 如果设置了 registry，推送到远程仓库
    if [ -n "$REGISTRY" ]; then
        echo "----------------------------------------"
        echo "📤 推送镜像到 $REGISTRY..."
        docker push "$FULL_IMAGE_NAME"
        echo "✅ 镜像推送成功！"
    fi

    echo "========================================"
    echo "🚀 使用以下命令运行："
    echo "   docker run -p 8000:8000 --env-file .env $FULL_IMAGE_NAME"
    echo "========================================"
else
    echo "❌ 镜像构建失败！"
    exit 1
fi
