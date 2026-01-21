#!/bin/bash
# 简单的 Pod 部署脚本

set -e

IMAGE_NAME="${IMAGE_NAME:-autotrade:latest}"
POD_NAME="${POD_NAME:-autotrade}"

echo "========================================"
echo "AutoTrade 快速部署 (仅 Pod)"
echo "========================================"
echo "镜像: $IMAGE_NAME"
echo "Pod: $POD_NAME"
echo "----------------------------------------"

# 检查 kubectl
if ! command -v kubectl &> /dev/null; then
    echo "❌ 错误: kubectl 未安装"
    exit 1
fi

# 删除已存在的 Pod
if kubectl get pod "$POD_NAME" &> /dev/null; then
    echo "🗑️  删除已存在的 Pod: $POD_NAME"
    kubectl delete pod "$POD_NAME" --ignore-not-found=true
fi

# 从 .env 文件读取环境变量（如果存在）
ENV_ARGS=""
if [ -f .env ]; then
    echo "📝 从 .env 文件加载环境变量..."
    while IFS='=' read -r key value; do
        # 跳过注释和空行
        [[ $key =~ ^#.*$ ]] && continue
        [[ -z $key ]] && continue
        # 跳过已经有值的变量
        if [ -n "$value" ]; then
            ENV_ARGS="$ENV_ARGS --env=$key=$value"
        fi
    done < .env
fi

echo "----------------------------------------"
echo "🚀 创建 Pod..."

kubectl run "$POD_NAME" \
    --image="$IMAGE_NAME" \
    --image-pull-policy=Never \
    --restart=Never \
    --port=8000 \
    $ENV_ARGS \
    --env=TZ=Asia/Shanghai \
    --env=HOST=0.0.0.0 \
    --env=PORT=8000 \
    --env=AUTOTRADE_ENV=production

echo "----------------------------------------"
echo "⏳ 等待 Pod 启动..."
sleep 3

kubectl wait --for=condition=Ready pod/"$POD_NAME" --timeout=30s || true

echo "========================================"
echo "✅ Pod 创建成功！"
echo "----------------------------------------"
echo "📊 查看 Pod 状态："
echo "   kubectl get pod $POD_NAME"
echo ""
echo "📋 查看日志："
echo "   kubectl logs -f $POD_NAME"
echo ""
echo "🌐 访问应用（端口转发）："
echo "   kubectl port-forward $POD_NAME 8000:8000"
echo "   然后访问 http://localhost:8000"
echo ""
echo "🗑️  删除 Pod："
echo "   kubectl delete pod $POD_NAME"
echo "========================================"

# 显示 Pod 状态
kubectl get pod "$POD_NAME"
