#!/bin/bash
# Kubernetes 删除部署脚本

set -e

NAMESPACE="${NAMESPACE:-default}"
K8S_DIR="$(dirname "$0")/../k8s"

echo "========================================"
echo "AutoTrade Kubernetes 清理"
echo "========================================"
echo "命名空间: $NAMESPACE"
echo "----------------------------------------"

# 检查 kubectl
if ! command -v kubectl &> /dev/null; then
    echo "❌ 错误: kubectl 未安装"
    exit 1
fi

# 删除资源
echo "🗑️  删除 Kubernetes 资源..."

kubectl delete -f "${K8S_DIR}/ingress.yaml" -n "$NAMESPACE" --ignore-not-found=true
kubectl delete -f "${K8S_DIR}/service.yaml" -n "$NAMESPACE" --ignore-not-found=true
kubectl delete -f "${K8S_DIR}/deployment.yaml" -n "$NAMESPACE" --ignore-not-found=true
kubectl delete -f "${K8S_DIR}/secret.yaml" -n "$NAMESPACE" --ignore-not-found=true
kubectl delete -f "${K8S_DIR}/configmap.yaml" -n "$NAMESPACE" --ignore-not-found=true

echo "----------------------------------------"
echo "✅ 清理完成！"
echo "========================================"
