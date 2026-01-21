#!/bin/bash
# Kubernetes 部署脚本

set -e

# 配置变量
NAMESPACE="${NAMESPACE:-default}"
IMAGE_NAME="${IMAGE_NAME:-autotrade:latest}"
K8S_DIR="$(dirname "$0")/../k8s"

echo "========================================"
echo "AutoTrade Kubernetes 部署"
echo "========================================"
echo "命名空间: $NAMESPACE"
echo "镜像: $IMAGE_NAME"
echo "----------------------------------------"

# 检查 kubectl
if ! command -v kubectl &> /dev/null; then
    echo "❌ 错误: kubectl 未安装"
    echo "请安装 kubectl: https://kubernetes.io/docs/tasks/tools/"
    exit 1
fi

# 检查集群连接
echo "🔍 检查 Kubernetes 集群连接..."
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ 错误: 无法连接到 Kubernetes 集群"
    echo "请确保集群正在运行且 kubeconfig 已正确配置"
    exit 1
fi
echo "✅ 集群连接正常"

# 更新 deployment 中的镜像
echo "----------------------------------------"
echo "📝 更新 deployment.yaml 中的镜像..."
sed "s|image: autotrade:latest|image: ${IMAGE_NAME}|g" \
    "${K8S_DIR}/deployment.yaml" > "${K8S_DIR}/deployment.yaml.tmp" \
    && mv "${K8S_DIR}/deployment.yaml.tmp" "${K8S_DIR}/deployment.yaml"

# 应用配置
echo "----------------------------------------"
echo "🚀 部署到 Kubernetes..."

echo "1️⃣  应用 ConfigMap..."
kubectl apply -f "${K8S_DIR}/configmap.yaml" -n "$NAMESPACE"

echo "2️⃣  应用 Secret..."
kubectl apply -f "${K8S_DIR}/secret.yaml" -n "$NAMESPACE"

echo "3️⃣  应用 Deployment..."
kubectl apply -f "${K8S_DIR}/deployment.yaml" -n "$NAMESPACE"

echo "4️⃣  应用 Service..."
kubectl apply -f "${K8S_DIR}/service.yaml" -n "$NAMESPACE"

echo "5️⃣  应用 Ingress..."
kubectl apply -f "${K8S_DIR}/ingress.yaml" -n "$NAMESPACE"

echo "----------------------------------------"
echo "⏳ 等待部署就绪..."
kubectl rollout status deployment/autotrade -n "$NAMESPACE" --timeout=60s

echo "========================================"
echo "✅ 部署成功！"
echo "----------------------------------------"
echo "📊 查看 Pod 状态："
echo "   kubectl get pods -n $NAMESPACE"
echo ""
echo "📋 查看日志："
echo "   kubectl logs -f deployment/autotrade -n $NAMESPACE"
echo ""
echo "🌐 访问应用："
echo "   kubectl port-forward svc/autotrade 8000:8000 -n $NAMESPACE"
echo "   然后访问 http://localhost:8000"
echo "========================================"

# 显示 Pod 状态
echo "当前 Pod 状态："
kubectl get pods -n "$NAMESPACE" -l app=autotrade
