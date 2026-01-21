# AutoTrade Kubernetes Pod 快速部署指南

## 🚀 3 步快速部署

### 1️⃣ 构建 Docker 镜像

```bash
make docker-build
```

### 2️⃣ 配置环境变量（可选）

如果已有 `.env` 文件，脚本会自动加载环境变量。或者手动创建：

```bash
# 创建 .env 文件
cat > .env << EOF
ALPACA_API_KEY=your-key-here
ALPACA_API_SECRET=your-secret-here
ALPACA_PAPER=True
EOF
```

### 3️⃣ 运行 Pod

```bash
make k8s-run
```

就这么简单！Pod 会自动创建并启动。

## 📊 查看状态和日志

```bash
# 查看 Pod 状态
make k8s-status
# 或
kubectl get pod autotrade

# 查看日志
kubectl logs -f autotrade

# 查看详细信息
kubectl describe pod autotrade
```

## 🌐 访问应用

```bash
# 端口转发
kubectl port-forward autotrade 8000:8000

# 在浏览器访问
open http://localhost:8000
```

## 🗑️ 删除 Pod

```bash
make k8s-delete-pod
# 或
kubectl delete pod autotrade
```

## 🔧 自定义配置

### 自定义镜像名称

```bash
IMAGE_NAME=my-autotrade:v1 make k8s-run
```

### 自定义 Pod 名称

```bash
POD_NAME=my-trade-app make k8s-run
```

### 指定环境变量

```bash
kubectl run autotrade \
  --image=autotrade:latest \
  --image-pull-policy=Never \
  --restart=Never \
  --port=8000 \
  --env=ALPACA_API_KEY=your-key \
  --env=ALPACA_API_SECRET=your-secret \
  --env=ALPACA_PAPER=True
```

## 📝 常用命令速查

```bash
# 构建
make docker-build

# 运行
make k8s-run

# 查看状态
make k8s-status

# 查看日志
kubectl logs -f autotrade

# 端口转发
kubectl port-forward autotrade 8000:8000

# 删除
make k8s-delete-pod

# 进入 Pod
kubectl exec -it autotrade -- /bin/bash
```

## 🐛 故障排查

### Pod 无法启动

```bash
# 查看 Pod 状态
kubectl get pod autotrade

# 查看事件
kubectl describe pod autotrade

# 查看日志
kubectl logs autotrade
```

### 镜像拉取失败

确保先构建镜像：
```bash
make docker-build
docker images | grep autotrade
```

### 环境变量问题

```bash
# 查看 Pod 中的环境变量
kubectl exec autotrade -- env | grep ALPACA

# 重新运行（确保 .env 文件正确）
make k8s-delete-pod
make k8s-run
```

## 🎯 完整示例

```bash
# 1. 确保集群运行
kubectl cluster-info

# 2. 构建镜像
make docker-build

# 3. 运行 Pod
make k8s-run

# 4. 等待 Pod 就绪
kubectl wait --for=condition=Ready pod/autotrade --timeout=30s

# 5. 查看日志
kubectl logs -f autotrade

# 6. 访问应用（另一个终端）
kubectl port-forward autotrade 8000:8000

# 7. 完成后清理
make k8s-delete-pod
```

## ⚡ 与完整部署的区别

| 特性 | Pod (k8s-run) | 完整部署 (k8s-deploy) |
|------|--------------|---------------------|
| 复杂度 | ⭐ 简单 | ⭐⭐⭐ 复杂 |
| 资源 | 仅 Pod | Deployment + Service + Ingress |
| 自重启 | ❌ 否 | ✅ 是 |
| 负载均衡 | ❌ 否 | ✅ 是 |
| 适用场景 | 开发测试 | 生产环境 |

**开发推荐**：使用 `make k8s-run`（快速、简单）
**生产推荐**：使用 `make k8s-deploy`（完整、可靠）
