# AutoTrade Docker & Kubernetes 快速开始

## 🐳 Docker 快速开始

### 1. 构建 Docker 镜像

```bash
# 使用 Makefile（推荐）
make docker-build

# 或直接使用脚本
./scripts/docker-build.sh

# 或手动构建
docker build -t autotrade:latest .
```

### 2. 运行 Docker 容器

```bash
# 使用 Makefile（推荐）
make docker-run

# 或手动运行
docker run -it --rm \
  -p 8000:8000 \
  --env-file .env \
  -v $(PWD)/logs:/app/logs \
  -v $(PWD)/reports:/app/reports \
  autotrade:latest
```

然后在浏览器访问 `http://localhost:8000`

## ☸️ Kubernetes 快速开始

### 前置条件

确保本地 Kubernetes 集群正在运行：

**Docker Desktop**（推荐）：
```bash
# 打开 Docker Desktop
# Settings → Kubernetes → Enable Kubernetes → Apply
kubectl cluster-info
```

**Minikube**：
```bash
minikube start
kubectl cluster-info
```

### 1. 构建 Docker 镜像

```bash
make docker-build
```

### 2. 配置环境变量

编辑 `k8s/secret.yaml`，填入你的 Alpaca API keys：

```bash
vi k8s/secret.yaml
```

```yaml
stringData:
  ALPACA_API_KEY: "your-actual-api-key"
  ALPACA_API_SECRET: "your-actual-api-secret"
```

### 3. 部署到 Kubernetes

```bash
# 使用 Makefile（推荐）
make k8s-deploy

# 或直接使用脚本
./scripts/k8s-deploy.sh
```

### 4. 验证部署

```bash
# 查看资源状态
make k8s-status

# 查看日志
make k8s-logs

# 或手动查看
kubectl get pods -l app=autotrade
kubectl logs -f deployment/autotrade
```

### 5. 访问应用

**方式 1: Port Forward（推荐）**
```bash
kubectl port-forward svc/autotrade 8000:8000
# 访问 http://localhost:8000
```

**方式 2: LoadBalancer (Docker Desktop)**
```bash
# 修改 k8s/service.yaml
# type: LoadBalancer
kubectl apply -f k8s/service.yaml

# 获取访问地址
kubectl get svc autotrade
```

### 6. 清理资源

```bash
make k8s-delete
```

## 📝 常用命令

### Docker

```bash
# 构建镜像
make docker-build

# 运行容器
make docker-run

# 查看运行中的容器
docker ps

# 查看日志
docker logs -f <container-id>

# 进入容器
docker exec -it <container-id> /bin/bash
```

### Kubernetes

```bash
# 部署
make k8s-deploy

# 删除
make k8s-delete

# 查看状态
make k8s-status

# 查看日志
make k8s-logs

# 端口转发
kubectl port-forward svc/autotrade 8000:8000

# 查看 Pod
kubectl get pods -l app=autotrade

# 进入 Pod
kubectl exec -it <pod-name> -- /bin/bash

# 重启部署
kubectl rollout restart deployment/autotrade
```

## 🔧 配置说明

### 环境变量

主要环境变量在 `k8s/configmap.yaml` 和 `k8s/secret.yaml` 中配置：

**ConfigMap**（非敏感）：
- `TZ`: 时区（默认：Asia/Shanghai）
- `HOST`: 监听地址（默认：0.0.0.0）
- `PORT`: 监听端口（默认：8000）
- `ALPACA_PAPER`: 是否纸面交易（默认：True）
- `AUTOTRADE_ENV`: 环境标识（默认：production）

**Secret**（敏感）：
- `ALPACA_API_KEY`: Alpaca API Key
- `ALPACA_API_SECRET`: Alpaca API Secret

### 资源限制

在 `k8s/deployment.yaml` 中配置：

```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

根据实际需求调整。

## 🐛 故障排查

### Docker

**镜像构建失败**：
```bash
# 清理 Docker 缓存
docker system prune -a

# 重新构建
docker build --no-cache -t autotrade:latest .
```

**容器无法启动**：
```bash
# 查看日志
docker logs autotrade

# 检查环境变量
docker exec autotrade env | grep ALPACA
```

### Kubernetes

**Pod 无法启动**：
```bash
# 查看 Pod 状态
kubectl get pods -l app=autotrade

# 查看详情
kubectl describe pod <pod-name>

# 查看日志
kubectl logs <pod-name>
```

**镜像拉取失败**：
```bash
# 确保镜像已构建
docker images | grep autotrade

# 本地集群使用 imagePullPolicy: Never
# 已在 k8s/deployment.yaml 中配置
```

**环境变量未生效**：
```bash
# 检查 ConfigMap 和 Secret
kubectl get configmap autotrade-config -o yaml
kubectl get secret autotrade-secret -o yaml

# 重新应用配置
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml

# 重启 Pod
kubectl rollout restart deployment/autotrade
```

## 📚 详细文档

- [Kubernetes 完整部署指南](./KUBERNETES.md)
- [Docker 官方文档](https://docs.docker.com/)
- [Kubernetes 官方文档](https://kubernetes.io/docs/)

## 🆘 获取帮助

遇到问题？
1. 查看日志：`make k8s-logs` 或 `kubectl logs -f deployment/autotrade`
2. 检查配置：`kubectl get configmap,secret -l app=autotrade`
3. 查看事件：`kubectl get events`
4. 查看 [完整文档](./KUBERNETES.md)
