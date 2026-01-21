# AutoTrade Kubernetes 部署指南

本文档介绍如何在本地 Kubernetes 环境中部署 AutoTrade。

## 📋 前置要求

1. **Docker** - 用于构建镜像
   ```bash
   docker --version
   ```

2. **kubectl** - Kubernetes 命令行工具
   ```bash
   kubectl version --client
   ```

3. **本地 Kubernetes 集群** - 选择以下之一：
   - **Docker Desktop** (推荐) - 内置 Kubernetes
   - **Minikube** - 轻量级本地集群
   - **Kind** - Docker 中的 Kubernetes
   - **MicroK8s** - 轻量级 Kubernetes

## 🚀 快速开始

### 1️⃣ 构建镜像

```bash
# 使用脚本构建（推荐）
./scripts/docker-build.sh

# 或手动构建
docker build -t autotrade:latest .
```

### 2️⃣ 配置环境变量

在部署前，需要编辑 `k8s/secret.yaml`，填入你的 Alpaca API keys：

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: autotrade-secret
type: Opaque
stringData:
  ALPACA_API_KEY: "your-actual-api-key"
  ALPACA_API_SECRET: "your-actual-api-secret"
```

### 3️⃣ 部署到 Kubernetes

```bash
# 使用脚本部署（推荐）
./scripts/k8s-deploy.sh

# 或手动部署
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

### 4️⃣ 验证部署

```bash
# 查看 Pod 状态
kubectl get pods -l app=autotrade

# 查看日志
kubectl logs -f deployment/autotrade

# 端口转发访问应用
kubectl port-forward svc/autotrade 8000:8000
```

然后在浏览器访问 `http://localhost:8000`

## 🔧 配置说明

### ConfigMap (`k8s/configmap.yaml`)

存储非敏感的环境变量：

```yaml
data:
  TZ: "Asia/Shanghai"          # 时区
  HOST: "0.0.0.0"              # 监听地址
  PORT: "8000"                 # 监听端口
  ALPACA_PAPER: "True"         # 是否使用纸面交易
  AUTOTRADE_ENV: "production"  # 环境标识
```

### Secret (`k8s/secret.yaml`)

存储敏感信息（API keys）：

```yaml
stringData:
  ALPACA_API_KEY: "your-key"
  ALPACA_API_SECRET: "your-secret"
```

⚠️ **重要**：
- 不要将包含真实 API keys 的 `secret.yaml` 提交到 Git
- 生产环境建议使用外部密钥管理系统（如 Kubernetes Secrets、AWS Secrets Manager 等）

### Deployment (`k8s/deployment.yaml`)

定义应用的部署配置：

- **replicas**: 副本数（当前为 1）
- **resources**: 资源限制
  - requests: 512Mi 内存, 250m CPU
  - limits: 2Gi 内存, 1000m CPU
- **volumes**: 持久化存储
  - logs: 日志目录
  - reports: 报告目录

### Service (`k8s/service.yaml`)

暴露应用服务：

- **type**: ClusterIP（集群内部访问）
- 如需外部访问，可改为 LoadBalancer 或 NodePort

### Ingress (`k8s/ingress.yaml`)

外部访问配置（需要 Ingress Controller）：

- 使用 NGINX Ingress Controller
- 访问地址: `http://autotrade.local`（需配置本地 hosts）

## 📊 常用命令

### 查看资源状态

```bash
# 查看 Pod
kubectl get pods -l app=autotrade

# 查看 Service
kubectl get svc autotrade

# 查看 Deployment
kubectl get deployment autotrade

# 查看所有资源
kubectl get all -l app=autotrade
```

### 查看日志

```bash
# 查看实时日志
kubectl logs -f deployment/autotrade

# 查看最近 100 行日志
kubectl logs --tail=100 deployment/autotrade

# 查看特定 Pod 的日志
kubectl logs -f <pod-name>
```

### 调试

```bash
# 进入 Pod 容器
kubectl exec -it <pod-name> -- /bin/bash

# 端口转发
kubectl port-forward svc/autotrade 8000:8000

# 查看 Pod 详情
kubectl describe pod <pod-name>

# 查看 Deployment 事件
kubectl describe deployment autotrade
```

### 更新部署

```bash
# 更新镜像
kubectl set image deployment/autotrade autotrade=autotrade:v2

# 重启 Deployment
kubectl rollout restart deployment/autotrade

# 查看滚动更新状态
kubectl rollout status deployment/autotrade

# 回滚到上一版本
kubectl rollout undo deployment/autotrade

# 回滚到指定版本
kubectl rollout undo deployment/autotrade --to-revision=2
```

### 清理资源

```bash
# 使用脚本清理
./scripts/k8s-delete.sh

# 或手动删除
kubectl delete -f k8s/ingress.yaml
kubectl delete -f k8s/service.yaml
kubectl delete -f k8s/deployment.yaml
kubectl delete -f k8s/secret.yaml
kubectl delete -f k8s/configmap.yaml
```

## 🌐 本地访问方式

### 方式 1: Port Forward（推荐用于开发）

```bash
kubectl port-forward svc/autotrade 8000:8000
# 访问 http://localhost:8000
```

### 方式 2: NodePort

修改 `k8s/service.yaml`，将 type 改为 NodePort：

```yaml
spec:
  type: NodePort
  ports:
  - port: 8000
    targetPort: 8000
    nodePort: 30080  # 30000-32767
```

然后访问 `http://localhost:30080`

### 方式 3: LoadBalancer (Docker Desktop)

Docker Desktop 支持 LoadBalancer，可直接访问：

```yaml
spec:
  type: LoadBalancer
```

### 方式 4: Ingress

需要安装 Ingress Controller（如 NGINX）：

```bash
# Docker Desktop 已内置 NGINX Ingress
kubectl apply -f k8s/ingress.yaml

# 添加本地 hosts
echo "127.0.0.1 autotrade.local" | sudo tee -a /etc/hosts
# 访问 http://autotrade.local
```

## 🐳 本地 Kubernetes 集群选项

### Docker Desktop (推荐)

**优点**：
- 内置 Kubernetes，开箱即用
- 支持 LoadBalancer
- 图形化管理界面

**启动**：
1. 打开 Docker Desktop
2. 进入 Settings → Kubernetes
3. 启用 Kubernetes
4. 点击 "Apply & Restart"

### Minikube

**安装**：
```bash
brew install minikube  # macOS
minikube start
```

**访问**：
```bash
minikube service autotrade
```

### Kind

**安装**：
```bash
brew install kind  # macOS
kind create cluster
```

### MicroK8s

**安装**：
```bash
brew install microk8s  # macOS
microk8s start
microk8s enable dns ingress registry
```

## 🔍 故障排查

### Pod 无法启动

```bash
# 查看 Pod 状态
kubectl get pods -l app=autotrade

# 查看 Pod 详情
kubectl describe pod <pod-name>

# 查看日志
kubectl logs <pod-name>
```

常见问题：
- **ImagePullBackOff**: 镜像不存在，需要先构建
- **CrashLoopBackOff**: 应用启动失败，查看日志排查
- **OOMKilled**: 内存不足，增加 deployment.yaml 中的内存限制

### 环境变量未生效

```bash
# 检查 Secret 和 ConfigMap
kubectl get configmap autotrade-config -o yaml
kubectl get secret autotrade-secret -o yaml

# 检查 Pod 中的环境变量
kubectl exec <pod-name> -- env | grep ALPACA
```

### 网络无法访问

```bash
# 检查 Service
kubectl get svc autotrade

# 检查 Endpoints
kubectl get endpoints autotrade

# 测试 Service 连通性
kubectl run test --image=busybox --rm -it -- wget -O- http://autotrade:8000
```

## 📈 监控和日志

### 查看资源使用

```bash
kubectl top pods -l app=autotrade
kubectl top nodes
```

### 持久化存储

日志和报告文件存储在宿主机：
- 日志: `/tmp/autotrade/logs`
- 报告: `/tmp/autotrade/reports`

可在 `k8s/deployment.yaml` 中修改存储路径。

## 🔐 安全建议

1. **不要将 Secret 提交到 Git**：
   ```bash
   echo "k8s/secret.yaml" >> .gitignore
   ```

2. **使用外部密钥管理**（生产环境）：
   - Kubernetes External Secrets Operator
   - HashiCorp Vault
   - AWS Secrets Manager
   - Azure Key Vault

3. **启用 RBAC**：
   ```yaml
   apiVersion: rbac.authorization.k8s.io/v1
   kind: Role
   metadata:
     name: autotrade-role
   rules:
   - apiGroups: [""]
     resources: ["configmaps", "secrets"]
     verbs: ["get", "list"]
   ```

4. **网络策略**：
   ```yaml
   apiVersion: networking.k8s.io/v1
   kind: NetworkPolicy
   metadata:
     name: autotrade-network-policy
   spec:
     podSelector:
       matchLabels:
         app: autotrade
     policyTypes:
     - Ingress
     - Egress
   ```

## 📚 参考资源

- [Kubernetes 官方文档](https://kubernetes.io/docs/)
- [kubectl 命令参考](https://kubernetes.io/docs/reference/kubectl/)
- [Docker Desktop Kubernetes](https://docs.docker.com/desktop/kubernetes/)
- [Minikube 文档](https://minikube.sigs.k8s.io/docs/)

## 🆘 获取帮助

遇到问题？
1. 查看 Pod 日志：`kubectl logs -f deployment/autotrade`
2. 检查集群状态：`kubectl cluster-info`
3. 查看事件：`kubectl get events`
