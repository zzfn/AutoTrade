#!/bin/bash
# 删除 Pod 脚本

POD_NAME="${POD_NAME:-autotrade}"

echo "🗑️  删除 Pod: $POD_NAME"
kubectl delete pod "$POD_NAME" --ignore-not-found=true

echo "✅ 删除完成"
