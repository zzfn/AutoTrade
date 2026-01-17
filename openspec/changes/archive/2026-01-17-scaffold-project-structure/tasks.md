# Tasks: 初始化项目目录基础架构

## 1. 核心包结构 (autotrade/)

- [x] 1.1 创建 `autotrade/` 目录并添加 `__init__.py`
- [x] 1.2 创建 `autotrade/core/` 目录（核心功能模块）并添加 `__init__.py`
- [x] 1.3 创建 `autotrade/brokers/` 目录（券商接口模块）并添加 `__init__.py`
- [x] 1.4 创建 `autotrade/backtests/` 目录（回测引擎逻辑）并添加 `__init__.py`

## 2. 策略模块 (autotrade/strategies/)

- [x] 2.1 创建 `autotrade/strategies/` 目录用于存放自定义策略
- [x] 2.2 添加 `autotrade/strategies/__init__.py`
- [x] 2.3 创建 `autotrade/strategies/examples/` 目录存放示例策略
- [x] 2.4 创建 `autotrade/strategies/ml/` 目录存放机器学习策略

## 3. 因子/特征工程模块 (autotrade/features/)

- [x] 3.1 创建 `autotrade/features/` 目录用于因子定义和特征工程
- [x] 3.2 添加 `autotrade/features/__init__.py`
- [x] 3.3 创建 `autotrade/features/alpha/` 目录存放 alpha 因子
- [x] 3.4 添加 `autotrade/features/alpha/__init__.py`

## 4. 配置模块 (autotrade/config/ & configs/)

- [x] 4.1 创建 `autotrade/config/` 目录（代码配置）
- [x] 4.2 添加 `autotrade/config/__init__.py` 占位
- [x] 4.3 创建 `configs/` 目录（外部配置）
- [x] 4.4 创建 `configs/strategies/` 和 `configs/backtests/` 目录

## 5. 工具模块 (autotrade/utils/)

- [x] 5.1 创建 `autotrade/utils/` 目录
- [x] 5.2 添加 `autotrade/utils/__init__.py` 占位
- [x] 5.3 添加 `autotrade/utils/logger.py` 占位（日志配置）
- [x] 5.4 添加 `autotrade/utils/data_loader.py` 占位（数据加载工具）

## 6. 实用脚本 (scripts/)

- [x] 6.1 创建 `scripts/` 目录用于存放独立脚本

## 7. 输出目录 (outputs/)

- [x] 7.1 创建 `outputs/` 目录（Git Ignored）
- [x] 7.2 创建 `outputs/logs/` 目录
- [x] 7.3 创建 `outputs/models/checkpoints/` 目录
- [x] 7.4 创建 `outputs/backtests/` 目录

## 8. 数据与研究 (data/ & notebooks/)

- [x] 8.1 创建 `data/raw/` 和 `data/processed/` 目录
- [x] 8.2 创建 `notebooks/` 目录用于实验
- [x] 8.3 更新 `.gitignore` 忽略 `outputs/` 内容但保留结构
- [x] 8.4 更新 `.gitignore` 忽略 `data/` 内容但保留结构

## 9. 测试目录 (tests/)

- [x] 9.1 创建 `tests/` 目录
- [x] 9.2 添加 `tests/__init__.py`
- [x] 9.3 添加 `tests/conftest.py` 占位
- [x] 9.4 创建 `tests/strategies/` 和 `tests/features/` 目录

## 10. 验证

- [x] 10.1 确认目录结构正确创建
- [x] 10.2 确认所有占位文件存在
- [x] 10.3 确认 `.gitignore` 配置正确忽略 outputs 和 data 内容
- [x] 10.4 确认使用 flat layout 且结构符合最佳实践
