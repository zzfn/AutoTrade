# Change: 初始化项目目录基础架构

## Why

AutoTrade 项目目前只有一个空的 `main.py` 文件，需要建立一套清晰、可扩展的目录结构来支持：

- 策略开发与管理
- **机器学习模型开发**（Qlib + LightGBM）
- **因子/特征工程**
- 配置管理
- 日志记录
- 工具函数复用

一个良好的项目结构是后续所有功能开发的基础，特别是需要支持传统规则策略和 ML 策略的混合开发。

## What Changes

- 创建 `src/autotrade/` 核心包目录结构
- 创建 `strategies/` 策略存放目录
- 创建 `models/` ML 模型存放目录（支持 LightGBM 等）
- 创建 `features/` 因子/特征工程目录（支持 Qlib 因子）
- 创建 `config/` 配置管理模块
- 创建 `utils/` 工具函数模块
- 创建 `logs/` 日志输出目录
- 创建 `data/` 数据缓存目录（支持 Qlib 数据格式）
- 创建 `backtests/` 回测配置和结果目录
- 创建 `tests/` 测试目录
- 添加必要的占位文件 (`__init__.py`, `.gitkeep`)

## Impact

- Affected specs: project-infrastructure (新增)
- Affected code:
  - 新增 `src/autotrade/` 包结构
  - 新增 `strategies/` 目录
  - 新增 `config/` 模块
  - 新增 `utils/` 模块
  - 新增 `tests/` 目录
  - 新增 `logs/` 和 `data/` 目录
