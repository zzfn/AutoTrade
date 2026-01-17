# Design: 项目目录基础架构

## Context

AutoTrade 是一个基于 LumiBot 的量化交易平台，需要建立清晰的目录结构来：

- 分离关注点（策略、配置、工具等）
- 支持多策略开发与管理
- **支持机器学习模型开发（Qlib + LightGBM）**
- **支持因子/特征工程**
- 便于测试和维护
- 遵循 Python 项目最佳实践

## Goals / Non-Goals

### Goals

- 建立模块化、可扩展的目录结构
- 采用简单直接的 flat layout
- 分离策略代码与核心代码
- 提供清晰的配置管理路径
- **支持 ML 模型和因子工程的目录组织**

### Non-Goals

- 不实现具体功能代码（仅占位）
- 不涉及 CI/CD 配置
- 不涉及文档生成

## Decisions

### 1. 使用 Flat Layout（私人项目最佳选择）

**决定**: 采用 `autotrade/` 直接放在根目录的 flat layout

**原因**:

- 简单直接，不需要额外安装步骤
- 直接 `uv run python main.py` 就能运行
- 更适合私人量化项目，不需要发布成包
- 减少不必要的复杂度

### 2. 模块组织

**决定**: 所有 Python 代码放在 `autotrade/` 内，包括回测代码

**原因**:

- 保持一致性，所有 Python 代码都在包内
- `autotrade/backtests/` 存放回测逻辑和策略配置代码
- 清晰的代码边界

### 3. 数据与输出分离

**决定**: 使用 `outputs/` 存放所有运行时生成的文件，`data/` 存放原始数据

**原因**:

- 清晰分离"代码"、"数据"和"产物"
- `outputs/` 包含日志、模型、回测结果，便于统一清理和忽略
- `configs/` 存放 YAML/JSON 配置文件，与代码分离

## 目录结构预览

```
AutoTrade/
├── autotrade/               # 核心代码库
│   ├── __init__.py
│   ├── core/                # 核心架构
│   ├── brokers/             # 交易接口
│   ├── strategies/          # 策略实现
│   ├── features/            # 因子/特征
│   ├── backtests/           # 回测引擎逻辑
│   ├── config/              # 代码内配置
│   └── utils/               # 工具函数
├── configs/                 # 外部配置文件
│   ├── strategies/          # 策略参数配置
│   └── backtests/           # 回测场景配置
├── notebooks/               # 研究与实验 Notebooks
├── data/                    # 数据存储
│   ├── raw/                 # 原始数据 (Qlib 等)
│   └── processed/           # 处理后数据
├── outputs/                 # 运行时产物 (Git Ignored)
│   ├── logs/                # 运行日志
│   ├── models/              # 训练好的模型
│   └── backtests/           # 回测结果图表/报告
├── scripts/                 # 实用脚本 (数据下载等)
├── tests/                   # 测试代码
└── main.py                  # 项目入口
```

## Risks / Trade-offs

| 风险                             | 缓解措施                       |
| -------------------------------- | ------------------------------ |
| Flat layout 可能导致意外导入问题 | 私人项目风险极低，正常开发即可 |
| 目录过多导致认知负担             | 保持扁平化，避免过深嵌套       |

## Open Questions

- `scripts/` 目录用于存放数据处理、维护等独立脚本（已添加）
