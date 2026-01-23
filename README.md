# AutoTrade

基于 LumiBot 的量化交易系统，使用机器学习驱动的策略进行自动交易。

## 功能特性

- 🚀 **实时交易** - 通过 Alpaca API 进行实盘/模拟盘交易
- 📊 **回测系统** - 使用历史数据验证策略表现
- 🤖 **ML 策略** - 基于 Qlib/LightGBM 的机器学习选股策略
- 🌐 **Web 界面** - 实时监控仪表盘和模型管理
- 🔄 **滚动训练** - 支持模型在线更新

## 快速开始

### 1. 安装依赖

```bash
# 安装 Python 依赖
uv sync
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入 Alpaca API 密钥
```

### 3. 启动服务

```bash
# 开发模式
make dev

# 或直接运行
uv run uvicorn autotrade.web_server:app --reload
```

访问 http://localhost:8000 查看仪表盘。

## Qlib ML 策略使用指南

### 架构概述

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (Web UI)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Strategy    │  │ Model       │  │ Rolling Update      │  │
│  │ Config      │  │ Management  │  │ Trigger             │  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     TradeManager                             │
│                    QlibMLStrategy                            │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│   Data Adapter  ──►  Feature Generator  ──►  ML Model       │
└─────────────────────────────────────────────────────────────┘
```

### 1. 初始化数据

首次使用需要获取历史数据并转换为 Qlib 格式。

**通过 Web 界面**：

1. 访问 http://localhost:8000/data
2. 点击「启动数据同步」按钮
3. 配置股票池和时间范围
4. 等待数据下载完成

**通过 API**：

```bash
curl -X POST http://localhost:8000/api/data/sync \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["SPY", "AAPL", "MSFT"], "days": 730}'
```

### 2. 训练模型

**通过 Web 界面**：

1. 访问 http://localhost:8000/models
2. 点击「开始训练」按钮
3. 配置训练参数（股票、天数、预测周期等）
4. 等待训练完成，新模型会自动保存

**通过 API**：

```bash
curl -X POST http://localhost:8000/api/models/train \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["SPY", "AAPL", "MSFT"], "train_days": 504}'
```

### 3. 通过 Web 界面管理

1. 访问 http://localhost:8000/models
2. 配置 ML 策略参数（Top-K、再平衡周期）
3. 选择要使用的模型
4. 点击「启动策略」开始交易
5. 可以点击「Rolling 更新」按钮更新模型

### 4. API 端点

| 端点                       | 方法 | 说明               |
| -------------------------- | ---- | ------------------ |
| `/api/strategy/config`     | GET  | 获取当前策略配置   |
| `/api/strategy/ml_config`  | POST | 设置 ML 策略参数   |
| `/api/models`              | GET  | 列出所有可用模型   |
| `/api/models/current`      | GET  | 获取当前选择的模型 |
| `/api/models/select`       | POST | 选择要使用的模型   |
| `/api/models/train`        | POST | 启动模型训练       |
| `/api/models/train/status` | GET  | 获取训练状态       |
| `/api/data/sync`           | POST | 启动数据同步       |
| `/api/data/sync/status`    | GET  | 获取同步状态       |

## 策略说明

### QlibMLStrategy（ML 策略）

基于机器学习模型预测的策略：

- 使用 LightGBM 预测未来收益率
- Top-K 排名选股（选择预测分数最高的 K 只股票）
- 定期再平衡
- 支持前端配置和模型热切换

#### Walk-Forward 验证

模型训练默认启用 **Walk-Forward 验证**（滚动窗口验证），这是一种更符合实盘交易场景的验证方法：

- ✅ **模拟实盘**：每次只用历史数据预测未来，符合真实交易场景
- ✅ **多周期测试**：在多个时间窗口验证，避免偶然性
- ✅ **稳健性评估**：提供指标的均值和标准差（如 `IC: 0.041 ± 0.008`）

**固定参数配置**（单位：根K线）：

| 参数              | 值   | 说明         |
| ----------------- | ---- | ------------ |
| train_window      | 2000 | 训练窗口大小 |
| test_window       | 200  | 测试窗口大小 |
| step_size         | 200  | 滚动步长     |
| MIN_BARS_REQUIRED | 2500 | 最小数据要求 |

**数据量建议**：

- **最小要求**：2500 根K线（至少进行 2 个窗口的验证）
- **推荐数据量**：20000 根K线（约 90 个验证窗口，更稳健的评估）
- **数据不足时**：自动降级到单次训练（80/20 分割），界面会显示警告

**验证结果展示**：

模型训练完成后，界面会显示聚合指标：

- `IC: mean ± std`（信息系数均值和标准差）
- `ICIR: mean ± std`（信息系数比率）
- 验证窗口数和失败窗口数

模型列表中会显示 `WF✓` 标识，表示该模型通过了 Walk-Forward 验证。

#### 特征工程

采用类似 Qlib Alpha158 的技术指标因子：

- 价格回报率（1/5/10/20 天）
- 移动平均线及斜率
- 波动率（ATR、标准差）
- 成交量因子
- RSI、MACD、布林带等技术指标

## 项目结构

```
autotrade/
├── execution/            # 交易执行
│   └── strategies/       # 交易策略
│       └── qlib_ml_strategy.py
├── research/             # 研究模块
│   ├── data/            # 数据适配
│   ├── features/        # 特征工程
│   └── models/          # 模型训练
├── ui/                   # Web 界面
│   └── templates/
├── web/                  # Web 服务器
│   └── server.py
├── trade_manager.py      # 交易管理器
├── config.yaml           # ML 策略配置
├── models/               # 训练好的模型
└── datasets/             # Qlib 格式数据
```

## 开发

```bash
# 运行测试
uv run pytest

# 代码格式化
uv run ruff format .
uv run ruff check . --fix
```

## 许可证

MIT
