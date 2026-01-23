# AutoTrade

基于 LumiBot 的量化交易系统，使用机器学习驱动的策略进行自动交易。

## 功能特性

- **实时交易** - 通过 Alpaca API 进行实盘/模拟盘交易
- **回测系统** - 异步回测任务管理，支持多种时间间隔
- **ML 策略** - 基于 LightGBM 的机器学习选股策略
- **Walk-Forward 验证** - 滚动窗口验证，更稳健的模型评估
- **Web 界面** - 实时监控仪表盘和模型管理

## 快速开始

### 1. 安装依赖

```bash
# 安装生产依赖
make install

# 或安装开发依赖
make dev
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入 Alpaca API 密钥
```

### 3. 启动服务

```bash
# 启动 Web 服务器
make run

# 或直接运行
uv run python main.py
```

访问 http://localhost:8000 查看仪表盘。

## 运行模式

```bash
# 回测模式
make backtest

# 模拟盘交易
make paper

# 实盘交易（谨慎使用！）
make live
```

## Qlib ML 策略使用指南

### 架构概述

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (Web UI)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Strategy    │  │ Model       │  │ Backtest Tasks      │  │
│  │ Config      │  │ Management  │  │ Management          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     Web Server (FastAPI)                     │
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

### 3. Walk-Forward 验证

模型训练默认启用 **Walk-Forward 验证**（滚动窗口验证），这是一种更符合实盘交易场景的验证方法：

- **模拟实盘**：每次只用历史数据预测未来
- **多周期测试**：在多个时间窗口验证，避免偶然性
- **稳健性评估**：提供指标的均值和标准差（如 `IC: 0.041 ± 0.008`）

**固定参数配置**（单位：根K线）：

| 参数              | 值   | 说明         |
| ----------------- | ---- | ------------ |
| train_window      | 2000 | 训练窗口大小 |
| test_window       | 200  | 测试窗口大小 |
| step_size         | 200  | 滚动步长     |
| MIN_BARS_REQUIRED | 2500 | 最小数据要求 |

**数据量建议**：

- **最小要求**：2500 根K线（至少进行 2 个窗口的验证）
- **推荐数据量**：20000 根K线（约 90 个验证窗口）
- **数据不足时**：自动降级到单次训练（80/20 分割）

### 4. 回测任务

系统支持异步回测任务，可在后台运行并通过 WebSocket 获取进度更新。

**支持的时间间隔**：1m, 5m, 15m, 1h, 1d

**通过 Web 界面**：

1. 访问 http://localhost:8000/backtest
2. 配置回测参数（起止时间、时间间隔、初始资金等）
3. 提交回测任务
4. 查看任务状态和结果

**通过 API**：

```bash
curl -X POST http://localhost:8000/api/backtest/create \
  -H "Content-Type: application/json" \
  -d '{
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "interval": "1d",
    "initial_cash": 100000
  }'
```

### 5. API 端点

| 端点                          | 方法 | 说明               |
| ----------------------------- | ---- | ------------------ |
| `/api/strategy/config`        | GET  | 获取当前策略配置   |
| `/api/strategy/ml_config`     | POST | 设置 ML 策略参数   |
| `/api/models`                 | GET  | 列出所有可用模型   |
| `/api/models/current`         | GET  | 获取当前选择的模型 |
| `/api/models/select`          | POST | 选择要使用的模型   |
| `/api/models/train`           | POST | 启动模型训练       |
| `/api/models/train/status`    | GET  | 获取训练状态       |
| `/api/data/sync`              | POST | 启动数据同步       |
| `/api/data/sync/status`       | GET  | 获取同步状态       |
| `/api/backtest/create`        | POST | 创建回测任务       |
| `/api/backtest/tasks`         | GET  | 获取任务列表       |
| `/api/backtest/task/{id}`     | GET  | 获取任务详情       |

## 策略说明

### QlibMLStrategy（ML 策略）

基于机器学习模型预测的策略：

- 使用 LightGBM 预测未来收益率
- Top-K 排名选股（选择预测分数最高的 K 只股票）
- 定期再平衡
- 支持前端配置和模型热切换

#### 交易流程

每次交易迭代按以下 5 个步骤执行：

1. **获取时间** - 获取当前市场时间
2. **检查订单** - 检查是否有待处理订单
3. **获取数据** - 获取市场数据
4. **计算逻辑** - 生成预测并选择 Top-K 股票
5. **执行下单** - 执行再平衡操作

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
├── core/                # 核心配置模块
├── data/                # 数据适配与提供商
│   ├── providers.py     # Alpaca 数据提供者
│   └── qlib_adapter.py  # Qlib 格式转换
├── ml/                  # 机器学习模块
│   ├── features.py      # 特征工程
│   ├── model_manager.py # 模型管理
│   ├── trainer.py       # 模型训练
│   └── inference.py     # 模型推理
├── web/                 # Web 服务器
│   ├── server.py        # FastAPI 应用
│   └── backtest_tasks.py # 回测任务管理
├── qlib_ml_strategy.py  # ML 交易策略
└── main.py              # 程序入口

models/                  # 训练好的模型
datasets/                # Qlib 格式数据
data/                    # 回测任务数据库
reports/                 # 回测报告
```

## 开发

```bash
# 运行测试
make test

# 代码格式化
make format

# 代码检查
make check

# 清理缓存
make clean

# 全部操作（格式化 + 检查 + 测试）
make all
```

## 许可证

MIT
