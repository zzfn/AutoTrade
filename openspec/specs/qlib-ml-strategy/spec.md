# qlib-ml-strategy Specification

## Purpose
TBD - created by archiving change add-qlib-ml-strategy. Update Purpose after archive.
## Requirements
### Requirement: Qlib 数据适配

系统 SHALL 提供数据适配器，支持从 Alpaca 和 YFinance 获取美股历史数据并转换为 Qlib 格式：

- 支持 Alpaca API 作为主要数据源
- 支持 YFinance 作为备用数据源
- 数据存储于 `data/qlib/` 目录
- 支持增量数据更新

#### Scenario: 获取美股历史数据

- **WHEN** 用户请求获取股票 AAPL 的历史数据
- **THEN** 系统从 Alpaca API 获取数据
- **AND** 数据转换为 Qlib 格式存储

#### Scenario: 数据源降级

- **WHEN** Alpaca API 不可用
- **THEN** 系统自动切换到 YFinance 获取数据
- **AND** 记录降级日志

---

### Requirement: 特征生成

系统 SHALL 提供特征生成器，基于历史价格数据生成 ML 模型所需的技术指标特征：

- 支持 Alpha158 基础因子集
- 支持特征标准化和缺失值处理
- 特征定义可配置

#### Scenario: 生成技术指标特征

- **WHEN** 提供股票 AAPL 的历史 OHLCV 数据
- **THEN** 系统生成包含 MA、RSI、MACD 等技术指标的特征矩阵
- **AND** 特征值经过标准化处理

---

### Requirement: ML 策略执行

系统 SHALL 提供 `QlibMLStrategy`，基于 ML 模型预测驱动交易决策：

- 继承 LumiBot Strategy 基类
- 使用 Top-K 排名策略选股
- 支持配置持仓数量（top_k）和再平衡周期
- 完全替代技术指标驱动的交易逻辑

#### Scenario: 基于预测信号交易

- **WHEN** 交易迭代开始
- **THEN** 策略获取所有候选股票的最新数据
- **AND** 生成特征并调用模型预测
- **AND** 选择预测分数最高的 Top-K 股票
- **AND** 执行买入/卖出以达到目标持仓

#### Scenario: 再平衡持仓

- **WHEN** 到达再平衡周期
- **AND** 当前持仓股票的预测分数下降到 Top-K 之外
- **THEN** 系统卖出该股票
- **AND** 买入新的 Top-K 股票

---

### Requirement: 模型管理界面

系统 SHALL 提供前端界面管理 ML 模型：

- 展示所有可用模型列表
- 显示模型元数据（名称、版本、训练日期、性能指标）
- 支持选择用于交易的模型
- 支持查看模型详情

#### Scenario: 查看模型列表

- **WHEN** 用户访问模型管理页面
- **THEN** 系统展示所有可用模型
- **AND** 每个模型显示名称、版本、训练日期、IC/ICIR 等指标

#### Scenario: 选择交易模型

- **WHEN** 用户在模型列表中选择一个模型
- **THEN** 该模型被设为当前使用的模型
- **AND** 下次策略启动时使用该模型

---

### Requirement: 策略选择

系统 SHALL 支持在前端选择使用的交易策略：

- 支持选择 "MomentumStrategy"（SMA 驱动）
- 支持选择 "QlibMLStrategy"（ML 模型驱动）
- 策略选择影响实盘和回测

#### Scenario: 选择 ML 策略

- **WHEN** 用户在策略下拉框选择 "Qlib ML Strategy"
- **AND** 选择要使用的模型
- **THEN** 系统使用 `QlibMLStrategy` 进行交易
- **AND** 策略加载选定的 ML 模型

#### Scenario: 切换回动量策略

- **WHEN** 用户选择 "Momentum Strategy"
- **THEN** 系统使用原有的 `MomentumStrategy`
- **AND** 不加载任何 ML 模型

### Requirement: Unified Model Training

The system SHALL provide a unified interface for training ML models, capable of handling both initial historical training and rolling updates with recent data.

#### Scenario: User initiates training

- **WHEN** user clicks "Train Model" in the UI
- **AND** confirms configuration (symbols, parameters)
- **THEN** system starts a background training task
- **AND** saves the result as a new model version upon completion

#### Scenario: Background execution

- **WHEN** training is triggered
- **THEN** the process runs asynchronously
- **AND** UI displays progress similar to previous rolling update
- **AND** user is notified upon success or failure

