# qlib-ml-strategy Specification

## Purpose

定义基于 Microsoft Qlib 的机器学习交易策略能力，包括数据适配、模型训练、交易信号生成和前端管理界面。

## ADDED Requirements

### Requirement: Qlib 数据适配

系统 SHALL 提供数据适配器，支持从 Alpaca 和 YFinance 获取美股历史数据并转换为 Qlib 格式：

- 支持 Alpaca API 作为主要数据源
- 支持 YFinance 作为备用数据源
- 数据存储于 `datasets/` 目录
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

### Requirement: 模型训练

系统 SHALL 支持 ML 模型的离线训练和保存：

- 支持 LightGBM 模型
- 支持 walk-forward 验证
- 模型保存包含元数据（版本、训练日期、参数、性能指标）
- 模型存储于 `models/` 目录

#### Scenario: 离线训练 LightGBM 模型

- **WHEN** 用户运行模型训练脚本并指定配置
- **THEN** 系统使用历史数据训练 LightGBM 模型
- **AND** 模型保存到 `models/<model_name>/`
- **AND** 生成 `metadata.json` 包含训练信息

#### Scenario: Walk-forward 验证

- **WHEN** 训练配置启用 walk-forward 验证
- **THEN** 系统使用滚动窗口训练和验证
- **AND** 输出每个窗口的性能指标

---

### Requirement: Rolling 模型更新

系统 SHALL 支持通过前端触发的模型 rolling 更新：

- 前端提供 "Rolling Update" 按钮
- 更新在后台异步执行
- 更新完成后保存为新版本模型
- 不自动切换使用中的模型

#### Scenario: 触发 rolling 更新

- **WHEN** 用户在前端点击 "Rolling Update" 按钮
- **THEN** 系统启动后台模型更新任务
- **AND** 前端显示更新进度
- **AND** 更新完成后通知用户

#### Scenario: 新模型版本创建

- **WHEN** Rolling 更新完成
- **THEN** 新模型保存为新版本（如 `lightgbm_v2`）
- **AND** 原模型保持不变
- **AND** 用户可手动切换到新模型

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
