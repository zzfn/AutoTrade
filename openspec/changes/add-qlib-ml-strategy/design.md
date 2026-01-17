# Design: Qlib ML 策略引擎

## Context

AutoTrade 当前使用 LumiBot 框架运行基于技术指标的交易策略。为了提升预测能力，需要集成 Microsoft Qlib 的 ML 能力，同时保持与 LumiBot 的兼容性。

**约束条件**：

- 必须使用美股数据（Alpaca/YFinance）
- Qlib 默认针对中国市场，需要适配美股
- 模型训练和交易执行需要解耦
- 前端需要支持模型管理和策略选择

## Goals / Non-Goals

### Goals

- 实现 Qlib 美股数据适配器
- 支持 LightGBM 模型的训练和推理
- 创建 ML 驱动的 LumiBot 策略
- 支持离线训练和前端触发的 rolling 更新
- 前端可选择使用的策略和模型

### Non-Goals

- 深度学习模型支持（后续扩展）
- A 股数据支持（仅美股）
- 自动化模型选择（需人工选择）
- 实时 tick 级别预测（日频足够）

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (Web UI)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Strategy    │  │ Model       │  │ Rolling Update      │  │
│  │ Selection   │  │ Management  │  │ Trigger             │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
└─────────┼────────────────┼───────────────────┼──────────────┘
          │                │                   │
          ▼                ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│                     TradeManager                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Strategy Factory (MomentumStrategy | QlibMLStrategy)│    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
          │                                    │
          ▼                                    ▼
┌──────────────────────┐          ┌──────────────────────────┐
│   QlibMLStrategy     │          │   Model Trainer           │
│   (LumiBot Strategy) │◀─────────│   (Offline / Rolling)     │
│                      │  loads   │                           │
│  - load_model()      │  model   │  - train()                │
│  - predict()         │          │  - save_model()           │
│  - trade()           │          │  - rolling_update()       │
└──────────┬───────────┘          └────────────┬──────────────┘
           │                                   │
           ▼                                   ▼
┌──────────────────────────────────────────────────────────────┐
│                    Qlib Data Adapter                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  │
│  │ Alpaca Provider│  │YFinance Provider│  │ Qlib Format   │  │
│  └────────────────┘  └────────────────┘  │ Storage        │  │
│                                          └────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

## Decisions

### Decision 1: 数据存储格式

**选择**: 使用 Qlib 原生格式存储数据

**原因**:

- Qlib 的数据处理流水线高度优化
- 使用原生格式可直接利用 Qlib 的 DataHandler 和 Dataset
- 避免重复数据转换

**替代方案**:

- 自定义 Parquet 存储：灵活但需要重写数据加载逻辑
- 实时获取：延迟高，不适合训练

### Decision 2: 模型存储

**选择**: 使用 `models/` 目录 + JSON 元数据

**格式**:

```
models/
├── lightgbm_v1/
│   ├── model.pkl          # 模型文件
│   ├── metadata.json      # 版本、训练日期、参数、性能指标
│   └── config.yaml        # 训练配置
└── lightgbm_v2/
    └── ...
```

**原因**:

- 简单直观，易于版本管理
- 元数据便于前端展示
- 符合项目现有结构（见 spec: ML Models Directory）

### Decision 3: 信号到交易的转换

**选择**: Top-K 排名策略

**逻辑**:

1. ML 模型对所有候选股票输出预测分数
2. 选择 Top-K 分数最高的股票
3. 等权重分配资金买入
4. 分数下降到排名外则卖出

**参数**:

- `top_k`: 持仓数量（默认 5）
- `rebalance_period`: 再平衡周期（默认 1 天）

**原因**:

- 简单易理解
- 与学术研究方法一致
- 易于回测验证

### Decision 4: Rolling 更新机制

**选择**: 前端触发 + 后台异步执行

**流程**:

1. 用户在前端点击 "Rolling Update" 按钮
2. 后端启动异步训练任务
3. 训练完成后保存新模型版本
4. 用户手动切换到新模型（不自动切换）

**原因**:

- 给用户完全控制权
- 避免意外中断正在运行的策略
- 支持 A/B 测试不同模型版本

## Data Flow

### 训练流程

```
1. DataAdapter.fetch_data(symbols, start_date, end_date)
   ↓
2. DataAdapter.prepare_qlib_data()  # 转换为 Qlib 格式
   ↓
3. FeatureGenerator.generate()      # 生成因子/特征
   ↓
4. ModelTrainer.train(config)       # 训练模型
   ↓
5. ModelTrainer.save(model_dir)     # 保存模型和元数据
```

### 交易流程

```
1. QlibMLStrategy.initialize()
   ├── load_model(model_path)
   └── setup_universe(symbols)
   ↓
2. QlibMLStrategy.on_trading_iteration()
   ├── fetch_latest_data()
   ├── generate_features()
   ├── model.predict() → scores
   ├── rank_and_select_top_k(scores)
   └── execute_trades()
```

## Risks / Trade-offs

| Risk                | Impact | Mitigation                             |
| ------------------- | ------ | -------------------------------------- |
| Qlib 美股适配复杂度 | 中     | 参考社区美股适配方案；最坏情况手动实现 |
| 数据质量问题        | 高     | 添加数据校验；使用多数据源对比         |
| 模型过拟合          | 高     | 使用滚动训练；walk-forward 验证        |
| 训练时间长          | 低     | 异步执行；支持后台运行                 |

## Migration Plan

1. **Phase 1: 数据适配** - 实现 Alpaca/YFinance 到 Qlib 格式转换
2. **Phase 2: 模型训练** - 实现离线训练脚本和模型管理
3. **Phase 3: ML 策略** - 实现 QlibMLStrategy 并接入 LumiBot
4. **Phase 4: 前端集成** - 添加模型管理和策略选择界面
5. **Phase 5: Rolling 更新** - 实现前端触发的模型更新

每个阶段独立可验证，可以逐步发布。

## Open Questions

1. ~~数据源优先级~~ → 已确定：Alpaca 优先，YFinance 备用
2. 初始训练数据的时间范围？建议：2 年历史数据
3. 是否需要支持自定义因子？建议：先使用 Qlib 内置因子，后续扩展
