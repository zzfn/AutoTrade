# Tasks: 集成 Qlib ML 策略引擎

## 1. 数据适配层

- [x] 1.1 创建 `autotrade/research/data/qlib_adapter.py` - Qlib 数据适配器基础类
- [x] 1.2 实现 `AlpacaDataProvider` - 从 Alpaca 获取历史数据
- [x] 1.3 实现 `YFinanceDataProvider` - 从 YFinance 获取历史数据（备用）
- [x] 1.4 实现 `QlibDataConverter` - 转换数据为 Qlib 格式并存储
- [x] 1.5 添加数据初始化脚本 `scripts/init_qlib_data.py`
- [x] 1.6 编写数据适配层单元测试

## 2. 特征工程

- [x] 2.1 创建 `autotrade/research/features/qlib_features.py` - Qlib 特征生成器
- [x] 2.2 实现基础技术指标因子（Alpha158 子集）
- [x] 2.3 实现特征预处理（标准化、缺失值处理）
- [x] 2.4 编写特征工程单元测试

## 3. 模型训练

- [x] 3.1 创建 `autotrade/research/models/` 目录结构
- [x] 3.2 实现 `ModelTrainer` 基础类 - 统一训练接口
- [x] 3.3 实现 `LightGBMTrainer` - LightGBM 模型训练
- [x] 3.4 实现模型保存/加载功能（含元数据）
- [x] 3.5 创建训练脚本 `scripts/train_model.py`
- [x] 3.6 实现 walk-forward 验证
- [x] 3.7 编写模型训练单元测试

## 4. ML 策略实现

- [x] 4.1 创建 `autotrade/strategies/qlib_ml_strategy.py`
- [x] 4.2 实现 `QlibMLStrategy` 继承 LumiBot Strategy
- [x] 4.3 实现模型加载和预测逻辑
- [x] 4.4 实现 Top-K 信号选股逻辑
- [x] 4.5 实现交易执行逻辑（买入/卖出/再平衡）
- [x] 4.6 在 `__init__.py` 中注册新策略
- [x] 4.7 编写策略单元测试

## 5. TradeManager 集成

- [x] 5.1 更新 `TradeManager` 支持策略工厂模式
- [x] 5.2 添加策略类型配置（momentum / ml）
- [x] 5.3 添加模型选择配置
- [x] 5.4 实现 rolling 更新 API 端点
- [x] 5.5 编写集成测试

## 6. 前端界面

- [x] 6.1 更新策略选择下拉框，添加 "Qlib ML Strategy" 选项
- [x] 6.2 创建模型管理页面 `/models`
  - [x] 6.2.1 模型列表展示（名称、版本、训练日期、性能指标）
  - [x] 6.2.2 模型选择功能
  - [x] 6.2.3 模型详情查看
- [x] 6.3 添加训练触发按钮和状态显示
- [x] 6.4 添加 Rolling Update 按钮
- [x] 6.5 添加策略参数配置（top_k, rebalance_period）

## 7. 文档和配置

- [x] 7.1 更新 README.md - 添加 Qlib ML 策略使用说明
- [x] 7.2 创建示例配置文件 `configs/qlib_ml_config.yaml`
- [x] 7.3 添加模型训练示例 notebook

## 8. 验证和测试

- [ ] 8.1 端到端回测验证 - 使用历史数据验证 ML 策略
- [ ] 8.2 与 MomentumStrategy 性能对比
- [ ] 8.3 前端功能手动测试
- [ ] 8.4 模拟盘测试（可选）

---

**依赖关系**:

- Phase 1-2 (数据 + 特征) 可并行
- Phase 3 (模型) 依赖 Phase 1-2
- Phase 4 (策略) 依赖 Phase 3
- Phase 5-6 (集成 + 前端) 依赖 Phase 4
- Phase 7-8 (文档 + 测试) 最后进行

**可并行工作**:

- 1.1-1.4 与 2.1-2.3 可并行
- 6.x (前端) 与 4.x (策略) 部分可并行开发

---

## 实现总结

### 已完成的核心模块

1. **数据适配层** (`autotrade/research/data/`)
   - `providers.py`: AlpacaDataProvider, YFinanceDataProvider, DataProviderFactory
   - `qlib_adapter.py`: QlibDataAdapter（数据获取、转换、存储、加载）

2. **特征工程** (`autotrade/research/features/`)
   - `qlib_features.py`: QlibFeatureGenerator, FeaturePreprocessor
   - 实现了 Alpha158 风格的技术指标因子

3. **模型训练** (`autotrade/research/models/`)
   - `trainer.py`: ModelTrainer, LightGBMTrainer, WalkForwardValidator
   - `model_manager.py`: ModelManager

4. **ML 策略** (`autotrade/strategies/`)
   - `qlib_ml_strategy.py`: QlibMLStrategy
   - 更新 `__init__.py` 添加策略注册表

5. **TradeManager 集成** (`autotrade/trade_manager.py`)
   - 策略工厂模式支持
   - ML 配置 API
   - Rolling 更新功能

6. **Web API 端点** (`autotrade/web_server.py`)
   - 策略配置 API
   - 模型管理 API

7. **前端界面** (`autotrade/ui/templates/`)
   - `models.html`: 模型管理页面

8. **脚本** (`scripts/`)
   - `init_qlib_data.py`: 数据初始化
   - `train_model.py`: 模型训练

9. **配置** (`configs/`)
   - `qlib_ml_config.yaml`: 示例配置

10. **测试** (`tests/`)
    - `test_qlib_ml.py`: 单元测试
