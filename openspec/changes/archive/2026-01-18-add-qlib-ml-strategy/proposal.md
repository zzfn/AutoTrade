# Change: 集成 Qlib ML 策略引擎

## Why

当前系统仅支持基于 SMA 的简单动量策略，无法利用机器学习模型进行更精确的股票预测。通过集成 Microsoft Qlib 框架，可以实现：

- 使用 ML 模型（如 LightGBM）进行股票收益预测
- 基于预测分数驱动交易决策，替代传统技术指标
- 支持模型的离线训练和在线 rolling 更新

## What Changes

1. **数据管道**
   - 新增 Qlib 数据适配器，支持从 Alpaca/YFinance 获取美股数据
   - 实现 Qlib 格式的数据存储和管理

2. **模型训练**
   - 新增离线模型训练脚本，支持 LightGBM 等传统 ML 模型
   - 新增模型存储和版本管理机制
   - 支持前端触发的 rolling 模型更新

3. **ML 策略**
   - 新增 `QlibMLStrategy`，完全由 ML 模型驱动
   - 模型输出预测分数，策略基于分数排名进行交易决策
   - 支持在前端选择使用的策略模型

4. **前端界面**
   - 新增模型管理界面（训练、查看、选择模型）
   - 新增 rolling 更新按钮
   - 策略选择支持 ML 策略

## Impact

- **Affected specs**: 新增 `qlib-ml-strategy` 规范
- **Affected code**:
  - `autotrade/strategies/` - 新增 QlibMLStrategy
  - `autotrade/research/` - 新增数据适配和模型训练模块
  - `autotrade/ui/` - 更新前端界面
  - `autotrade/trade_manager.py` - 支持新策略类型
- **Dependencies**: 项目已包含 `pyqlib` 依赖
