# Proposal: add-walk-forward-validation

## Summary

为 ML 模型训练添加 **Walk-Forward 验证**（滚动窗口验证）功能，提供更稳健的时序模型评估方法。

## Motivation

### 当前问题
1. **单一时间点验证**：现有训练流程使用简单的 80/20 划分，只在一个时间点测试模型
2. **过拟合风险**：模型可能在特定时间段表现良好，但在其他市场环境下失效
3. **缺少稳健性指标**：无法评估模型在不同市场环境下的稳定性

### Walk-Forward 验证的优势
- ✅ **模拟实盘**：每次只用历史数据预测未来，符合实盘交易场景
- ✅ **多周期测试**：在多个时间窗口验证，避免运气好
- ✅ **稳健性评估**：提供指标的均值和标准差（如 IC ± std）
- ✅ **参数稳定性**：验证模型在不同市场环境下的表现

## Proposed Solution

### 1. 固定参数设计

Walk-Forward 验证使用固定窗口参数（单位：**根K线数量**）：

```python
# Walk-Forward 窗口配置（固定）
WALK_FORWARD_CONFIG = {
    "train_window": 2000,   # 训练窗口：2000 根K线
    "test_window": 200,     # 测试窗口：200 根K线
    "step_size": 200,       # 滚动步长：200 根K线
}

# 最小数据要求
MIN_BARS_REQUIRED = 2500   # train_window + test_window
```

**用户配置**：通过 `num_bars` 参数控制获取的总数据量（如 20000 根K线）。

**验证窗口数**：`(num_bars - train_window) / step_size ≈ 90 个窗口`

### 2. 简化逻辑

- **无需配置文件**：Walk-Forward 验证成为默认行为
- **固定参数**：所有训练使用相同的窗口参数
- **数据充足检查**：
  - 数据不足 `MIN_BARS_REQUIRED` → 显示警告但仍尝试训练
  - 数据充足 → 正常执行 Walk-Forward 验证

### 3. WalkForwardValidator 集成

`WalkForwardValidator` 类已存在于 `autotrade/ml/trainer.py:306`，需要：
- 修改 `start_model_training_internal` 直接调用 Walk-Forward 验证
- 收集所有窗口的结果并计算聚合指标

在模型训练页面显示：
- Walk-Forward 验证进度（当前窗口 X/总计 Y）
- 聚合指标：IC (mean ± std), ICIR (mean ± std)
- 每个窗口的详细结果（可展开查看）

## Impact

### 用户影响
- **训练时间**：Walk-Forward 验证会增加训练时间（约 3-5x，取决于窗口数）
- **模型质量**：所有训练都使用更可靠的验证方法，减少上线风险
- **透明化**：用户无需理解 Walk-Forward，系统自动处理
- **数据建议**：训练时可能提示用户增加历史数据以获得更好结果

### 系统影响
- **新增依赖**：无（使用现有的 `WalkForwardValidator`）
- **配置简化**：移除 `validation` 配置段，减少配置复杂度
- **代码变更**：主要集中在 `autotrade/web/server.py` 的训练流程
- **向后兼容**：训练流程改变，但输出模型格式不变

## Alternatives Considered

### 1. K-Fold 交叉验证
**缺点**：不适合时序数据（会打乱时间顺序）

### 2. 简单的时间序列分割
**缺点**：只测试一次，无法评估稳健性

### 3. 保持现状
**缺点**：上线风险高，模型可能在实盘中失效

## Open Questions

1. **进度展示粒度？** 每个窗口都更新（已决定：每个窗口都更新）
2. **结果存储？** 是否需要保存每个窗口的详细结果到磁盘（建议：只保存聚合指标，前端显示详细结果）

## Dependencies

- 无新增外部依赖
- 依赖现有的 `WalkForwardValidator` 类
- 与现有 `qlib-ml-strategy` spec 兼容
