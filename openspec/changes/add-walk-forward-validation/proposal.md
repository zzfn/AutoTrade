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

### 1. 移除配置文件依赖
- 从 `qlib_ml_config.yaml` 中移除 `validation` 配置段
- Walk-Forward 验证成为**默认行为**
- 参数在代码中写死（可配置的常量）

### 2. 默认参数（平衡模式）
```python
DEFAULT_WALK_FORWARD_CONFIG = {
    "train_window": 2000,   # 2000 根K线训练（5min数据约 18 天）
    "test_window": 200,     # 200 根K线测试（5min数据约 1.8 天）
    "step_size": 200        # 200 根K线滚动
}
```

### 3. 动态窗口调整
- 根据数据点数量自动调整窗口大小
- 确保至少能进行 2 个窗口的验证
- 调整策略：
  - 如果数据不足 5000 根：缩小到 `train_window=1000, test_window=100, step_size=100`
  - 如果数据不足 2000 根：进一步缩小到 `train_window=500, test_window=50, step_size=50`
  - 如果数据不足 1000 根：降级到单次训练（80/20）+ 警告提示

### 4. WalkForwardValidator 集成
`WalkForwardValidator` 类已存在于 `autotrade/ml/trainer.py:306`，需要：
- 修改 `start_model_training_internal` 直接调用 Walk-Forward 验证
- 实现动态窗口调整逻辑
- 收集所有窗口的结果并计算聚合指标

### 4. 前端展示增强
在模型训练页面显示：
- Walk-Forward 验证进度（每个窗口）
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
