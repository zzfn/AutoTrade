# Design: add-walk-forward-validation

## Architecture Overview

### 组件关系

```
┌─────────────────────────────────────────────────────────┐
│                     Web UI (models.html)                 │
│  - 训练按钮触发                                           │
│  - 显示 Walk-Forward 验证进度                             │
│  - 展示聚合指标 (IC ± std)                                │
└────────────────────┬────────────────────────────────────┘
                     │ POST /api/models/train
                     │
┌────────────────────▼────────────────────────────────────┐
│              server.py: start_model_training_internal()   │
│  - 检查数据长度是否满足最小要求                            │
│  - 分支:                                                 │
│    * 数据充足 → Walk-Forward 验证（固定参数）              │
│    * 数据不足 → 单次训练 + 警告                           │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌──────────────┐      ┌─────────────────────┐
│ 单次训练      │      │ WalkForwardValidator│
│ (降级逻辑)    │      │  - 固定窗口参数      │
│ + 警告提示    │      │  - 循环滚动窗口      │
│              │      │  - 训练 + 验证       │
└──────────────┘      │  - 收集指标          │
                      └─────────────────────┘
                               │
                               ▼
                      ┌────────────────┐
                      │ 结果聚合        │
                      │ - IC mean ± std│
                      │ - ICIR mean    │
                      └────────────────┘
```

## Data Flow

### 1. 固定参数配置
```python
# server.py: start_model_training_internal()

# Walk-Forward 窗口配置（固定，单位：根K线数量）
WALK_FORWARD_CONFIG = {
    "train_window": 2000,   # 训练窗口：2000 根K线
    "test_window": 200,     # 测试窗口：200 根K线
    "step_size": 200,       # 滚动步长：200 根K线
}

MIN_BARS_REQUIRED = 2500   # 最小数据要求

# 用户通过 num_bars 控制总数据量
num_bars = 20000  # 用户配置的K线数量

# 数据充足性检查
if len(X) >= MIN_BARS_REQUIRED:
    # Walk-Forward 验证
    validator = WalkForwardValidator(
        trainer_class=LightGBMTrainer,
        train_window=WALK_FORWARD_CONFIG["train_window"],
        test_window=WALK_FORWARD_CONFIG["test_window"],
        step_size=WALK_FORWARD_CONFIG["step_size"]
    )
    results = validator.validate(X, y)
    # 约进行 (num_bars - 2000) / 200 = 90 个窗口的验证
else:
    # 降级到单次训练
    log_warning(f"数据不足 {MIN_BARS_REQUIRED} 根K线，降级到单次训练")
    # ...
```

### 2. Walk-Forward 验证流程
```python
# 数据充足性检查
if len(X) >= MIN_BARS_REQUIRED:
    # 执行 Walk-Forward 验证
    validator = WalkForwardValidator(
        trainer_class=LightGBMTrainer,
        **WALK_FORWARD_CONFIG
    )
    results = validator.validate(X, y)

    # 聚合结果
    aggregated = {
        "ic_mean": np.mean([r["ic"] for r in results]),
        "ic_std": np.std([r["ic"] for r in results]),
        "icir_mean": np.mean([r["icir"] for r in results]),
        "num_windows": len(results)
    }
else:
    # 降级到单次训练
    log_warning(f"数据不足 {MIN_BARS_REQUIRED} 根K线")
    # 执行现有 80/20 分割逻辑...
```

### 3. 进度更新
```python
# Walk-Forward 有多个窗口，需要更细粒度的进度报告
for i, window_result in enumerate(results):
    progress = 50 + (50 * (i + 1) / num_windows)  # 50-100%
    training_status["progress"] = progress
    training_status["message"] = f"验证窗口 {i+1}/{num_windows}..."
```

## UI/UX Design

### 训练配置展示
```
┌─────────────────────────────────────────┐
│ 训练配置                                 │
├─────────────────────────────────────────┤
│ 数据数量：20000 根K线                    │
│ Walk-Forward 验证：✓ 启用                │
│   - 训练窗口：2000 根K线                 │
│   - 测试窗口：200 根K线                  │
│   - 滚动步长：200 根K线                  │
│   - 预计窗口数：90 个                    │
└─────────────────────────────────────────┘
```

### 数据不足警告
```
┌─────────────────────────────────────────┐
│ ⚠️  数据量警告                          │
├─────────────────────────────────────────┤
│ 当前数据长度：1200 根K线                 │
│                                          │
│ Walk-Forward 验证需要至少 2500 根K线。   │
│ 将降级到单次训练（80% 训练 + 20% 验证）。 │
│                                          │
│ 建议增加 num_bars 到 5000+ 以获得更稳健  │
│ 的模型评估（推荐 20000 根K线）。          │
│                                          │
│ [继续训练] [取消]                        │
└─────────────────────────────────────────┘
```

### 验证进度展示
```
┌─────────────────────────────────────────┐
│ Walk-Forward 验证进度                   │
├─────────────────────────────────────────┤
│ 窗口 3/10 ████████░░░░░░░░ 30%          │
│                                          │
│ 当前窗口：                              │
│   IC: 0.042  ICIR: 1.23                 │
│                                          │
│ 累计平均（已完成窗口）：                 │
│   IC: 0.041 ± 0.008                     │
│   ICIR: 1.18 ± 0.15                     │
└─────────────────────────────────────────┘
```

### 完成后结果展示
```
┌─────────────────────────────────────────┐
│ Walk-Forward 验证完成 ✓                 │
├─────────────────────────────────────────┤
│ 验证窗口数：10                          │
│                                          │
│ IC: 0.041 ± 0.008                       │
│ ICIR: 1.18 ± 0.15                       │
│                                          │
│ [展开查看每个窗口详细结果 ▼]            │
└─────────────────────────────────────────┘
```

## Implementation Details

### 1. 配置验证
- 确保 `train_window > test_window`
- 确保 `step_size <= test_window`
- 验证数据长度足够进行至少 2 个窗口

### 2. 错误处理
- 数据不足：提示需要更多历史数据
- 训练失败：记录具体哪个窗口失败，继续其他窗口
- 内存不足：提示减小窗口大小

### 3. 性能优化
- 每个窗口训练完成后立即释放模型
- 可选：并行训练多个窗口（高级功能）

## Trade-offs

### 训练时间 vs 模型质量
- **Walk-Forward**: 训练时间 3-5x，但模型更稳健
- **单次训练**: 快速，但可能过拟合

**决策**: 默认禁用，让用户根据需求选择

### 进度更新频率
- **每个窗口更新**: 更实时，但 API 请求多
- **批量更新**: 减少 API 负载，但体验稍差

**决策**: 每个窗口更新（已有轮询机制，开销可控）

### 结果存储
- **只返回聚合指标**: 节省内存，但无法回溯
- **保存所有窗口结果**: 可追溯，但占用空间

**决策**: 返回所有结果（前端可选择展示），元数据只保存聚合指标

## Testing Strategy

### 单元测试
- `WalkForwardValidator.validate()` 返回正确格式的结果
- 进度更新在正确的时间点触发
- 聚合指标计算正确

### 集成测试
- `validation.enabled: false` → 单次训练
- `validation.enabled: true` → Walk-Forward 验证
- 配置缺失 → 使用默认值

### 手动测试
- 小数据集（1个月）：快速验证流程
- 完整数据集（1年+）：验证性能
- 前端展示：验证进度和结果正确显示
