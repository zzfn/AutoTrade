# walk-forward-validation Spec Delta

## Purpose

添加 Walk-Forward 验证功能，为 ML 模型提供更稳健的时序评估方法。

## ADDED Requirements

### Requirement: Walk-Forward 验证默认启用

系统 SHALL 将 Walk-Forward 验证作为所有模型训练的默认方法：

- 无需配置文件控制，始终使用 Walk-Forward 验证
- 使用固定窗口参数（单位：根K线数量）：`train_window=2000`, `test_window=200`, `step_size=200`
- 最小数据要求：`2500` 根K线
- 数据不足时降级到单次训练并给出警告

#### Scenario: 标准训练（数据充足）

- **GIVEN** 用户配置 `num_bars=20000`（或更多）
- **WHEN** 用户启动模型训练
- **THEN** 系统使用固定窗口参数（2000/200/200）执行 Walk-Forward 验证
- **AND** 在约 90 个时间窗口上训练和验证模型
- **AND** 计算聚合指标（IC mean ± std, ICIR mean）

#### Scenario: 较小数据集训练

- **GIVEN** 用户配置 `num_bars=5000`
- **WHEN** 用户启动模型训练
- **THEN** 系统使用相同的固定窗口参数执行 Walk-Forward 验证
- **AND** 在约 15 个时间窗口上训练和验证模型
- **AND** 正常计算聚合指标

#### Scenario: 数据不足降级

- **GIVEN** 训练数据不足 2500 根K线
- **WHEN** 用户启动模型训练
- **THEN** 系统降级到单次训练（80% 训练 + 20% 验证）
- **AND** 显示警告："数据严重不足，Walk-Forward 验证已禁用。建议至少提供 5000 根K线（推荐 20000）"

---

### Requirement: Walk-Forward 验证执行

系统 SHALL 提供完整的 Walk-Forward 验证执行流程：

- 调用 `WalkForwardValidator.validate()` 执行验证
- 使用固定窗口参数（2000 根K线训练 + 200 根K线测试）
- 每次滚动 200 根K线
- 收集每个窗口的评估指标
- 计算聚合统计量（均值、标准差）

#### Scenario: 验证执行

- **GIVEN** 数据充足（≥ 2500 根K线）
- **WHEN** 验证开始
- **THEN** 系统按照固定窗口参数滚动训练
- **AND** 每个窗口使用历史数据训练，在未来数据上验证
- **AND** 记录每个窗口的 IC、ICIR 等指标

#### Scenario: 结果聚合

- **GIVEN** 所有窗口验证完成
- **WHEN** 计算最终结果
- **THEN** 系统计算 IC 的均值和标准差
- **AND** 计算 ICIR 的均值和标准差
- **AND** 返回每个窗口的详细结果

---

### Requirement: 验证进度展示

系统 SHALL 在前端展示 Walk-Forward 验证的实时进度：

- 显示当前验证窗口编号（如：窗口 3/90）
- 显示当前窗口的评估指标
- 显示已完成窗口的累计平均指标
- 更新总体进度百分比

#### Scenario: 进度更新

- **GIVEN** Walk-Forward 验证进行中
- **WHEN** 完成一个窗口的验证
- **THEN** 系统更新训练状态 `training_status`
- **AND** 前端轮询获取最新进度
- **AND** 显示当前窗口和累计结果

#### Scenario: 完成通知

- **GIVEN** 所有窗口验证完成
- **WHEN** 最终结果计算完成
- **THEN** 系统显示完成消息
- **AND** 展示聚合指标（IC mean ± std, ICIR mean）
- **AND** 可选展开查看每个窗口的详细结果

---

### Requirement: 模型元数据扩展

系统 SHALL 在模型元数据中记录 Walk-Forward 验证结果：

- 保存聚合指标（IC mean, IC std, ICIR mean）
- 记录验证配置（窗口数、train_window, test_window, step_size）
- 支持在模型列表中显示稳健性指标

#### Scenario: 元数据保存

- **GIVEN** Walk-Forward 验证完成
- **WHEN** 保存模型元数据
- **THEN** 元数据包含 `walk_forward` 字段
- **AND** 记录 `ic_mean`, `ic_std`, `icir_mean`
- **AND** 记录 `num_windows`, `train_window`, `test_window`, `step_size`

#### Scenario: 模型列表展示

- **GIVEN** 模型通过 Walk-Forward 验证训练
- **WHEN** 用户在模型列表中查看模型
- **THEN** 显示 IC 聚合指标（如：IC: 0.041 ± 0.008）
- **AND** 区分单次训练和 Walk-Forward 验证的模型

---

## MODIFIED Requirements

### Requirement: Unified Model Training (from qlib-ml-strategy)

**BEFORE**:
The system SHALL provide a unified interface for training ML models, capable of handling both initial historical training and rolling updates with recent data.

**AFTER**:
The system SHALL provide a unified interface for training ML models，使用 Walk-Forward 验证作为默认方法：
- 数据充足（≥ 2500 根K线）时使用 Walk-Forward 滚动窗口验证
- 数据不足时自动降级到单次训练（80/20）
- 使用固定窗口参数：train_window=2000, test_window=200, step_size=200

#### Scenario: User initiates training (extended)

- **WHEN** user clicks "Train Model" in the UI
- **AND** training data is sufficient (>= 2500 bars)
- **THEN** system performs Walk-Forward validation with fixed parameters
- **AND** saves aggregated results as new model version

- **WHEN** user clicks "Train Model" in the UI
- **AND** training data is insufficient (< 2500 bars)
- **THEN** system performs single training split (80/20)
- **AND** shows warning about data insufficiency
- **AND** saves result as new model version

#### Scenario: Background execution (extended)

- **WHEN** training is triggered
- **THEN** the process runs asynchronously
- **AND** UI displays progress:
  - Walk-Forward mode: data loading → feature generation → validation (window 1/N, 2/N, ...) → aggregation
  - Fallback mode: data loading → feature generation → training → evaluation
- **AND** user is notified upon success or failure with appropriate metrics

---

## Cross-References

- **Related Spec**: `qlib-ml-strategy` - 扩展模型训练功能
- **Related Spec**: `model-management` - 模型元数据存储
- **Uses**: `WalkForwardValidator` class (已存在于 `autotrade/ml/trainer.py:306`)
- **Config**: 无需配置文件，使用代码常量 `WALK_FORWARD_CONFIG`
