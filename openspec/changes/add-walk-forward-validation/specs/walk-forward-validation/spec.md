# walk-forward-validation Spec Delta

## Purpose

添加 Walk-Forward 验证功能，为 ML 模型提供更稳健的时序评估方法。

## ADDED Requirements

### Requirement: Walk-Forward 验证默认启用

系统 SHALL 将 Walk-Forward 验证作为所有模型训练的默认方法：

- 无需配置文件控制，始终使用 Walk-Forward 验证
- 使用固定默认参数：`train_window=180`, `test_window=15`, `step_size=15`
- 根据数据长度动态调整窗口大小，确保验证能进行
- 数据不足时降级到单次训练并给出警告

#### Scenario: 标准训练（数据充足）

- **GIVEN** 训练数据有 10 个月以上历史
- **WHEN** 用户启动模型训练
- **THEN** 系统使用默认窗口参数（180/15/15）执行 Walk-Forward 验证
- **AND** 在多个时间窗口上训练和验证模型
- **AND** 计算聚合指标（IC mean ± std, ICIR mean）

#### Scenario: 动态窗口调整（数据较少）

- **GIVEN** 训练数据只有 4 个月历史
- **WHEN** 用户启动模型训练
- **THEN** 系统自动缩小窗口参数（60/7/7）以适应数据长度
- **AND** 执行 Walk-Forward 验证
- **AND** 显示警告："数据量较少，建议增加历史数据以获得更稳健的模型"

#### Scenario: 数据不足降级

- **GIVEN** 训练数据不足 2 个月
- **WHEN** 用户启动模型训练
- **THEN** 系统降级到单次训练（80% 训练 + 20% 验证）
- **AND** 显示警告："数据严重不足，Walk-Forward 验证已禁用。建议至少提供 4 个月历史数据"

---

### Requirement: Walk-Forward 验证执行

系统 SHALL 提供完整的 Walk-Forward 验证执行流程：

- 调用 `WalkForwardValidator.validate()` 执行验证
- 支持滚动窗口训练（训练窗口 + 测试窗口）
- 收集每个窗口的评估指标
- 计算聚合统计量（均值、标准差）

#### Scenario: 验证执行

- **GIVEN** 启用 Walk-Forward 验证
- **WHEN** 验证开始
- **THEN** 系统按照配置的窗口大小滚动训练
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

- 显示当前验证窗口编号（如：窗口 3/10）
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
- 记录验证配置（窗口数、train_window, test_window）
- 支持在模型列表中显示稳健性指标

#### Scenario: 元数据保存

- **GIVEN** Walk-Forward 验证完成
- **WHEN** 保存模型元数据
- **THEN** 元数据包含 `walk_forward` 字段
- **AND** 记录 `ic_mean`, `ic_std`, `icir_mean`
- **AND** 记录 `num_windows`, `train_window`, `test_window`

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
- 数据充足时使用 Walk-Forward 滚动窗口验证
- 数据不足时自动降级到单次训练（80/20）
- 动态调整窗口大小以适应数据长度

#### Scenario: User initiates training (extended)

- **WHEN** user clicks "Train Model" in the UI
- **AND** training data is sufficient (4+ months)
- **THEN** system performs Walk-Forward validation with appropriate window size
- **AND** saves aggregated results as new model version

- **WHEN** user clicks "Train Model" in the UI
- **AND** training data is insufficient (< 2 months)
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
- **Uses**: `WalkForwardValidator` class (already exists in `autotrade/ml/trainer.py`)
- **Config**: 无需配置文件，使用代码常量 `DEFAULT_WALK_FORWARD_CONFIG`
