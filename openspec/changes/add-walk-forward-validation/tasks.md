# Tasks: add-walk-forward-validation

## Implementation Tasks

### Phase 1: 动态窗口参数调整

- [ ] **Task 1.1**: 定义默认常量
  - 在 `autotrade/web/server.py` 顶部定义 `DEFAULT_WALK_FORWARD_CONFIG`
  - 参数：`train_window=180`, `test_window=15`, `step_size=15`
  - **验证**: 常量定义正确，可访问

- [ ] **Task 1.2**: 实现动态窗口调整函数
  - 创建 `adjust_walk_forward_windows(data_length_days)` 函数
  - 根据数据长度返回窗口参数：
    - `>= 300 天` → (180, 15, 15)
    - `>= 120 天` → (90, 10, 10)
    - `>= 60 天` → (60, 7, 7)
    - `< 60 天` → None（降级）
  - **验证**: 不同数据长度返回正确参数

- [ ] **Task 1.3**: 集成到训练流程
  - 在 `start_model_training_internal()` 中计算数据长度
  - 调用 `adjust_walk_forward_windows()` 获取窗口参数
  - 根据返回值分支：`config is not None` → Walk-Forward，否则 → 单次训练
  - **验证**: 数据充足时走 Walk-Forward，不足时降级

### Phase 2: Walk-Forward 验证集成

- [ ] **Task 2.1**: 实现 Walk-Forward 验证主流程
  - 在 `start_model_training_internal()` 中添加验证分支
  - 初始化 `WalkForwardValidator`（传入动态调整后的参数）
  - 调用 `validator.validate(X, y)` 执行验证
  - **验证**: 验证流程能正常运行无错误

- [ ] **Task 2.2**: 实现结果聚合逻辑
  - 收集所有窗口的评估指标（IC, ICIR, etc.）
  - 计算聚合统计量：`ic_mean`, `ic_std`, `icir_mean`
  - 格式化为前端可用的结果结构
  - **验证**: 聚合指标计算正确（与手动计算对比）

- [ ] **Task 2.3**: 模型保存逻辑调整
  - Walk-Forward 模式下：
    - 选择最佳窗口的模型，或
    - 使用全部数据训练最终模型
  - 更新模型元数据，包含 Walk-Forward 结果
  - **验证**: 模型能正确保存和加载

### Phase 3: 进度展示

- [ ] **Task 3.1**: 实现细粒度进度更新
  - 在 Walk-Forward 验证循环中更新 `training_status`
  - 计算进度百分比：`50 + 50 * (current_window / total_windows)`
  - 更新消息：`"验证窗口 {i}/{total}..."`
  - **验证**: 前端能实时看到进度更新

- [ ] **Task 3.2**: 扩展训练状态结构
  - 添加 `walk_forward_progress` 字段（可选）
  - 包含：`current_window`, `total_windows`, `window_results`
  - 前端可选择性展示详细窗口信息
  - **验证**: 状态结构包含所有必要信息

### Phase 4: 前端 UI 增强

- [ ] **Task 4.1**: 更新训练配置展示
  - 在"训练配置"区域显示 Walk-Forward 自动启用
  - 显示动态计算的窗口参数
  - 标识当前使用的窗口大小
  - **验证**: 用户能清楚看到自动配置

- [ ] **Task 4.2**: 增强 Walk-Forward 进度展示
  - 显示当前窗口编号（如：窗口 3/10）
  - 显示当前窗口的 IC、ICIR
  - 显示已完成窗口的累计平均
  - **验证**: 进度展示清晰、实时

- [ ] **Task 4.3**: 展示 Walk-Forward 完成结果
  - 显示聚合指标：`IC: 0.041 ± 0.008`, `ICIR: 1.18 ± 0.15`
  - 可选：展开查看每个窗口详细结果
  - 在模型列表中区分 Walk-Forward 验证的模型
  - **验证**: 结果展示完整、易读

- [ ] **Task 4.4**: 数据不足警告
  - 检测数据长度不足 60 天时显示警告
  - 提示用户增加历史数据
  - 允许用户选择继续训练或取消
  - **验证**: 警告提示清晰、友好

### Phase 5: 错误处理与边界情况

- [ ] **Task 5.1**: 数据长度计算健壮性
  - 处理 MultiIndex 和单层 Index 的时间序列
  - 计算实际天数（考虑交易日 vs 自然日）
  - 边界情况：空数据、单条数据
  - **验证**: 各种数据格式都能正确计算长度

- [ ] **Task 5.2**: 窗口失败处理
  - 单个窗口训练失败不影响其他窗口
  - 记录失败窗口的错误信息
  - 最终结果中标注失败的窗口
  - **验证**: 部分窗口失败时流程继续

- [ ] **Task 5.3**: 动态调整边界测试
  - 测试刚好 60 天数据（最小窗口）
  - 测试 59 天数据（降级到单次训练）
  - 测试 300 天数据（标准窗口）
  - **验证**: 各种边界都能正确处理

### Phase 6: 测试与文档

- [ ] **Task 6.1**: 单元测试
  - 测试 `adjust_walk_forward_windows()` 返回正确参数
  - 测试 `WalkForwardValidator.validate()` 返回格式
  - 测试聚合指标计算正确性
  - **验证**: 所有测试通过

- [ ] **Task 6.2**: 集成测试
  - 测试不同数据长度下的行为：
    - 400 天 → 标准窗口 (180, 15, 15)
    - 150 天 → 较小窗口 (90, 10, 10)
    - 70 天 → 最小窗口 (60, 7, 7)
    - 30 天 → 单次训练 + 警告
  - **验证**: 端到端流程正常

- [ ] **Task 6.3**: 手动测试
  - 使用小数据集快速验证流程（1个月数据）
  - 使用完整数据集验证性能（1年+数据）
  - 前端展示手动验证
  - **验证**: 用户体验符合预期

- [ ] **Task 6.4**: 更新文档
  - 移除 `qlib_ml_config.yaml` 中的 `validation` 段
  - 添加 Walk-Forward 验证说明到 README
  - 更新模型训练文档，说明自动行为
  - **验证**: 文档清晰、准确

## Task Dependencies

```
Phase 1 (动态窗口调整)
    ↓
Phase 2 (验证集成) ← 需要窗口参数
    ↓
Phase 3 (进度展示) ← 需要验证流程运行
    ↓
Phase 4 (前端 UI) ← 需要进度数据结构
    ↓
Phase 5 (错误处理) ← 可并行，依赖于核心流程
    ↓
Phase 6 (测试文档) ← 最后执行
```

## Parallelizable Work

- **可并行**:
  - Phase 4 (前端 UI) 可以在 Phase 2 完成后开始
  - Phase 5 (动态调整边界测试) 可以在 Phase 1 完成后进行
  - Phase 6.4 (文档更新) 可以随时开始（移除配置文件）

- **需串行**:
  - Phase 1 → Phase 2 → Phase 3 (核心流程依赖)
  - Phase 6 (测试) 必须在所有开发任务完成后

## Estimated Effort

- Phase 1: 2-3 小时（动态调整逻辑）
- Phase 2: 4-6 小时（核心验证逻辑）
- Phase 3: 2-3 小时（进度更新）
- Phase 4: 3-4 小时（前端展示）
- Phase 5: 2-3 小时（错误处理）
- Phase 6: 2-3 小时（测试文档）

**总计**: 15-22 小时

## Key Simplifications (vs 原提案)

- ❌ 不需要配置文件读取逻辑
- ❌ 不需要用户配置 validation 参数
- ✅ 代码中写死默认参数
- ✅ 自动根据数据长度调整
- ✅ 简化用户决策
