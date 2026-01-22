# Tasks: add-walk-forward-validation

## Implementation Tasks

### Phase 1: 固定参数配置

- [x] **Task 1.1**: 定义固定参数常量
  - 在 `autotrade/web/server.py` 顶部定义 `WALK_FORWARD_CONFIG`
  - 参数：`train_window=2000`, `test_window=200`, `step_size=200`（单位：根K线）
  - 定义 `MIN_BARS_REQUIRED = 2500`
  - **验证**: 常量定义正确，可访问

- [x] **Task 1.2**: 集成到训练流程
  - 在 `start_model_training_internal()` 中检查数据长度
  - 分支逻辑：`len(X) >= MIN_BARS_REQUIRED` → Walk-Forward，否则 → 单次训练
  - 数据不足时记录警告日志
  - **验证**: 数据充足时走 Walk-Forward，不足时降级

### Phase 2: Walk-Forward 验证集成

- [x] **Task 2.1**: 实现 Walk-Forward 验证主流程
  - 在 `start_model_training_internal()` 中添加验证分支
  - 初始化 `WalkForwardValidator`（传入固定参数）
  - 调用 `validator.validate(X, y)` 执行验证
  - **验证**: 验证流程能正常运行无错误

- [x] **Task 2.2**: 实现结果聚合逻辑
  - 收集所有窗口的评估指标（IC, ICIR, etc.）
  - 计算聚合统计量：`ic_mean`, `ic_std`, `icir_mean`
  - 格式化为前端可用的结果结构
  - **验证**: 聚合指标计算正确（与手动计算对比）

- [x] **Task 2.3**: 模型保存逻辑调整
  - Walk-Forward 模式下：使用全部数据训练最终模型
  - 更新模型元数据，包含 Walk-Forward 结果（聚合指标 + 配置）
  - **验证**: 模型能正确保存和加载

### Phase 3: 进度展示

- [x] **Task 3.1**: 实现细粒度进度更新
  - 在 Walk-Forward 验证循环中更新 `training_status`
  - 计算进度百分比：`50 + 50 * (current_window / total_windows)`
  - 更新消息：`"验证窗口 {i}/{total}..."`
  - **验证**: 前端能实时看到进度更新

- [x] **Task 3.2**: 扩展训练状态结构
  - 添加 `walk_forward_progress` 字段（可选）
  - 包含：`current_window`, `total_windows`, `window_results`
  - 前端可选择性展示详细窗口信息
  - **验证**: 状态结构包含所有必要信息

### Phase 4: 前端 UI 增强

- [x] **Task 4.1**: 更新训练配置展示
  - 在"训练配置"区域显示 Walk-Forward 自动启用
  - 显示固定窗口参数（2000/200/200 根K线）
  - 计算并显示预计窗口数
  - **验证**: 用户能清楚看到配置

- [x] **Task 4.2**: 增强 Walk-Forward 进度展示
  - 显示当前窗口编号（如：窗口 3/90）
  - 显示当前窗口的 IC、ICIR
  - 显示已完成窗口的累计平均
  - **验证**: 进度展示清晰、实时

- [x] **Task 4.3**: 展示 Walk-Forward 完成结果
  - 显示聚合指标：`IC: 0.041 ± 0.008`, `ICIR: 1.18 ± 0.15`
  - 可选：展开查看每个窗口详细结果
  - 在模型列表中区分 Walk-Forward 验证的模型
  - **验证**: 结果展示完整、易读

- [x] **Task 4.4**: 数据不足警告
  - 检测数据长度不足 2500 根K线时显示警告
  - 提示用户增加 `num_bars`（推荐 20000，最小 5000）
  - 允许用户选择继续训练或取消
  - **验证**: 警告提示清晰、友好

### Phase 5: 错误处理与边界情况

- [x] **Task 5.1**: 窗口失败处理
  - 单个窗口训练失败不影响其他窗口
  - 记录失败窗口的错误信息
  - 最终结果中标注失败的窗口
  - **验证**: 部分窗口失败时流程继续

- [ ] **Task 5.2**: 边界情况测试
  - 测试刚好 2500 根K线（最小窗口）
  - 测试 2499 根K线（降级到单次训练）
  - 测试 20000 根K线（标准配置）
  - **验证**: 各种边界都能正确处理

### Phase 6: 测试与文档

- [ ] **Task 6.1**: 单元测试
  - 测试 `WalkForwardValidator.validate()` 返回格式
  - 测试聚合指标计算正确性
  - **验证**: 所有测试通过

- [ ] **Task 6.2**: 集成测试
  - 测试不同数据长度下的行为：
    - 20000 根K线 → Walk-Forward 验证（~90 个窗口）
    - 5000 根K线 → Walk-Forward 验证（~15 个窗口）
    - 2500 根K线 → Walk-Forward 验证（~2 个窗口）
    - 1000 根K线 → 单次训练 + 警告
  - **验证**: 端到端流程正常

- [ ] **Task 6.3**: 手动测试
  - 使用小数据集快速验证流程（3000 根K线）
  - 使用完整数据集验证性能（20000 根K线）
  - 前端展示手动验证
  - **验证**: 用户体验符合预期

- [x] **Task 6.4**: 更新文档
  - 添加 Walk-Forward 验证说明到 README
  - 更新模型训练文档，说明固定参数配置
  - 说明数据量要求（最小 2500，推荐 20000）
  - **验证**: 文档清晰、准确

## Task Dependencies

```
Phase 1 (固定参数配置)
    ↓
Phase 2 (验证集成) ← 需要参数常量
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
  - Phase 5 (边界测试) 可以在 Phase 1 完成后进行
  - Phase 6.4 (文档更新) 可以随时开始

- **需串行**:
  - Phase 1 → Phase 2 → Phase 3 (核心流程依赖)
  - Phase 6 (测试) 必须在所有开发任务完成后

## Estimated Effort

- Phase 1: 1-2 小时（固定参数定义）
- Phase 2: 3-4 小时（核心验证逻辑）
- Phase 3: 2-3 小时（进度更新）
- Phase 4: 3-4 小时（前端展示）
- Phase 5: 1-2 小时（错误处理）
- Phase 6: 2-3 小时（测试文档）

**总计**: 12-18 小时（比原方案减少约 4 小时）

## Key Simplifications (vs 原提案)

- ❌ 移除动态窗口调整逻辑
- ❌ 移除基于"天数"的计算
- ✅ 固定参数（2000/200/200 根K线）
- ✅ 简化为数据充足性检查
- ✅ 统一使用"根K线数量"作为单位
