## ADDED Requirements
### Requirement: Async Backtest Task Execution
系统 MUST 允许用户提交回测任务并异步执行，返回可追踪的任务 ID。

#### Scenario: Submit backtest task
- **WHEN** 用户在回测 UI 提交回测请求
- **THEN** 后端创建回测任务并返回 `task_id`
- **AND** Web 进程不阻塞

### Requirement: Backtest Task Status Persistence
系统 MUST 持久化回测任务状态、日志片段与结果元数据，以支持断线恢复与历史查询。

#### Scenario: Resume status after reconnect
- **WHEN** 客户端断线后重新连接
- **THEN** 可通过 `task_id` 获取最新状态与已产生的日志

### Requirement: Backtest Status Streaming
系统 MUST 向客户端推送回测任务的状态变化与关键日志片段。

#### Scenario: Stream status updates
- **WHEN** 回测任务进入运行或完成状态
- **THEN** 客户端在 1 秒内收到状态更新推送
