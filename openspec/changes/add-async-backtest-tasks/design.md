## Context
回测属于重计算任务，直接在 Web 进程或简单多进程中执行会导致状态丢失和前端无法持续获取进度。需要将回测执行与 Web 进程解耦，并提供可靠的状态与日志读取。

## Goals / Non-Goals
- Goals:
  - 回测任务可异步执行，Web 进程不阻塞
  - 回测状态、日志、结果可持久化并可被前端实时展示
  - 任务具备可追踪性（任务 ID）与可恢复性（断线重连后仍可展示）
- Non-Goals:
  - 分布式大规模调度（仅需单机 worker）
  - 引入复杂的多租户或权限系统

## Decisions
- 采用任务队列 + worker 模式执行回测
- 使用持久化存储保存回测状态、日志片段与结果元数据
- WebSocket/SSE 从持久化存储拉取并推送状态更新

## Alternatives considered
- 继续用线程/进程执行并共享内存
  - 优点：实现简单
  - 缺点：状态不可持久化，断线后丢失，进程边界不共享
- 使用 FastAPI BackgroundTasks
  - 优点：无需额外依赖
  - 缺点：进程内执行，服务重启即中断，状态不可恢复

## Risks / Trade-offs
- 引入队列与持久化存储带来部署成本
  - Mitigation: 选用轻量级方案（如 Redis + RQ），并提供最小可运行配置

## Migration Plan
- 新增任务模型与状态存储
- 新增回测提交与查询接口（返回任务 ID）
- 前端改为轮询/订阅任务状态
- 保留旧接口一段时间并标记弃用（如有需要）

## Open Questions
- 选择 Celery/RQ/Arq 中哪一个作为任务队列（优先轻量）
- 状态持久化使用 Redis 还是 SQLite（Redis 更适合短期日志，SQLite 便于本地开发）
