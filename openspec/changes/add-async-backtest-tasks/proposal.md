# Change: Add async backtest task execution

## Why
目前回测在独立进程内运行时，Web 进程无法获取实时状态与日志，导致前端一直停留在“回测开始”。需要一个可恢复、可追踪、可推送的回测执行机制。

## What Changes
- 引入异步回测任务：回测由独立 worker 执行，Web 进程只提交任务并查询状态
- 回测状态与日志持久化（便于刷新/断线后继续展示）
- WebSocket/SSE 推送回测状态与进度，前端显示实时状态与完成结果

## Impact
- Affected specs: new `backtest-execution` capability
- Affected code: Web API、WebSocket 推送、回测执行入口、任务执行与状态存储
