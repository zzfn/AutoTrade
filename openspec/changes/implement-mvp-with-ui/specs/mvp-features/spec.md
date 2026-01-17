# 规格: MVP 功能特性

## ADDED Requirements

### Requirement: 简单移动平均线策略

系统必须包含一个内置策略，基于简单移动平均线 (SMA) 交叉执行交易。(MUST)

#### Scenario: 策略买入信号

给定 短期 SMA (如 10 天) 上穿 长期 SMA (如 30 天)
当 策略处理交易迭代时
那么 应该针对目标代码生成买入订单。

#### Scenario: 策略卖出信号

给定 短期 SMA 下穿 长期 SMA
当 策略处理交易迭代时
那么 应该生成卖出订单（或清仓）。

### Requirement: 实时 API 与服务

后端必须通过 FastAPI 提供 REST API 和 WebSocket 服务以暴露交易状态。(MUST)

#### Scenario: 获取状态

给定 策略正在运行
当 客户端请求 `GET /api/status`
那么 系统返回包含 `status: "running"`, `cash` (现金), `portfolio_value` (组合价值) 和 `last_updated` (最后更新时间) 的 JSON。

#### Scenario: 实时更新

给定 策略下了一个新订单
当 订单提交给 Alpaca 后
那么 该事件应在 1 秒内通过 WebSocket 推送给已连接的客户端。

### Requirement: Web 仪表盘

系统必须直接提供 HTML 页面作为仪表盘，用于查看实时交易活动。(MUST)

#### Scenario: 查看活跃订单

给定 系统中有活跃订单
当 用户访问首页 (`/`)
那么 页面显示的表格中应列出订单详情（代码、方向、数量、状态）。

#### Scenario: 查看组合价值

给定 策略正在执行
当 用户访问首页
那么 页面应醒目地显示当前组合总价值和现金余额。

### Requirement: 回测 UI

系统必须提供一个界面来配置和运行回测。(MUST)

#### Scenario: 运行回测

给定 用户在回测页选择了日期范围（开始、结束）和代码（如 "SPY"）
当 用户点击“运行回测”按钮
那么 后端执行回测
并且 在前端通过图表展示绩效指标（CAGR, 最大回撤）和权益曲线。
