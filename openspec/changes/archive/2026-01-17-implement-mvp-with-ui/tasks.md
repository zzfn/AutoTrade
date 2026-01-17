# 任务: 实现 MVP 及 Web UI

## 后端与策略基础

- [x] 安装依赖: `fastapi`, `uvicorn`, `websockets`, `jinja2`, `python-multipart` (添加到 `pyproject.toml`)。 <!-- id: 0 -->
- [x] 创建 `autotrade/web_server.py`: FastAPI 应用设置，配置 Jinja2 模板和静态文件挂载。 <!-- id: 1 -->
- [x] 实现 `autotrade/strategies/simple_strategy.py`: 使用 LumiBot 实现基础 SMA 交叉策略。 <!-- id: 2 -->
- [x] 创建 `autotrade/trade_manager.py`: 单例模式管理策略线程和状态（订单、日志）。 <!-- id: 3 -->
- [x] 更新 `autotrade/web_server.py`: 添加 API 接口用于开始/停止策略和获取状态。 <!-- id: 4 -->

## 前端界面 (React + Tailwind via CDN)

- [x] 创建公共布局 `autotrade/templates/layout.html`: 引入 React, ReactDOM, Babel, Tailwind CDN 脚本。 <!-- id: 5 -->
- [x] 实现 React 组件基础: 在 `autotrade/static/utils.js` 中创建通用的 UI 组件（如 Card, Button）。 <!-- id: 6 -->
- [x] 实现仪表盘 `autotrade/templates/index.html`: 编写 React 组件 `Dashboard`，通过 WebSocket 获取状态并在界面渲染。 <!-- id: 7 -->
- [x] 实现回测页 `autotrade/templates/backtest.html`: 编写 React 组件 `Backtest`，包含表单和 Chart.js 图表组件。 <!-- id: 8 -->
- [x] 样式优化: 确保所有组件使用 Tailwind 类实现深色玻璃拟态风格。 <!-- id: 9 -->

## 集成与验证

- [x] 更新 `main.py`: 作为 CLI 入口点，提供启动 Web 服务器的命令。 <!-- id: 10 -->
- [x] 更新 `Makefile`: 添加 `run` 目标以启动 FastAPI 服务。 <!-- id: 11 -->
- [x] 验证: 通过 UI 运行回测并检查结果显示。 <!-- id: 12 -->
- [x] 验证: 通过 UI 运行模拟盘（Mock 或 Paper）并检查实时更新。 <!-- id: 13 -->
