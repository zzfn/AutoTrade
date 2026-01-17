# AutoTrade 工作流指南

本文档总结了从修改代码、重新训练模型到启动回测和实盘交易的完整步骤。

## 1. 代码修改与准备

在修改了特征工程（`autotrade/research/features`）、模型逻辑（`autotrade/research/models`）或策略代码（`autotrade/execution/strategies`）后，建议先进行基础检查：

```bash
# 自动格式化代码
make format

# 运行代码规范检查
make lint
```

## 2. 重新训练模型 (ML 策略)

当逻辑变更涉及模型预测时，需要重新训练模型。

### 命令行操作

如果你想通过终端精确控制训练参数：

1. **更新/初始化数据**（可选，若需要最新数据）：

   ```bash
   uv run python scripts/init_qlib_data.py --symbols SPY,AAPL,MSFT --days 730
   ```

2. **执行训练并设置为当前模型**：
   ```bash
   # --set-current 会自动将新模型设为 TradeManager 的默认模型
   uv run python scripts/train_model.py --symbols SPY,AAPL,MSFT --set-current
   ```

### Web UI 操作

也可以在 Web 界面快速重构与管理：

1. 启动服务：`make run`。
2. 访问 `http://localhost:8000/models`。
3. **数据管理**：在 「数据中心」 板块，可以点击 **「同步市场数据」** 获取最新的历史行情。
4. **模型管理**：
   - 点击 **「滚动训练」** 按钮，系统将自动以后台任务形式完成数据抓取与重训。
   - 在模型列表中，点击垃圾桶图标或详情弹窗中的 **「删除模型」** 按钮可以清理不再需要的旧模型（当前选中的模型不可删除）。

## 3. 运行回测 (Backtesting)

在投入实盘之前，必须验证策略在历史数据上的表现。

1. **启动 Web 服务器**：
   ```bash
   make run
   ```
2. **进入回测页面**：
   导航至 `http://localhost:8000/backtest`。
3. **配置参数**：
   - 选择策略类型（如 `ML 策略`）。
   - 输入股票池（如 `AAPL,MSFT,GOOGL`）。
   - 设置时间范围和 K 线频率（1d 或 1h）。
4. **分析结果**：
   - 运行结束后，页面会展示由 `LumiBot` 生成的 **Tearsheet (性能报表)**。
   - 详细日志和 HTML 报告保存在 `logs/` 文件夹下。

## 4. 启动交易 (Live/Paper Trading)

### 配置策略

在 `http://localhost:8000/models` 页面：

- 确认 **策略类型** 设置正确（动量策略 或 ML 策略）。
- 确认选中了正确的 **模型文件**。

### 启动运行

根据需求选择模拟盘或实盘：

- **启动模拟盘 (Paper Trading)**：
  ```bash
   make paper
  ```
- **启动实盘 (Live Trading)**：
  ```bash
  # 会有二次确认提示，请谨慎操作
  make live
  ```

## 5. 实时监控

系统启动后，通过 `http://localhost:8000` 实时查看：

- **Dashboard**: 当前净值、持仓盈亏、账户余额。
- **Orders**: 策略触发的最新订单及其执行状态。
- **Logs**: 系统的运行日志和交易逻辑触发信息。

---

_注意：在实盘运行前，请务必检查 `.env` 文件中的 API 密钥及权限设置。_
