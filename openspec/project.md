# Project Context

## Purpose

AutoTrade 是一个基于 **LumiBot** 的量化交易自动化平台，专注于：

- 使用 LumiBot 框架开发和回测交易策略
- 对接 **Alpaca** 券商进行美股模拟盘交易
- 提供可复用的策略模板和风险管理工具

## Tech Stack

### 核心框架

- **Python 3.11+** - 主要开发语言
- **uv** - 包管理与虚拟环境管理
- **LumiBot** - 回测与算法交易框架（核心）
- **Alpaca-py** - Alpaca 券商 API

### 数据与分析

- **YFinance** - 历史数据获取（回测用）
- **TA-Lib** - 技术分析指标
- **Pandas** - 数据处理
- **NumPy** - 数值计算

### 日志

- **Loguru** - 日志记录

### 开发工具

- **Ruff** - 代码检查和格式化

## Project Conventions

### Code Style

- 使用 **Ruff** 进行代码检查和格式化
- 行长度限制：88 字符
- 引号风格：双引号
- Import 排序：使用 isort（通过 Ruff）

### 命名规范

- 策略类名：`PascalCase`，以 `Strategy` 结尾（如 `BuyAndHoldStrategy`）
- 模块名：`snake_case`
- 函数/变量：`snake_case`
- 常量：`UPPER_SNAKE_CASE`

### 策略开发规范

1. 所有策略必须继承 LumiBot 的 `Strategy`
2. 实现 `on_trading_iteration()` 方法
3. 使用 `parameters` 字典定义可配置参数
4. 先回测验证，再模拟盘测试，最后实盘

### Architecture Patterns

- **策略模式**：每个交易策略是独立的类
- **配置驱动**：API 密钥和参数通过环境变量和参数字典管理
- **生命周期钩子**：
  - `initialize()` - 初始化
  - `before_market_opens()` - 开盘前
  - `on_trading_iteration()` - 交易迭代（核心逻辑）
  - `after_market_closes()` - 收盘后

### Testing Strategy

- 策略必须通过历史回测验证
- 模拟盘运行验证后才能上实盘

### Git Workflow

- 主分支：`main`
- 功能分支：`feature/<description>`
- Commit 格式：`<type>: <description>`
  - `feat`, `fix`, `docs`, `refactor`, `test`

## Domain Context

### LumiBot 核心概念

- **Strategy**: 策略类，包含交易逻辑
- **Broker**: 券商接口（Alpaca）
- **Trader**: 策略运行器
- **Order**: 订单对象
- **Position**: 持仓对象

### LumiBot 常用方法

```python
# 获取价格
price = self.get_last_price("AAPL")

# 获取投资组合信息
self.portfolio_value  # 总价值
self.cash             # 现金

# 创建和提交订单
order = self.create_order("AAPL", 10, "buy")
self.submit_order(order)

# 获取持仓
position = self.get_position("AAPL")
positions = self.get_positions()
```

### 美股交易时间

- 常规交易：9:30 AM - 4:00 PM (ET)
- 盘前：4:00 AM - 9:30 AM (ET)
- 盘后：4:00 PM - 8:00 PM (ET)

## Important Constraints

### 技术约束

- Python 代码必须使用 `uv run python` 执行
- API 密钥存储在 `.env` 文件，禁止提交到 Git
- 所有策略必须先通过回测

### 业务约束

- 实盘前必须通过模拟盘验证
- 必须设置止损机制
- PDT 规则：25,000 美元以下账户每周限制 3 次日内交易

### Alpaca 限制

- API 速率限制：200 请求/分钟
- 仅支持美股市场
- Paper 账户和 Live 账户使用不同的 API 端点

## External Dependencies

### Alpaca Markets

- **官网**: https://alpaca.markets/
- **API 文档**: https://docs.alpaca.markets/
- 需要注册账户获取 API Key 和 Secret
- 建议先使用 Paper Trading 账户测试

### LumiBot

- **文档**: https://lumibot.lumiwealth.com/
- **GitHub**: https://github.com/Lumiwealth/lumibot
- **示例策略**: https://github.com/Lumiwealth/lumibot/tree/dev/lumibot/example_strategies
