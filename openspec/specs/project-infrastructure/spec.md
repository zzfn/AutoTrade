# project-infrastructure Specification

## Purpose
TBD - created by archiving change scaffold-project-structure. Update Purpose after archive.
## Requirements
### Requirement: Core Package Structure

系统 SHALL 使用 `src/autotrade/` 作为核心 Python 包目录，包含：

- `core/` 模块用于核心功能
- `brokers/` 模块用于券商接口

#### Scenario: Core package exists

- **WHEN** 检查项目结构
- **THEN** `src/autotrade/__init__.py` 存在
- **AND** `src/autotrade/core/__init__.py` 存在
- **AND** `src/autotrade/brokers/__init__.py` 存在

---

### Requirement: Strategy Directory

系统 SHALL 提供独立的 `strategies/` 目录用于存放交易策略：

- 根目录存放用户自定义策略
- `examples/` 子目录存放示例策略
- `ml/` 子目录存放机器学习策略

#### Scenario: Strategy directory exists

- **WHEN** 检查项目结构
- **THEN** `strategies/__init__.py` 存在
- **AND** `strategies/examples/` 目录存在
- **AND** `strategies/ml/` 目录存在

---

### Requirement: ML Models Directory

系统 SHALL 提供 `models/` 目录用于存放训练好的机器学习模型：

- 支持 LightGBM、XGBoost 等模型文件
- `checkpoints/` 子目录存放训练检查点

#### Scenario: Models directory exists

- **WHEN** 检查项目结构
- **THEN** `models/.gitkeep` 存在
- **AND** `models/checkpoints/.gitkeep` 存在

---

### Requirement: Features Directory

系统 SHALL 提供 `features/` 目录用于因子定义和特征工程：

- 支持 Qlib 因子开发
- `alpha/` 子目录存放 alpha 因子

#### Scenario: Features directory exists

- **WHEN** 检查项目结构
- **THEN** `features/__init__.py` 存在
- **AND** `features/alpha/__init__.py` 存在

---

### Requirement: Configuration Module

系统 SHALL 提供 `config/` 模块用于配置管理：

- 集中管理所有配置项
- 支持环境变量覆盖

#### Scenario: Config module exists

- **WHEN** 检查项目结构
- **THEN** `config/__init__.py` 存在
- **AND** `config/settings.py` 存在

---

### Requirement: Utility Module

系统 SHALL 提供 `utils/` 模块用于工具函数：

- 日志配置
- 通用辅助函数

#### Scenario: Utils module exists

- **WHEN** 检查项目结构
- **THEN** `utils/__init__.py` 存在
- **AND** `utils/logger.py` 存在

---

### Requirement: Data and Logs Directories

系统 SHALL 提供独立的数据和日志目录：

- `logs/` 用于日志文件输出
- `data/` 用于数据缓存
- 这些目录的内容 SHALL 被 `.gitignore` 忽略（保留 `.gitkeep`）

#### Scenario: Data and logs directories exist

- **WHEN** 检查项目结构
- **THEN** `logs/.gitkeep` 存在
- **AND** `data/.gitkeep` 存在

---

### Requirement: Test Directory

系统 SHALL 提供 `tests/` 目录用于测试代码：

- 使用 pytest 作为测试框架
- 提供 `conftest.py` 用于共享 fixtures

#### Scenario: Test directory exists

- **WHEN** 检查项目结构
- **THEN** `tests/__init__.py` 存在
- **AND** `tests/conftest.py` 存在

---

### Requirement: Backtest Directory

系统 SHALL 提供 `backtests/` 目录用于回测相关内容：

- `configs/` 子目录存放回测配置
- `results/` 子目录存放回测结果
- 回测结果 SHALL 被 `.gitignore` 忽略

#### Scenario: Backtest directory exists

- **WHEN** 检查项目结构
- **THEN** `backtests/configs/.gitkeep` 存在
- **AND** `backtests/results/.gitkeep` 存在

