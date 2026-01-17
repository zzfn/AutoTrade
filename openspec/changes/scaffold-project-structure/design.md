# Design: Research & Execution Separation

## Architecture Overview

The project adapts a standard "Alpha Factory" vs "Execution Engine" separation.

### 1. Module Responsibilities

We will implement the User's proposed 3-layer definition within the `autotrade` package.

| Module | Scope | Technology | Responsibility |
|/---|---|---|---|
| `autotrade.research` | Offline | Qlib | Factor computation, Model training, Vectorized Backtesting (fast), Workflow management. |
| `autotrade.execution` | Online/Live | LumiBot | Signal consumption, Event-driven Backtesting (accurate), Order routing, Risk Checks. |
| `autotrade.shared` | Common | - | Logging, Configuration (Hydra/Omegaconf), Data Schemas, DB Connectors. |

### 2. Workflow Integration (Qlib + LumiBot)

**This is the industry standard "Alpha Factory" pattern.**

The separation is critical because Qlib and LumiBot have different "views" of time:

- **Qlib (Research)**: Views data as a large matrix (Time x Stocks). Optimized for batch processing, training, and vectorized backtesting.
- **LumiBot (Execution)**: Views data as a stream (Tick by Tick). Optimized for event handling, order management, and risk control.

**The Best Practice Bridge:**

1.  **Training (Research)**: Qlib outputs a **Model Artifact** (e.g., `.pkl`, `.onnx`) and a **Feature Config**.
2.  **Inference (Execution)**: The LumiBot Strategy initializes an `InferenceEngine` (located in `shared` or `execution/utils`) that connects the live data stream to the Qlib Model.

**Why this structure works:**

- You can change the broker (LumiBot) without touching the model code.
- You can retrain the model (Qlib) without restarting the trading engine (just reload the artifact).
- It prevents "Look-ahead Bias" by forcing the LumiBot strategy to step through time, even when using Qlib models.

### 3. Workflow Integration

The flow `Research -> Signals -> Execution` implies a decoupling point.

- **Research** produces **Models** or **Signal Files** (e.g., Target positions/weights for next T).
- **Execution** consumes these artifacts.
- _Crucially_, `execution` code should generally NOT call `research` code directly during a trading loop. It should load a pre-trained model or a signal file.

### 4. Verification of Design

The user asked: _Is this module design reasonable?_

**Yes, it is the best practice for this stack.**

- **Isolation**: Prevents "God objects" that know about both model training (PyTorch/LightGBM) and IBKR API endpoints.
- **Dependency Management**: `research` needs heavy libs (pandas, pytorch, qlib). `execution` needs stability libs (lumibot, ib_insync). They can be kept in separate envs if needed (though here shared in `uv`).
- **Safety**: "Read-only" nature of Research vs "Action-taking" nature of Execution is preserved.

### 5. Implementation Details

**Current Mapping to New Structure:**

- `autotrade/features/` -> `autotrade/research/features/`
- `autotrade/backtests/` -> `autotrade/research/backtest/` (Qlib backtest) / `autotrade/execution/backtest/` (LumiBot backtest)
  - _Note_: We likely have two types of backtests. Qlib is strict vectorized. LumiBot is event driven. We should clarify naming.
- `autotrade/strategies/` -> `autotrade/execution/strategies/`
- `autotrade/brokers/` -> `autotrade/execution/brokers/`
- `autotrade/config/` -> `autotrade/shared/config/`
- `autotrade/utils/` -> `autotrade/shared/utils/`

### 6. Configuration Management

**Requirement**: Trading symbols and other static parameters (timeframes, risk limits) MUST be configurable via YAML.

**Strategy**:

- **Format**: `config.yaml` or specific `instruments.yaml`.
- **Loader**: A configuration utility in `autotrade.shared` (e.g., using `OmegaConf` or Python's `yaml` lib) will load these files at startup.
- **Usage**:
  - **Research**: Qlib reads the YAML to determine the universe of stocks to download/train on.
  - **Execution**: LumiBot reads the SAME YAML to determine which symbols to subscribe to for real-time data.
- **Benefit**: Ensures Research and Execution are always aligned on the "Trading Universe".

## Trade-offs

- **Refactoring Cost**: Moving files requires updating all imports.
- **Naming Conflicts**: We might have `research.backtest` and `execution.backtest`. Clear naming is required.
