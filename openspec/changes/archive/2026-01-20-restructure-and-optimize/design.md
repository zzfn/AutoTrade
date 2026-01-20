# Design: Restructure & Data Pipeline

## 1. Directory Architecture

We will adopt a functional layering approach to separate concerns while keeping related code accessible.

```
autotrade/
├── core/               # Core infrastructure
│   ├── config.py       # Configuration loading
│   ├── logger.py       # Logging setup
│   └── persistence.py  # DB/State connections
├── data/               # Data ingestion & transformation
│   ├── qlib_adapter.py # DataFrame -> Qlib Feature mapping
│   └── providers.py    # Non-standard data sources
├── ml/                 # Machine Learning components
│   ├── definitions/    # Model architectures (LSTM, Transformer, etc.)
│   ├── training/       # Training loops/pipelines
│   └── inference.py    # Inference wrappers (load model -> predict)
├── strategies/         # Trading strategies
│   ├── base.py         # Common enhancements (auto-backup)
│   └── qlib_strat.py   # The specific ML strategy
├── web/                # Web Interface
│   ├── server.py       # FastAPI app
│   └── templates/      # HTML
└── main.py             # Entry point
```

## 2. In-Memory Qlib Pipeline (No DB)

**Current Flow (Traditional):** `Price` -> `DB/File` -> `Qlib Load` -> `Features` -> `Model`
**New Flow (Streamlined):** `Lumibot.on_trading_iteration` -> `Memory(DataFrame)` -> `Qlib Feature Generator` -> `Model`

- **Mechanism**:
  - Use Qlib's `ExpressionEngine` or hardcoded feature calculation on Pandas DataFrames if Qlib's data loader is too rigid for memory-only.
  - _Correction_: If Qlib strictly requires disk binaries, we might need a small ramdisk or temporary directory. However, the user states "Can directly feed". We will assume we can use `qlib.contrib.data.handler.MutilFrameDataHandler` or similar, OR simply bypass Qlib's data layer for _inference_ if the model just needs a tensor.
  - **Decision**: We will implement a `RealtimeFeatureCalculator` that mimics the Qlib training feature generation using standard Pandas/Numpy, ensuring the inputs to the model match exactly without invoking the heavy Qlib disk-based `Dataset` during live inference. Or, use Qlib's `online` mode if supported/configured.

## 3. Storage & State

- **Model Artifacts**:
  - **Single Model**: "deepalaph"
  - Location: `models/deepalaph/`
  - Structure: Can just be the latest file `model.pkl` or timestamped folders if versioning is desired.
  - Format: `.pkl` (Python object) or `.bin` (Torch/LightGBM native).
- **Trading State (`self.vars`)**:
  - Mechanism: Lumibot's auto-backup.
  - Configuration: `DB_CONNECTION_STR` in `.env`.
  - default: `sqlite:///data/trade_state.sqlite`
