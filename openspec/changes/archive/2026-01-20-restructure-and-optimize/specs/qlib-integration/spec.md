# Qlib In-Memory Integration

## ADDED Requirements

### Requirement: Live Feature Calculation

The system MUST support calculating Qlib-compatible features on-the-fly from live market data without disk I/O.

#### Scenario: Live Data Feeding

- **Given** a generic Pandas DataFrame provided by Lumibot's `get_historical_prices`
- **When** `ml.inference.predict(df)` is called
- **Then** the system must calculate factors (e.g., RSI, MACD, Qlib alpha158) in memory
- **And** return a prediction signal without writing to disk or DB

### Requirement: Training Parity

Inference-time feature calculation MUST be mathematically identical to training-time feature calculation to ensure model validity.

#### Scenario: Feature Parity

- **Given** a model trained on Qlib's standard dataset
- **When** performing inference on live data
- **Then** the calculated features must mathematically match Qlib's training features (handling NaN, normalization)

### Requirement: Inference Efficiency

Loading models and performing inference MUST be optimized for real-time trading iterations.

#### Scenario: Model Loading

- **Given** a saved Qlib model checkpoint in `resources/models/`
- **When** the Strategy initializes
- **Then** it must load the model into memory once and reuse it for each iteration
