# Restructure and Optimize Core Data Flow

## Goal

Restructure the project for clarity, implement a high-performance in-memory data pipeline for Qlib (No DB required for live data), and ensure robust trading state persistence.

## Problem

- **Project Structure**: Current structure splits `execution` and `research` but may lack shared access to model utilities.
- **Data Latency/Complexity**: Storing high-frequency live data to DB before feeding Qlib is inefficient and unnecessary.
- **Reliability**: Trading state (positions, stops, cycle counts) risks loss on restart.
- **Model Management**: Need a clear strategy for storing and loading trained Qlib models.

## Solution

1. **Refactor Directory**: Flatten and simplify into `strategies`, `ml` (shared), `data` (adapters).
2. **In-Memory Pipeline**: Adapt Lumibot's `get_historical_prices` output directly into Qlib's expected format on the fly.
3. **State Persistence**: Enable Lumibot's built-in `self.vars` backup with SQLite/Postgres.
4. **Model Storage**: Define `models/` for versioned checkpoints.

## Risks

- **Qlib Compatibility**: Qlib usually expects disk-based binary datasets. We must ensure the `Dataset` class can accept in-memory DataFrames or minimal temporary buffers correctly.
- **Migration**: Moving files requires updating imports in `main.py` and tests.
