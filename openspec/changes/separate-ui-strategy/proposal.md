# Separate UI and Strategy Execution

## Context

Currently, `autotrade/web/server.py` acts as the primary entry point, using FastAPI's lifespan events to spawn the trading strategy in a background thread. The user wants to invert this control: the Strategy should run in the Main Thread (for stability and signal handling), and the UI (FastAPI) should run in a background thread purely for display and monitoring.

## Goals

1. Refactor `autotrade/web/server.py` to remove strategy management from the FastAPI `lifespan`.
2. Implement a mechanism to start the FastAPI server in a background daemon thread.
3. Ensure the main execution flow runs the Trading Strategy in the main thread.
4. Maintain the existing shared state mechanism so the UI can still display strategy status.

## Architecture

- **Thread 1 (Background)**: FastAPI + Uvicorn (with `install_signal_handlers=False`). Serves UI and API.
- **Thread 0 (Main)**: LumiBot execution `trader.run_all()`.
- **Shared State**: The global `state` dictionary in `server.py` continues to serve as the bridge. The Strategy (running in Main) updates this stricture; The Server (running in BG) reads it.

## Risks

- **Uvicorn in Thread**: Uvicorn is typically designed to run in the main thread. We must ensure it is configured correctly (`install_signal_handlers=False`) to avoid interfering with the main thread.
- **Shutdown**: We need to ensure that when the Strategy finishes or crashes, the UI server is also shut down (or vice versa).
