# Design: UI-Strategy Separation

## Current Flow

1. User runs `uvicorn autotrade.web.server:app`.
2. Uvicorn starts Main Loop (Main Thread).
3. FastAPI `lifespan` triggers `initialize_and_start`.
4. `initialize_and_start` spawns `strategy_thread`.
5. Strategy runs in background.

## Proposed Flow

1. User runs `python autotrade/web/server.py` (or similar entry point).
2. Script starts `UI Thread`.
   - Runs `uvicorn.run(app, host=..., port=..., install_signal_handlers=False)`.
3. Script initializes Strategy.
4. Script runs Strategy in `Main Thread` (blocking).
5. On exit, cleanup both.

## Component Changes

### `autotrade/web/server.py`

- **Remove**: `lifespan` context manager that auto-starts strategy.
- **Remove**: `initialize_and_start` usage from startup.
- **Add**: `def start_server_background()` helper.
- **Modify**: `if __name__ == "__main__":` block to implement the new flow.
- **Keep**: API endpoints that query `state`.
- **Check**: API endpoints that _control_ strategy (Start/Stop).
  - `api_start_strategy` / `api_stop_strategy`: These currently spawn threads.
  - If the user explicitly wants "Strategy runs on Main Thread", then **automatic** start via API becomes impossible if the main thread is already occupied or if we strictly require main thread execution.
  - **Resolution**: FOR NOW, we will prioritize the initial startup flow. If the user wants to restart the strategy _after_ it stops, it might be complex if it requires the main thread.
  - _Assumption_: The primary use case is "Run script -> Strategy runs". Re-running might not be needed or can be handled via restart.
  - However, the user said "FastAPI server as UI display, do nothing else". This might imply removing control endpoints or making them no-ops if they violate the threading model.
  - We will keep the endpoints but they might need to be adjusted to purely update status or ignored if the strategy is running in main.
  - Actually, if the strategy runs in the main thread, `trader.run_all()` blocks. Once it returns, the script ends. So "starting" it again via API is not applicable in the same run unless we wrap it in a loop.
  - We will focus on the startup requirement first.

### Data Flow

- **Strategy** (Main Thread) -> `update_status()`, `update_portfolio()` -> `state` (Global).
- **UI** (Bg Thread) -> `GET /api` -> reads `state`.

## Threading Model

- **Main**: Strategy Logic (Critical path).
- **Background**: UI/Network (Auxiliary).
