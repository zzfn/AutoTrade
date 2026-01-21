# Tasks

- [x] Refactor `server.py` to remove `lifespan` strategy auto-start. <!-- id: 0 -->
- [x] Create `start_background_server` function in `server.py`. <!-- id: 1 -->
- [x] Implement `run_strategy_main` logic (or put it in `if __name__ == "__main__"`) in `server.py` that runs the strategy in the main thread. <!-- id: 2 -->
- [x] Verify `api_start_strategy` and `api_stop_strategy` behavior (disable or adapt if necessary, as main thread is blocked). <!-- id: 3 -->
- [x] Ensure `uvicorn` is run with `install_signal_handlers=False`. <!-- id: 4 -->
