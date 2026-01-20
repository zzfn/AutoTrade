# Tasks

## Refactoring

- [x] Create new directory structure (`core`, `ml`, `strategies`, `web`) <!-- id: 0 -->
- [x] Move `web_server.py` to `web/server.py` and update imports <!-- id: 1 -->
- [x] Move `strategies/*.py` to `strategies/` <!-- id: 2 -->
- [x] Move `research/models` to `ml/` <!-- id: 3 -->
- [x] Consolidate shared logic into `core/` <!-- id: 4 -->

## Qlib Integration

- [x] Implement `ml/inference.py`: Class to load Qlib model and accept DataFrame input <!-- id: 5 -->
- [x] Implement `data/qlib_adapter.py`: Function to transform Lumibot OHLCV DataFrame into Qlib-compatible Feature Tensor <!-- id: 6 -->
- [x] Update `strategies/qlib_strat.py` to use the new `ml` and `data` modules instead of old `research` imports <!-- id: 7 -->

## Persistence

- [x] Add `DB_CONNECTION_STR` parsing in `core/config.py` <!-- id: 8 -->
- [x] Verify `self.vars` implementation in `strategies/base.py` (or ensure `qlib_strat.py` inherits correctly) <!-- id: 9 -->

## Validation

- [x] Run `main.py` to check server startup <!-- id: 10 -->
- [x] Run a backtest in "dry run" mode to verify data pipeline throws no errors <!-- id: 11 -->
