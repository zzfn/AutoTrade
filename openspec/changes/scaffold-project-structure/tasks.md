# Tasks: Refine Project Structure

## Group: Scaffolding

- [x] Create `autotrade/research/__init__.py` <!-- id: create-research-pkg -->
- [x] Create `autotrade/execution/__init__.py` <!-- id: create-execution-pkg -->
- [x] Create `autotrade/shared/__init__.py` <!-- id: create-shared-pkg -->
- [x] Create `autotrade/shared/config` directory <!-- id: create-config-dir -->

## Group: Configuration Implementation

- [x] Create initial `configs/universe.yaml` example <!-- id: create-yaml-example -->
- [x] Implement YAML loader in `autotrade/shared/config/loader.py` <!-- id: impl-yaml-loader -->
- [x] Ensure loader returns typed objects or Dict for symbols <!-- id: ensure-loader-types -->

## Group: Migration - Shared

- [x] Move `autotrade/config` python files to `autotrade/shared/config` <!-- id: move-config -->
- [x] Move `autotrade/utils` to `autotrade/shared/utils` <!-- id: move-utils -->
- [ ] Refactor imports in `shared` modules <!-- id: refactor-shared-imports -->

## Group: Migration - Research

- [x] Move `autotrade/features` to `autotrade/research/features` <!-- id: move-features -->
- [x] Move `autotrade/backtests` to `autotrade/research/backtest` (assuming Qlib based) <!-- id: move-backtests -->
- [ ] Refactor imports in `research` modules <!-- id: refactor-research-imports -->

## Group: Migration - Execution

- [x] Move `autotrade/strategies` to `autotrade/execution/strategies` <!-- id: move-strategies -->
- [x] Move `autotrade/brokers` to `autotrade/execution/brokers` <!-- id: move-brokers -->
- [ ] Refactor imports in `execution` modules <!-- id: refactor-execution-imports -->

## Group: Cleanup

- [x] Remove empty top-level directories (`features`, `strategies`, etc) <!-- id: cleanup-dirs -->
- [x] Update `main.py` or entrypoints to reflect new paths <!-- id: update-entrypoints -->
