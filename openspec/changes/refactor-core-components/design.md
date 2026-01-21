# Design: Refactor Core Components

## Architecture Changes

- **Single Entry Point**: `autotrade/web/server.py` becomes the definitive owner of the trading strategy lifecycle and state.
- **State Management**: The `state` dictionary and related synchronization locks previously in `TradeManager` will now be managed directly within the `server.py` module scope or a simplified internal helper class if needed for scoping, but primarily integrated into the FastAPI `lifespan` and global context.

## Implementation Details

- **Monkey Patches**:
  - `signal.signal`: Will be removed.
  - `Alpaca.process_pending_orders`: Will be removed.
- **TradeManager Integration**:
  - The `TradeManager` class code will be effectively moved to `server.py`.
  - To avoid a single 1500-line file becoming unmanageable, we will organize the code using regions or potential helper functions, but the _file_ `trade_manager.py` will cease to exist.
  - The `tm` global instance in `server.py` will be replaced by direct function calls or a local instance defined within `server.py`.

## Risks

- **Signal Handling**: Removing the `signal` patch might cause `ValueError` if `lumibot` registers signals in threads. We assume this is either fixed or we accept the risk/will debug if it occurs.
- **Alpaca Patch**: Removing the `process_pending_orders` patch might cause crashes on shutdown with `lumibot`.
