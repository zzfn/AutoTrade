# Refactor Core Components and Remove Monkey Patches

## Summary

Refactor the application to remove unstable monkey patches and simplify the architecture by integrating `TradeManager` logic directly into `server.py`, removing the separate `autotrade/trade_manager.py` file.

## Motivation

- **Stability**: Monkey patches (`signal.signal`, `Alpaca.process_pending_orders`) are fragile and hide underlying issues or workarounds that should be handled cleaner or are no longer needed.
- **Simplification**: `TradeManager` adds an extra layer of abstraction. Integrating it into `server.py` (the interface) centralizes the control logic, making the system state easier to manage within the web context.

## Proposed Changes

1.  **Remove Monkey Patches**: Delete the monkey patching code in `autotrade/web/server.py`.
2.  **Integrate Logic**: Move all functionality from `autotrade/trade_manager.py` (strategy execution, monitoring, state management, ML integration) into `autotrade/web/server.py`.
3.  **Delete File**: Remove `autotrade/trade_manager.py`.
