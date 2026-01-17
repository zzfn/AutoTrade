# Proposal: Refine Project Structure for Research/Execution Separation

## Summary

Restructure the `autotrade` source package to explicitly separate identifying capabilities (Research/Qlib) from trade execution capabilities (Execution/LumiBot), using a `shared` module for common infrastructure.

## Motivation

The current flat structure of `autotrade/` mixes research components (features, backtests) with execution components (brokers, strategies). This creates:

1.  **Dependency coupling**: Hard to run execution without heavy research dependencies.
2.  **Logic bleeding**: Risk of using future data or research shortcuts in live execution code.
3.  **Mental model mismatch**: The workflow clearly separates "Offline Research" from "Online Execution".

## Proposed Changes

Refactor `autotrade/` into three sub-packages:

- `research/`: Factor mining, Qlib workflows, model training.
- `execution/`: LumiBot strategies, live trading loops, broker adapters.
- `shared/`: Configuration, common utils, data interfaces.

## Verification

- Verify that `research` imports `shared` but NOT `execution`.
- Verify that `execution` imports `shared` but NOT `research` (except perhaps for signal loading, which should be done via artifacts, not code import).
- Existing tests pass after refactor.
