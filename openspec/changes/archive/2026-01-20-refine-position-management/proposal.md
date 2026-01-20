# Refine Position Management

## Goal

Improve portfolio rebalancing robustness and introduce dynamic position sizing capabilities to `QlibMLStrategy`.

## Context

The current implementation of `QlibMLStrategy` uses a fixed equal-weight allocation and rebalances based only on the defined `symbols` candidate list. This has two limitations:

1. **Orphaned Positions**: If a held position is removed from the `symbols` list (or manually acquired), it is ignored during rebalancing and never sold, potentially leading to unintended exposure ("ABC ignored" issue).
2. **Static Sizing**: Capital is allocated equally among Top-K stocks regardless of prediction confidence, limiting the strategy's ability to capitalize on high-conviction signals.

## Solution

1. **Robust Rebalancing**: Modify `_rebalance_portfolio` to iterate over _all_ currently held positions (via `self.get_positions()`) to ensure no position is orphaned. Any position not in the new target list will be sold.
2. **Dynamic Sizing**: Introduce a `position_sizing` parameter with support for:
   - `equal`: Current behavior (default).
   - `weighted`: Allocate capital proportional to prediction scores (normalized).

## Risks

- **Selling Manual Positions**: If the user manually buys a stock that is not in the strategy's target list, the strategy will sell it on the next iteration. This is standard for a fully automated strategy but might surprise users doing hybrid trading.
