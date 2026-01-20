# Add Long-Short Position Support

## Goal

Add comprehensive long-short position support to `QlibMLStrategy` to enable the strategy to handle both long (quantity > 0) and short (quantity < 0) positions correctly during rebalancing.

## Context

The current implementation of `_rebalance_portfolio` in `QlibMLStrategy` only handles long positions:

```python
qty = float(pos.quantity)
if qty > 0:  # Only records long positions
    current_positions[symbol] = qty
```

This limitation has several issues:

1. **Ignores Short Positions**: Short positions (quantity < 0) are completely ignored during rebalancing, leaving open short positions uncovered.
2. **No Short Closing**: When a short position is not in the target list, it remains open instead of being closed (buying to cover).
3. **Incomplete Portfolio Value**: The portfolio value calculation may not properly account for short position market values.
4. **No Short Targeting**: The strategy cannot target short positions even when prediction scores are negative (bearish signals).

## Solution

### 1. Track Both Long and Short Positions

Modify position tracking to record both positive and negative quantities:

```python
current_positions[symbol] = qty  # Record both > 0 and < 0
```

### 2. Handle Position Closing Logic

- **Long Position Close**: Sell (`order_side="sell"`) when `qty > 0` not in target
- **Short Position Close**: Buy (`order_side="buy"`) when `qty < 0` not in target

### 3. Support Negative Target Quantities (Optional Enhancement)

Allow target positions to specify direction:
- Positive target quantity → Long position
- Negative target quantity → Short position
- Zero target quantity → Close position

### 4. Portfolio Value Adjustment

Ensure `portfolio_value` properly accounts for short positions (may already be handled by Lumibot).

## Changes

### Core Logic Changes

**File**: `autotrade/strategies/qlib_strat.py`

1. **Position Tracking** (line ~422):
   - Remove `if qty > 0` condition
   - Track all positions regardless of direction

2. **Closing Positions** (line ~434):
   - Determine close side based on quantity sign
   - Use `"sell"` for long positions, `"buy"` for short positions

3. **Target Adjustment** (line ~469):
   - Handle negative target quantities if supporting short targeting

### Parameters

Consider adding:
- `allow_short` (bool): Whether to allow opening new short positions (default: False for backward compatibility)
- `max_short_ratio` (float): Maximum short exposure as ratio of portfolio value (e.g., 0.3 for 30%)

## Risks

- **Short Selling Risk**: Short positions have unlimited loss potential. Should be disabled by default.
- **Margin Requirements**: Short positions require margin accounts; ensure broker/execution supports this.
- **Borrowing Costs**: Short positions incur borrowing fees that reduce returns.
- **Buy-In Risk**: Short positions may be forcibly closed by the broker due to lending constraints.
- **Complexity**: Long-short rebalancing is more complex and may introduce bugs.

## Open Questions

1. Should negative prediction scores automatically trigger short positions, or should this require explicit `allow_short=True`?
2. How to handle partial closing of short positions (e.g., target is -50, currently -100)?
3. Should we add validation to prevent short positions when broker doesn't support it?
