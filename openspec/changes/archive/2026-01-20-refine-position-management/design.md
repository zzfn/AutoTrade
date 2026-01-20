# Design: Refine Position Management

## 1. Robust Rebalancing logic

Currently:

```python
current_positions = {}
for symbol in self.symbols:
    pos = self.get_position(symbol)
```

New Logic:

```python
# Pseudo-code relying on Lumibot's get_positions which returns all held assets
all_positions = self.get_positions() # Returns list of Position objects
current_positions = {p.asset.symbol: float(p.quantity) for p in all_positions}

# Calculate to_sell based on ALL holdings vs target
to_sell = set(current_positions.keys()) - set(target_symbols)
```

This ensures we clean up _any_ position not in our target list, fixing the "ABC ignored" issue.

## 2. Dynamic Position Sizing

We will add `position_sizing` parameter to `QlibMLStrategy`.

### Weighted Calculation

If `position_sizing="weighted"`:

1. Get prediction scores for Top-K stocks.
2. Apply softmax to scores to standardise them to (0, 1) summing to 1.
   $$ w_i = \frac{e^{s_i}}{\sum e^{s_j}} $$
   where $s$ are the raw prediction scores.
3. Allocation = `Total Capital * 0.95 * Weight`.

### Interface Changes

```python
class QlibMLStrategy(Strategy):
    parameters = {
        ...,
        "position_sizing": "equal", # "equal", "weighted"
    }
```
