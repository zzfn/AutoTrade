# Design: Long-Short Position Support

## Architecture Overview

The design focuses on minimal changes to the existing rebalancing logic while adding comprehensive long-short support.

## Data Flow

### Current Flow (Long-Only)

```
get_positions() → filter(qty > 0) → current_positions
                                          ↓
                                    to_sell = current - target
                                          ↓
                                    submit sell orders
                                          ↓
                                    calculate targets → buy/adjust
```

### New Flow (Long-Short)

```
get_positions() → all positions (qty can be ±) → current_positions
                                                    ↓
                                              to_close = current - target
                                                    ↓
                                    determine close side (buy/sell)
                                          ↓
                                    submit close orders
                                          ↓
                                    calculate targets → buy/sell/adjust
```

## Component Changes

### 1. Position Tracking Module

**Location**: `qlib_strat.py:416-423`

**Current**:
```python
all_positions = self.get_positions()
current_positions = {}
for pos in all_positions:
    symbol = pos.asset.symbol if hasattr(pos.asset, 'symbol') else str(pos.asset)
    qty = float(pos.quantity)
    if qty > 0:  # ❌ Only long
        current_positions[symbol] = qty
```

**Updated**:
```python
all_positions = self.get_positions()
current_positions = {}
for pos in all_positions:
    symbol = pos.asset.symbol if hasattr(pos.asset, 'symbol') else str(pos.asset)
    qty = float(pos.quantity)
    current_positions[symbol] = qty  # ✅ Both long (+) and short (-)
```

### 2. Position Closing Logic

**Location**: `qlib_strat.py:434-438`

**Current**:
```python
for symbol in to_sell:
    qty = current_positions[symbol]
    self.log_message(f"Selling {symbol}: {qty} shares")
    order = self.create_order(symbol, qty, "sell")
    self.submit_order(order)
```

**Updated**:
```python
for symbol in to_close:
    qty = current_positions[symbol]

    # Determine close side based on position direction
    if qty > 0:
        # Close long: sell the shares
        side = "sell"
        close_qty = qty
        self.log_message(f"Closing long {symbol}: {close_qty} shares (selling)")
    else:
        # Close short: buy to cover
        side = "buy"
        close_qty = abs(qty)
        self.log_message(f"Closing short {symbol}: {close_qty} shares (buying to cover)")

    order = self.create_order(symbol, close_qty, side)
    self.submit_order(order)
```

### 3. Target Position Adjustment (Optional Short Opening)

**Location**: `qlib_strat.py:469-493`

**Enhancement** (if supporting short targeting):
```python
for symbol in target_symbols:
    # Calculate target quantity (can be negative for short)
    target_qty = calculate_target_qty(...)  # May return negative value

    current_qty = current_positions.get(symbol, 0)
    diff = target_qty - current_qty

    if diff > 0:
        # Buying (open long or cover short)
        # ... existing buy logic
    elif diff < 0:
        # Selling (close long or open short)
        sell_qty = abs(diff)

        # Check if this would open a new short
        if current_qty >= 0 and target_qty < 0:
            # Opening short position
            if not self.allow_short:
                self.log_message(f"Skipping short open for {symbol}: allow_short=False")
                continue

            # Check short ratio limit
            short_value = sell_qty * price
            if short_value > total_value * self.max_short_ratio:
                self.log_message(f"Reducing short {symbol}: exceeds max_short_ratio")
                sell_qty = int(total_value * self.max_short_ratio / price)

        self.log_message(f"Selling {symbol}: {sell_qty} shares @ ${price:.2f}")
        order = self.create_order(symbol, sell_qty, "sell")
        self.submit_order(order)
```

## Configuration

### New Parameters

Add to `DEFAULT_PARAMETERS`:

```python
DEFAULT_PARAMETERS = {
    # ... existing parameters
    "allow_short": False,           # Enable short selling
    "max_short_ratio": 0.3,         # Max short exposure (30%)
}
```

### Initialization

```python
def initialize(self):
    # ... existing init
    self.allow_short = self.parameters.get("allow_short", False)
    self.max_short_ratio = self.parameters.get("max_short_ratio", 0.3)
```

## Edge Cases

### 1. Zero Quantity
- Should be treated as "close position"
- Handled by: `to_close = current.keys() - target.keys()`

### 2. Partial Closing
- Long: 100 → 50 (sell 50)
- Short: -100 → -50 (buy 50 to cover)
- Handled by: `diff = target - current`

### 3. Direction Reversal
- Long to Short: 100 → -50 (sell 100, then sell 50 more)
- Short to Long: -100 → 50 (buy 100 to cover, then buy 50 more)
- Handled by: `diff = target - current` where sign changes

### 4. Portfolio Value with Shorts
- Lumibot's `portfolio_value` should account for short P&L
- Validation: Ensure total_value includes unrealized P&L from shorts

## Testing Strategy

See [tasks.md](./tasks.md) for comprehensive test plan.
