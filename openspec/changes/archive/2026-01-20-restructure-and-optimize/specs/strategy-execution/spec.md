# Strategy Execution Logic

## ADDED Requirements

### Requirement: Structured Trading Iteration

The `on_trading_iteration` method MUST follow a strict, sequential logical flow to ensure reliability and state consistency.

#### Scenario: Execution Sequence

- **Given** the strategy enters a new trading iteration
- **Then** it must execute steps in this exact order:
  1. **Time Check**: Call `self.get_datetime()` to establish the current context.
  2. **Order Check**: Verify if there are any pending or open orders that block new actions.
  3. **Data Fetch**: Call `self.get_historical_prices` to get latest market data.
  4. **Logic Calculation**: Invoke the Qlib pipeline to calculate signals/features.
  5. **Execution**: If a signal exists, use `create_order` and immediately update `self.vars` with the new state.

### Requirement: State Synchronization

Trading state MUST be captured immediately upon decision making to prevent memory amnesia on restart.

#### Scenario: Post-Trade Persistence

- **Given** an order has been created/submitted
- **When** the iteration logic concludes
- **Then** `self.vars` must reflect the latest action (e.g., entered position, stop loss level)
- **And** this state must be backed up (handled by the persistent variable backup requirement).
