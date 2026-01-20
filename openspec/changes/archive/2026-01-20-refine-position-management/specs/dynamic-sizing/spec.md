## ADDED Requirements

### Requirement: Orphan Position Cleanup

The strategy MUST identify and sell positions that are no longer in the target portfolio, even if they are not in the current candidate list.

#### Scenario: Rebalancing cleans up orphan positions

Given the strategy holds positions A, B, C
And A, B, C are NOT in the `symbols` candidate list provided to the strategy
And the new target portfolio is D, F, E
When `on_trading_iteration` runs
Then positions A, B, C should be sold completely
And positions D, F, E should be bought.

### Requirement: Dynamic Position Sizing

The strategy MUST support different capital allocation methods based on model confidence.

#### Scenario: Weighted position sizing

Given `position_sizing` is set to "weighted"
And Top-K selected stocks have prediction scores: A=0.8, B=0.5, C=0.2
When portfolio allocation is calculated
Then stock A should receive the highest allocation
And stock C should receive the lowest allocation
And allocations should sum to ~95% of total capital (preserving cash buffer).

#### Scenario: Equal position sizing (default)

Given `position_sizing` is set to "equal" (or default)
When portfolio allocation is calculated
Then all Top-K stocks should receive approximately equal capital allocation.
