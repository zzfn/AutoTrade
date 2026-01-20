# Tasks: Long-Short Position Support

## Phase 1: Core Implementation

- [ ] **Task 1.1**: Update position tracking to include negative quantities
  - File: `autotrade/strategies/qlib_strat.py:422`
  - Remove `if qty > 0` condition
  - Track all positions regardless of sign

- [ ] **Task 1.2**: Implement smart closing logic
  - File: `autotrade/strategies/qlib_strat.py:434-438`
  - Detect position direction (long/short)
  - Use correct order side for closing (sell for long, buy for short)
  - Add logging to indicate closing action

- [ ] **Task 1.3**: Add configuration parameters
  - Add `allow_short` parameter (default: False)
  - Add `max_short_ratio` parameter (default: 0.3)
  - Update `initialize()` to load parameters

- [ ] **Task 1.4**: (Optional) Implement short opening logic
  - Detect when selling would open short position
  - Check `allow_short` flag before opening shorts
  - Validate against `max_short_ratio` limit

## Phase 2: Testing

- [ ] **Task 2.1**: Add unit tests for short position closing
  - Test closing short positions with buy orders
  - Verify correct quantity (absolute value)
  - Check logging indicates "closing short"

- [ ] **Task 2.2**: Add unit tests for mixed long/short portfolios
  - Test portfolio with both long and short positions
  - Verify only non-target positions are closed
  - Ensure target positions are adjusted correctly

- [ ] **Task 2.3**: Add unit tests for short opening (if implemented)
  - Test `allow_short=False` prevents short opening
  - Test `allow_short=True` allows short opening
  - Test `max_short_ratio` enforcement

- [ ] **Task 2.4**: Add edge case tests
  - Test zero quantity handling
  - Test partial closing of short positions
  - Test direction reversal (long→short, short→long)
  - Test empty target list closes all positions

- [ ] **Task 2.5**: Integration tests
  - Run full strategy cycle with short positions
  - Verify portfolio value calculation
  - Test order execution sequence

## Phase 3: Documentation

- [ ] **Task 3.1**: Update strategy documentation
  - Document long-short capabilities
  - Explain risks of short selling
  - Provide configuration examples

- [ ] **Task 3.2**: Update parameter documentation
  - Document `allow_short` parameter
  - Document `max_short_ratio` parameter
  - Add warnings about short selling risks

## Phase 4: Validation

- [ ] **Task 4.1**: Code review
  - Review position closing logic
  - Review short opening logic
  - Verify edge case handling

- [ ] **Task 4.2**: Backward compatibility check
  - Ensure default behavior unchanged (allow_short=False)
  - Test with existing long-only configurations
  - Verify no breaking changes

- [ ] **Task 4.3**: Risk assessment
  - Validate short position risk handling
  - Check margin requirement handling
  - Verify broker compatibility

## Dependencies

- Lumibot position object API (need to verify `quantity` attribute behavior)
- Broker short selling support (IBKR, Alpaca, etc.)
- Margin account requirements

## Definition of Done

- [ ] All Phase 1 tasks complete
- [ ] All Phase 2 tests passing
- [ ] Documentation updated
- [ ] Code reviewed and approved
- [ ] Backward compatibility verified
- [ ] OpenSpec spec created and validated
