"""
Unit tests for QlibMLStrategy position management.

Tests cover:
1. Orphan position detection and handling
2. Equal weight allocation
3. Weighted allocation based on prediction scores
"""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np


class MockPosition:
    """Mock Lumibot Position object."""
    def __init__(self, symbol: str, quantity: float):
        self.asset = MagicMock()
        self.asset.symbol = symbol
        self.quantity = quantity


class MockStrategy:
    """
    Minimal mock of QlibMLStrategy for testing weight calculation
    and rebalancing logic without actual Lumibot dependencies.
    """
    def __init__(self, symbols: list, position_sizing: str = "equal"):
        self.symbols = symbols
        self.position_sizing = position_sizing
        self.current_predictions = {}
        self.logs = []
        self.orders = []
        self._positions = []
        self._portfolio_value = 100000.0
        self._cash = 100000.0
        self._prices = {}
        
    def log_message(self, msg):
        self.logs.append(msg)
        
    def get_positions(self):
        return self._positions
    
    def get_position(self, symbol):
        for pos in self._positions:
            if pos.asset.symbol == symbol:
                return pos
        return None
    
    @property
    def portfolio_value(self):
        return self._portfolio_value
    
    def get_cash(self):
        return self._cash
    
    def get_last_price(self, symbol):
        return self._prices.get(symbol, 100.0)
    
    def create_order(self, symbol, qty, side):
        return {"symbol": symbol, "quantity": qty, "side": side}
    
    def submit_order(self, order):
        self.orders.append(order)

    def _calculate_target_weights(self, predictions: dict, target_symbols: list) -> dict:
        """
        Calculate target weights for each symbol based on position_sizing mode.
        """
        if not target_symbols:
            return {}

        if self.position_sizing == "weighted":
            # Linear normalization: shift scores to be positive, then normalize
            scores = np.array([predictions.get(s, 0.0) for s in target_symbols])
            min_score = np.min(scores)
            shifted_scores = scores - min_score + 1e-6
            weights = shifted_scores / np.sum(shifted_scores)
            return {symbol: float(w) for symbol, w in zip(target_symbols, weights)}
        else:
            weight = 1.0 / len(target_symbols)
            return {symbol: weight for symbol in target_symbols}

    def _rebalance_portfolio(self, target_symbols: list, predictions: dict = None):
        """
        Rebalance portfolio to target holdings.
        """
        all_positions = self.get_positions()
        current_positions = {}
        for pos in all_positions:
            symbol = pos.asset.symbol if hasattr(pos.asset, 'symbol') else str(pos.asset)
            qty = float(pos.quantity)
            # Track both long (qty > 0) and short (qty < 0) positions
            current_positions[symbol] = qty

        to_close = set(current_positions.keys()) - set(target_symbols)

        orphans = to_close - set(self.symbols)
        if orphans:
            self.log_message(f"Found orphan positions to close: {orphans}")

        # Close positions not in target (handle both long and short)
        for symbol in to_close:
            qty = current_positions[symbol]
            if qty == 0:
                continue

            # Determine close side based on position direction
            if qty > 0:
                # Close long position: sell the shares
                side = "sell"
                close_qty = qty
                self.log_message(f"Closing long {symbol}: {close_qty} shares (selling)")
            else:
                # Close short position: buy to cover
                side = "buy"
                close_qty = abs(qty)
                self.log_message(f"Closing short {symbol}: {close_qty} shares (buying to cover)")

            order = self.create_order(symbol, close_qty, side)
            self.submit_order(order)

        total_value = self.portfolio_value or self.get_cash()
        if total_value <= 0:
            return

        available_capital = total_value * 0.95

        if predictions is None:
            predictions = self.current_predictions or {}
        weights = self._calculate_target_weights(predictions, target_symbols)

        for symbol in target_symbols:
            price = self.get_last_price(symbol)
            if price is None or price <= 0:
                continue

            weight = weights.get(symbol, 0.0)
            target_value = available_capital * weight
            target_qty = int(target_value / price)

            current_qty = current_positions.get(symbol, 0)
            if current_qty < 0:
                cover_qty = abs(current_qty)
                if cover_qty > 0:
                    self.log_message(
                        f"Covering short {symbol}: {cover_qty} shares (buying to cover)"
                    )
                    order = self.create_order(symbol, cover_qty, "buy")
                    self.submit_order(order)
                current_qty = 0
            diff = target_qty - current_qty

            if diff > 0:
                self._cash -= diff * price
                buy_qty = diff
                if buy_qty > 0:
                    self.log_message(
                        f"Buying {symbol}: {buy_qty} shares @ ${price:.2f} (weight: {weight:.2%})"
                    )
                    order = self.create_order(symbol, buy_qty, "buy")
                    self.submit_order(order)

            elif diff < 0:
                sell_qty = abs(diff)
                if sell_qty > 0:
                    self.log_message(
                        f"Reducing {symbol}: {sell_qty} shares @ ${price:.2f}"
                    )
                    order = self.create_order(symbol, sell_qty, "sell")
                    self.submit_order(order)


class TestOrphanPositionHandling:
    """Test that orphan positions (not in symbols list) are properly sold."""
    
    def test_orphan_position_detected_and_sold(self):
        """Positions not in self.symbols should be detected and sold."""
        strategy = MockStrategy(symbols=["AAPL", "MSFT", "GOOGL"])
        
        # Create positions including an orphan (ABC not in symbols)
        strategy._positions = [
            MockPosition("AAPL", 10),
            MockPosition("ABC", 20),  # Orphan position
        ]
        
        # Rebalance to target only AAPL and MSFT
        target_symbols = ["AAPL", "MSFT"]
        predictions = {"AAPL": 0.1, "MSFT": 0.05}
        
        strategy._rebalance_portfolio(target_symbols, predictions)
        
        # Verify orphan was detected
        assert any("orphan" in log.lower() for log in strategy.logs), \
            f"Expected orphan detection log, got: {strategy.logs}"
        
        # Verify ABC was sold
        sell_orders = [o for o in strategy.orders if o["side"] == "sell" and o["symbol"] == "ABC"]
        assert len(sell_orders) == 1
        assert sell_orders[0]["quantity"] == 20
    
    def test_no_orphan_log_when_all_in_symbols(self):
        """No orphan log should appear when all positions are in symbols."""
        strategy = MockStrategy(symbols=["AAPL", "MSFT", "GOOGL"])
        
        strategy._positions = [
            MockPosition("AAPL", 10),
            MockPosition("MSFT", 15),
        ]
        
        target_symbols = ["AAPL", "GOOGL"]
        predictions = {"AAPL": 0.1, "GOOGL": 0.08}
        
        strategy._rebalance_portfolio(target_symbols, predictions)
        
        # Should not have orphan log
        assert not any("orphan" in log.lower() for log in strategy.logs), \
            f"Unexpected orphan log: {strategy.logs}"
        
        # MSFT should be sold (it's in symbols but not in target)
        sell_orders = [o for o in strategy.orders if o["side"] == "sell" and o["symbol"] == "MSFT"]
        assert len(sell_orders) == 1


class TestEqualWeightAllocation:
    """Test equal weight allocation mode."""
    
    def test_equal_weights_calculated_correctly(self):
        """Equal mode should give each symbol the same weight."""
        strategy = MockStrategy(symbols=["AAPL", "MSFT", "GOOGL"], position_sizing="equal")
        
        predictions = {"AAPL": 0.15, "MSFT": 0.10, "GOOGL": 0.05}
        target_symbols = ["AAPL", "MSFT", "GOOGL"]
        
        weights = strategy._calculate_target_weights(predictions, target_symbols)
        
        # All weights should be equal
        expected_weight = 1.0 / 3
        for symbol in target_symbols:
            assert abs(weights[symbol] - expected_weight) < 1e-9
    
    def test_equal_weights_sum_to_one(self):
        """Equal weights should sum to 1.0."""
        strategy = MockStrategy(symbols=["AAPL", "MSFT"], position_sizing="equal")
        
        predictions = {"AAPL": 0.1, "MSFT": 0.2}
        target_symbols = ["AAPL", "MSFT"]
        
        weights = strategy._calculate_target_weights(predictions, target_symbols)
        
        assert abs(sum(weights.values()) - 1.0) < 1e-9


class TestWeightedAllocation:
    """Test weighted allocation based on prediction scores."""
    
    def test_weighted_higher_score_gets_more_weight(self):
        """Higher prediction score should result in higher weight."""
        strategy = MockStrategy(symbols=["AAPL", "MSFT", "GOOGL"], position_sizing="weighted")
        
        # AAPL has highest score
        predictions = {"AAPL": 0.20, "MSFT": 0.10, "GOOGL": 0.05}
        target_symbols = ["AAPL", "MSFT", "GOOGL"]
        
        weights = strategy._calculate_target_weights(predictions, target_symbols)
        
        # AAPL should have highest weight
        assert weights["AAPL"] > weights["MSFT"] > weights["GOOGL"]
    
    def test_weighted_weights_sum_to_one(self):
        """Weighted allocation weights should sum to 1.0."""
        strategy = MockStrategy(symbols=["AAPL", "MSFT", "GOOGL"], position_sizing="weighted")
        
        predictions = {"AAPL": 0.15, "MSFT": 0.10, "GOOGL": 0.05}
        target_symbols = ["AAPL", "MSFT", "GOOGL"]
        
        weights = strategy._calculate_target_weights(predictions, target_symbols)
        
        assert abs(sum(weights.values()) - 1.0) < 1e-9
    
    def test_weighted_with_negative_scores(self):
        """Weighted allocation should handle negative scores via linear normalization."""
        strategy = MockStrategy(symbols=["AAPL", "MSFT", "GOOGL"], position_sizing="weighted")
        
        predictions = {"AAPL": 0.05, "MSFT": -0.05, "GOOGL": -0.10}
        target_symbols = ["AAPL", "MSFT", "GOOGL"]
        
        weights = strategy._calculate_target_weights(predictions, target_symbols)
        
        # Should still sum to 1
        assert abs(sum(weights.values()) - 1.0) < 1e-9
        # Higher score still gets more weight
        assert weights["AAPL"] > weights["MSFT"] > weights["GOOGL"]
    
    def test_weighted_with_equal_scores(self):
        """When all scores are equal, weighted should give equal weights."""
        strategy = MockStrategy(symbols=["AAPL", "MSFT"], position_sizing="weighted")
        
        predictions = {"AAPL": 0.10, "MSFT": 0.10}
        target_symbols = ["AAPL", "MSFT"]
        
        weights = strategy._calculate_target_weights(predictions, target_symbols)
        
        # Should be approximately equal
        assert abs(weights["AAPL"] - weights["MSFT"]) < 1e-9
        assert abs(weights["AAPL"] - 0.5) < 1e-9
    
    def test_linear_normalization_numerical_stability(self):
        """Linear normalization should handle large score differences without issues."""
        strategy = MockStrategy(symbols=["AAPL", "MSFT"], position_sizing="weighted")
        
        # Large score difference
        predictions = {"AAPL": 100.0, "MSFT": 0.0}
        target_symbols = ["AAPL", "MSFT"]
        
        weights = strategy._calculate_target_weights(predictions, target_symbols)
        
        # Should not have NaN or Inf
        assert not np.isnan(weights["AAPL"])
        assert not np.isnan(weights["MSFT"])
        assert not np.isinf(weights["AAPL"])
        assert not np.isinf(weights["MSFT"])
        
        # Sum should still be 1
        assert abs(sum(weights.values()) - 1.0) < 1e-9
        
        # AAPL should have much higher weight (linear, so proportional)
        assert weights["AAPL"] > weights["MSFT"]


class TestEmptyTargets:
    """Test edge cases with empty targets."""
    
    def test_empty_target_symbols(self):
        """Empty target list should return empty weights."""
        strategy = MockStrategy(symbols=["AAPL"], position_sizing="equal")
        
        weights = strategy._calculate_target_weights({}, [])
        
        assert weights == {}
    
    def test_rebalance_with_empty_targets_sells_all(self):
        """Rebalancing to empty targets should sell all positions."""
        strategy = MockStrategy(symbols=["AAPL", "MSFT"])
        
        strategy._positions = [
            MockPosition("AAPL", 10),
            MockPosition("MSFT", 15),
        ]
        
        strategy._rebalance_portfolio([], {})
        
        # Both positions should be sold
        sell_orders = [o for o in strategy.orders if o["side"] == "sell"]
        assert len(sell_orders) == 2




class TestLongShortPositions:
    """Test long and short position handling."""

    def test_short_position_closing_with_buy(self):
        """Short positions should be closed with buy orders."""
        strategy = MockStrategy(symbols=["AAPL", "MSFT", "GOOGL"])

        # Create positions including a short position
        strategy._positions = [
            MockPosition("AAPL", 100),   # Long
            MockPosition("TSLA", -50),   # Short (orphan)
        ]

        # Rebalance to only AAPL
        target_symbols = ["AAPL"]
        predictions = {"AAPL": 0.1}

        strategy._rebalance_portfolio(target_symbols, predictions)

        # Verify short was detected
        assert any("short" in log.lower() for log in strategy.logs), \
            f"Expected short position log, got: {strategy.logs}"

        # Verify TSLA short was closed with a buy order
        buy_orders = [o for o in strategy.orders if o["side"] == "buy" and o["symbol"] == "TSLA"]
        assert len(buy_orders) == 1
        assert buy_orders[0]["quantity"] == 50  # abs(-50)

    def test_long_position_closing_with_sell(self):
        """Long positions should be closed with sell orders."""
        strategy = MockStrategy(symbols=["AAPL", "MSFT"])

        strategy._positions = [
            MockPosition("AAPL", 100),
            MockPosition("MSFT", 50),
        ]

        # Rebalance to only AAPL
        target_symbols = ["AAPL"]
        predictions = {"AAPL": 0.1}

        strategy._rebalance_portfolio(target_symbols, predictions)

        # Verify MSFT long was closed with a sell order
        sell_orders = [o for o in strategy.orders if o["side"] == "sell" and o["symbol"] == "MSFT"]
        assert len(sell_orders) == 1
        assert sell_orders[0]["quantity"] == 50

    def test_mixed_long_and_short_portfolio(self):
        """Test portfolio with both long and short positions."""
        strategy = MockStrategy(symbols=["AAPL", "MSFT", "GOOGL"])

        # Mixed portfolio
        strategy._positions = [
            MockPosition("AAPL", 100),   # Long
            MockPosition("MSFT", -30),   # Short
            MockPosition("GOOGL", 50),   # Long
        ]

        # Rebalance to AAPL and GOOGL (close MSFT short)
        target_symbols = ["AAPL", "GOOGL"]
        predictions = {"AAPL": 0.1, "GOOGL": 0.05}

        strategy._rebalance_portfolio(target_symbols, predictions)

        # MSFT should be closed
        close_orders = [o for o in strategy.orders if o["symbol"] == "MSFT"]
        assert len(close_orders) == 1
        assert close_orders[0]["side"] == "buy"  # Buy to cover short
        assert close_orders[0]["quantity"] == 30

        # AAPL and GOOGL should NOT be closed
        aapl_close = [o for o in strategy.orders if o["symbol"] == "AAPL"]
        googl_close = [o for o in strategy.orders if o["symbol"] == "GOOGL"]
        assert len(aapl_close) == 0
        assert len(googl_close) == 0

    def test_all_short_positions_closed(self):
        """All short positions should be closed when target is empty."""
        strategy = MockStrategy(symbols=["AAPL", "MSFT"])

        strategy._positions = [
            MockPosition("AAPL", -50),   # Short
            MockPosition("MSFT", -30),   # Short
        ]

        strategy._rebalance_portfolio([], {})

        # Both shorts should be closed with buy orders
        buy_orders = [o for o in strategy.orders if o["side"] == "buy"]
        assert len(buy_orders) == 2

        # Verify quantities
        aapl_buy = [o for o in buy_orders if o["symbol"] == "AAPL"]
        msft_buy = [o for o in buy_orders if o["symbol"] == "MSFT"]
        assert len(aapl_buy) == 1 and aapl_buy[0]["quantity"] == 50
        assert len(msft_buy) == 1 and msft_buy[0]["quantity"] == 30

    def test_orphan_short_position_detected(self):
        """Orphan short positions should be logged and closed."""
        strategy = MockStrategy(symbols=["AAPL", "MSFT"])

        strategy._positions = [
            MockPosition("AAPL", 100),
            MockPosition("TSLA", -40),  # Orphan short
        ]

        target_symbols = ["AAPL"]
        predictions = {"AAPL": 0.1}

        strategy._rebalance_portfolio(target_symbols, predictions)

        # Should detect orphan
        assert any("orphan" in log.lower() for log in strategy.logs)

        # Should close TSLA short with buy
        tsla_orders = [o for o in strategy.orders if o["symbol"] == "TSLA"]
        assert len(tsla_orders) == 1
        assert tsla_orders[0]["side"] == "buy"
        assert tsla_orders[0]["quantity"] == 40


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
