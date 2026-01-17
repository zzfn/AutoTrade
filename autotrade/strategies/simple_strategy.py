from lumibot.strategies.strategy import Strategy
from autotrade.trade_manager import TradeManager
from datetime import datetime
import threading
import time

class SimpleStrategy(Strategy):
    def initialize(self):
        self.sleeptime = "5S" 
        self.tm = TradeManager()
        self.tm.log("SimpleStrategy initialized")
        # Use BTC/USD to allow trading 24/7 for demo purposes
        self.symbol = "BTC/USD"
        
        # Initial update
        self._update_portfolio_state()
        
        # Start a background thread to keep updating portfolio even if market is closed
        self._stop_poller = threading.Event()
        self._poller_thread = threading.Thread(target=self._portfolio_poller, daemon=True)
        self._poller_thread.start()

    def _portfolio_poller(self):
        """Poll portfolio separately from trading iteration to support closed markets."""
        while not self._stop_poller.is_set():
            try:
                self._update_portfolio_state()
                time.sleep(10)
            except Exception as e:
                # self.tm.log(f"Portfolio poll warning: {e}") # Reduce spam
                time.sleep(10)

    def _update_portfolio_state(self):
        try:
            # Use broker directly for more robust fetching
            # Attempt to get cash
            try:
                cash = float(self.broker.get_cash())
            except:
                cash = 0.0
            
            # Attempt to get portfolio value
            try:
                # Some brokers/strategies wrap this differently
                portfolio_value = float(self.portfolio_value)
            except:
                portfolio_value = cash
            
            # Check market status
            market_status = "unknown"
            try:
                # Alpaca specific check
                clock = self.broker.get_clock()
                if clock:
                    market_status = "open" if clock.is_open else "closed"
            except:
                pass

            positions_list = []
            try:
                # Try to get positions from broker directly if strategy.positions is stale/empty
                # But Strategy.positions should be kept in sync by LumiBot.
                
                # NOTE: self.positions can be a list or a dict depending on version/broker
                raw_positions = self.positions
                if raw_positions:
                    # Normalize to an iterator of Position objects
                    if isinstance(raw_positions, dict):
                        iterator = raw_positions.values()
                    elif isinstance(raw_positions, list):
                        iterator = raw_positions
                    else:
                        iterator = []

                    for position in iterator:
                        # Extract asset/symbol safely
                        asset = getattr(position, "asset", None)
                        if asset is None:
                            # Fallback if asset is not an attribute (unlikely)
                            asset = "UNKNOWN"
                        
                        symbol = str(asset)
                        
                        # Helper to get attribute with fallbacks
                        def get_attr(obj, attrs, default=0.0):
                            for attr in attrs:
                                if hasattr(obj, attr):
                                    val = getattr(obj, attr)
                                    if val is not None:
                                        return float(val)
                            return default

                        quantity = get_attr(position, ["quantity", "qty"], 0.0)
                        avg_price = get_attr(position, ["average_price", "avg_price", "avg_entry_price", "cost_basis"], 0.0)
                        
                        current_price = 0.0
                        try:
                            # Try getting price from strategy first, then position, then fallback
                            if hasattr(self, 'get_last_price'):
                                current_price = float(self.get_last_price(asset))
                            elif hasattr(position, 'current_price'):
                                current_price = float(position.current_price)
                        except:
                            pass
                        
                        if current_price == 0.0:
                             current_price = avg_price

                        # Try to get unrealized PL directly from position object
                        unrealized_pl = get_attr(position, ["unrealized_pl", "unrealized_profit_loss"], 0.0)
                        
                        # If PL is not directly available, calculate it only if we have valid prices
                        # But user requested "no calculate if returned", so we prioritize theattribute.
                        # If attribute is 0.0, we might still want to calc if it really missing, 
                        # but let's assume if it is returned it is correct.
                        
                        # However, for robustness:
                        if unrealized_pl == 0.0 and current_price != 0.0 and avg_price != 0.0:
                             unrealized_pl = (current_price - avg_price) * quantity

                        positions_list.append({
                            "symbol": symbol,
                            "quantity": quantity,
                            "average_price": avg_price,
                            "current_price": current_price,
                            "unrealized_pl": unrealized_pl
                        })
            except Exception as e:
                self.tm.log(f"Error fetching positions: {e}")
            
            self.tm.update_portfolio(cash, portfolio_value, positions_list, market_status)
        except Exception as e:
            self.tm.log(f"Update/Poll critical error: {e}")

    def on_trading_iteration(self):
        try:
            self.tm.log("Executing trading iteration...")
            self._update_portfolio_state()
            self.tm.log(f"Heartbeat: {datetime.now().time()}")
            
            # Simple Trading Logic for Demo:
            # If we hold no SPY, buy some.
            # If we hold SPY, just hold (or sell if profit > X, but lets keep it simple)
            
            # Use get_position safely
            position = self.get_position(self.symbol)
            qty = 0
            if position:
                qty = float(position.quantity)
            
            if qty == 0:
                self.tm.log(f"No position in {self.symbol}. Buying...")
                # Calculate quantity to buy
                price = self.get_last_price(self.symbol)
                if price:
                    # Buy 5 shares
                    order = self.create_order(self.symbol, 5, "buy")
                    self.submit_order(order)
                    self.tm.log(f"Submitted buy order for 5 {self.symbol} @" + str(price))
            else:
                self.tm.log(f"Holding {qty} {self.symbol}. No action.")

        except Exception as e:
            self.tm.log(f"Error in trading iteration: {e}")
            
    def stop(self):
        # Cleanup
        if hasattr(self, "_stop_poller"):
            self._stop_poller.set()
        if hasattr(self, "_poller_thread"):
            self._poller_thread.join(timeout=2)
