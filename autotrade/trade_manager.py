import threading
from datetime import datetime
from typing import Any


class TradeManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self.active_strategy = None
        self.strategy_thread: threading.Thread | None = None
        self.is_running = False

        # State storage
        self.state: dict[str, Any] = {
            "status": "stopped",
            "logs": [],
            "orders": [],
            "portfolio": {"cash": 0.0, "value": 0.0, "positions": []},
            "market_status": "unknown",
            "last_update": None,
        }
        self._initialized = True

    def set_strategy(self, strategy_instance):
        """Set the strategy instance to be managed."""
        self.active_strategy = strategy_instance

    def start_strategy(self, runner=None):
        """Start the strategy in a separate thread and begin monitoring."""
        if self.is_running:
            return False

        def run_target():
            try:
                self.log("Starting strategy...")
                if runner:
                    runner()
                elif self.active_strategy:
                    if hasattr(self.active_strategy, "run_all"):
                        self.active_strategy.run_all()
                    elif hasattr(self.active_strategy, "run"):
                        self.active_strategy.run()
                    else:
                        raise AttributeError(
                            "Strategy has no run() or run_all() method and no runner provided."
                        )
                else:
                    raise ValueError("No strategy set and no runner provided.")
            except Exception as e:
                self.log(f"Strategy error: {str(e)}")
            finally:
                self.is_running = False
                self.update_status("stopped")
                self.log("Strategy stopped.")

        def monitor_target():
            """Polls the strategy for state updates while it is running."""
            import time
            while self.is_running:
                try:
                    if self.active_strategy and hasattr(self.active_strategy, 'get_datetime'):
                        # Only update if the strategy is actually initialized and running
                        # Note: We use a try-except because LumiBot methods might fail if not ready
                        try:
                            cash = float(self.active_strategy.get_cash())
                            value = float(self.active_strategy.portfolio_value)
                            
                            # Get symbols from the strategy
                            symbols = getattr(self.active_strategy, 'symbols', [])
                            positions_data = []
                            for symbol in symbols:
                                pos = self.active_strategy.get_position(symbol)
                                if pos:
                                    positions_data.append({
                                        "symbol": symbol,
                                        "quantity": float(pos.quantity),
                                        "average_price": float(pos.average_price),
                                        "current_price": float(self.active_strategy.get_last_price(symbol)) if hasattr(self.active_strategy, 'get_last_price') else 0.0,
                                        "unrealized_pl": float(pos.unrealized_pl) if hasattr(pos, 'unrealized_pl') else 0.0,
                                        "unrealized_plpc": float(pos.unrealized_plpc) if hasattr(pos, 'unrealized_plpc') else 0.0,
                                    })
                            
                            # Sync orders
                            lumi_orders = self.active_strategy.get_orders()
                            current_order_ids = [o["id"] for o in self.state["orders"]]
                            
                            for o in lumi_orders:
                                order_id = str(o.identifier)
                                if order_id not in current_order_ids:
                                    order_info = {
                                        "id": order_id,
                                        "symbol": o.asset.symbol,
                                        "action": str(o.side).upper(),
                                        "quantity": float(o.quantity),
                                        "price": float(o.price) if o.price else 0.0,
                                        "status": str(o.status),
                                        "timestamp": self.active_strategy.get_datetime().isoformat()
                                    }
                                    self.add_order(order_info)
                                    self.log(f"New Order: {order_info['action']} {order_info['quantity']} {order_info['symbol']} @ {order_info['price']}")
                                else:
                                    # Update status of existing orders if they changed
                                    for existing in self.state["orders"]:
                                        if existing["id"] == order_id and existing["status"] != str(o.status):
                                            existing["status"] = str(o.status)
                                            self.log(f"Order Update: {order_id} is now {str(o.status)}")

                            market_status = "open" if self.active_strategy.get_datetime().weekday() < 5 else "closed"
                            self.update_portfolio(cash, value, positions_data, market_status=market_status)
                        except:
                            pass # Might not be fully initialized yet
                except Exception as e:
                    print(f"Monitor error: {e}")
                time.sleep(1) # Poll every 1 second

        self.is_running = True
        self.update_status("running")
        
        # Start both strategy and monitor
        self.strategy_thread = threading.Thread(target=run_target, daemon=True)
        self.monitor_thread = threading.Thread(target=monitor_target, daemon=True)
        
        self.strategy_thread.start()
        self.monitor_thread.start()
        return True

    def stop_strategy(self):
        """Stop the running strategy."""
        if self.active_strategy:
            # LumiBot strategies usually have a minimal teardown or we just stop the process
            # But here we rely on the framework or just wait if we can signal it.
            # Assuming we can't easily 'kill' it gracefully without support from strategy logic.
            # For now, we update status and hopefully the strategy implementation checks it,
            # but LumiBot's `run()` is blocking loop.
            # We might need to call `self.active_strategy.stop()` if that exists?
            # Checking LumiBot docs mentally: Lifecycle can be tricky.
            # Use lifecycle management if possible.
            pass

        # For this MVP, we might treat 'stop' as just marking it.
        # But really we want to abort.
        # We will assume the strategy checks a flag or we restart the server.
        # Let's just update the status for now.
        self.update_status("stopping")
        # In a real app we'd need a way to interrupt the loop.

    def update_status(self, status: str):
        self.state["status"] = status
        self.state["last_update"] = datetime.now().isoformat()

    def log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.state["logs"].append(f"[{timestamp}] {message}")
        print(f"[TM LOG] {message}")
        if len(self.state["logs"]) > 100:
            self.state["logs"].pop(0)

    def update_portfolio(self, cash, value, positions, market_status="unknown"):
        # print(f"DEBUG: Updating portfolio: Cash={cash}, Val={value}, Pos={len(positions)}, Market={market_status}")
        self.state["portfolio"] = {"cash": cash, "value": value, "positions": positions}
        self.state["market_status"] = market_status
        self.state["last_update"] = datetime.now().isoformat()

    def add_order(self, order_info):
        self.state["orders"].insert(0, order_info)
        if len(self.state["orders"]) > 50:
            self.state["orders"].pop()

    def get_state(self):
        return self.state
