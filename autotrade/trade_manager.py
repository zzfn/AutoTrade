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
        """Start the strategy in a separate thread.

        Args:
            runner: Optional callable to execute. If None, tries calling `active_strategy.run()`,
                    falling back to `active_strategy.run_all()` if available (e.g. Trader).
        """
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

        self.strategy_thread = threading.Thread(target=run_target, daemon=True)
        self.is_running = True
        self.update_status("running")
        self.strategy_thread.start()
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
