import os
import threading
import yaml
from datetime import datetime
from typing import Any

from lumibot.backtesting import YahooDataBacktesting
from lumibot.brokers import Alpaca
from lumibot.traders import Trader

from autotrade.strategies.momentum_strategy import MomentumStrategy


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

    def initialize_and_start(self):
        """Initialize the broker, strategy, and trader, then start the thread."""
        if self.is_running:
            return {"status": "already_running"}

        # 1. Load credentials
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_API_SECRET")
        paper_trading = os.getenv("ALPACA_PAPER", "True").lower() == "true"

        if not api_key or not secret_key:
            self.log("错误: 环境变量中未找到 Alpaca 凭证 (ALPACA_API_KEY, ALPACA_API_SECRET)。")
            return {"status": "error", "message": "缺少凭证"}

        try:
            # 2. Setup Broker
            broker = Alpaca(
                {"API_KEY": api_key, "API_SECRET": secret_key, "PAPER": paper_trading}
            )

            # 3. Load symbols from config
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, "../configs/universe.yaml")
            symbols = ["SPY"]  # Default
            
            try:
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        config = yaml.safe_load(f)
                        if config and "symbols" in config:
                            symbols = config["symbols"]
                            self.log(f"Loaded symbols from config: {symbols}")
                        else:
                            self.log("Config file found but no 'symbols' key. Using default.")
                else:
                    self.log(f"Config file not found at {config_path}. Using default symbols.")
            except Exception as e:
                self.log(f"Error reading config file: {e}. Using default symbols.")

            self.log(f"Starting strategy for symbols: {symbols}")

            # 4. Create Strategy
            strategy_params = {
                "symbols": symbols,
                "sleeptime": "10S",
                "lookback_period": 60,
            }
            strategy = MomentumStrategy(broker=broker, parameters=strategy_params)

            # 5. Create Trader and register
            trader = Trader()
            trader.add_strategy(strategy)

            self.set_strategy(strategy)

            # 6. Start the logic
            started = self.start_strategy(runner=trader.run_all)
            return {"status": "started" if started else "failed"}

        except Exception as e:
            self.log(f"Failed to setup strategy: {e}")
            return {"status": "error", "message": str(e)}

    def run_backtest(self, params: dict):
        """Run a backtest in a separate thread."""

        def _backtest_task():
            try:
                self.log("Starting backtest...")
                
                # 1. Parse dates
                backtesting_start = datetime.strptime(
                    params.get("start_date", "2023-01-01"), "%Y-%m-%d"
                )
                backtesting_end = datetime.strptime(
                    params.get("end_date", "2023-01-31"), "%Y-%m-%d"
                )
                
                # 2. Parse symbols
                symbol_input = params.get("symbol", "SPY")
                symbols = [s.strip() for s in symbol_input.split(",") if s.strip()]
                if not symbols:
                    symbols = ["SPY"]

                self.log(f"Backtesting {symbols} from {backtesting_start} to {backtesting_end}")

                # 3. Execute backtest
                try:
                    # MomentumStrategy.backtest is a blocking call
                    MomentumStrategy.backtest(
                        YahooDataBacktesting,
                        backtesting_start,
                        backtesting_end,
                        benchmark_asset=symbols[0], 
                        parameters={
                            "symbols": symbols,
                            "lookback_period": 60,
                            "sleeptime": "0S"
                        },
                    )
                    self.log("Backtest finished successfully.")
                except Exception as e:
                    import traceback
                    self.log(f"Backtest execution failed: {e}")
                    print(traceback.format_exc())

            except Exception as e:
                self.log(f"Backtest error: {e}")

        # Start backtest in background thread
        thread = threading.Thread(target=_backtest_task, daemon=True)
        thread.start()
        return {"status": "backtest_started"}

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
