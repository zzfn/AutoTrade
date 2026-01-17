from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import asyncio
import os
import threading
from datetime import datetime
from dotenv import load_dotenv

import logging
import signal

load_dotenv()
from autotrade.trade_manager import TradeManager
from autotrade.strategies.simple_strategy import SimpleStrategy

# Monkey patch signal.signal to prevent ValueError in non-main threads
_original_signal = signal.signal
def _thread_safe_signal(signum, handler):
    if threading.current_thread() is not threading.main_thread():
        logging.warning(f"Ignored signal registration for {signum} from non-main thread.")
        return
    return _original_signal(signum, handler)

signal.signal = _thread_safe_signal

# LumiBot Imports
from lumibot.brokers import Alpaca
from lumibot.traders import Trader
from lumibot.backtesting import YahooDataBacktesting

app = FastAPI()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Mounts
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Manager
tm = TradeManager()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/backtest", response_class=HTMLResponse)
async def read_backtest(request: Request):
    return templates.TemplateResponse("backtest.html", {"request": request})

@app.post("/api/start")
async def start_trading():
    if tm.is_running:
        return {"status": "already_running"}

    # Setup Broker and Strategy
    # Using Environment Variables
    API_KEY = os.getenv("ALPACA_API_KEY")
    SECRET_KEY = os.getenv("ALPACA_API_SECRET")
    PAPER = os.getenv("ALPACA_PAPER", "True").lower() == "true"

    if not API_KEY or not SECRET_KEY:
        tm.log("错误: 环境变量中未找到 Alpaca 凭证 (ALPACA_API_KEY, ALPACA_API_SECRET)。")
        return {"status": "error", "message": "缺少凭证"}

    try:
        broker = Alpaca({
            "API_KEY": API_KEY,
            "API_SECRET": SECRET_KEY,
            "PAPER": PAPER
        })
        
        strategy = SimpleStrategy(broker=broker)
        
        # LumiBot: Create a Trader to manage the strategy
        trader = Trader()
        trader.add_strategy(strategy)
        
        tm.set_strategy(strategy) # Store strategy ref
        
        # Run using the trader
        started = tm.start_strategy(runner=trader.run_all)
        return {"status": "started" if started else "failed"}
    except Exception as e:
        tm.log(f"Failed to setup strategy: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/stop")
async def stop_trading():
    tm.stop_strategy()
    return {"status": "stopping"}

@app.post("/api/run_backtest")
async def run_backtest(request: Request):
    # This might block, so we should run in thread again or use Celery (overkill).
    # MVP: Run in simple thread.
    params = await request.json()
    
    def _backtest_task():
        try:
            tm.log("Starting backtest...")
            # Setup Backtest
            start_date = datetime(2023, 1, 1) # Mock or parse from params
            end_date = datetime(2023, 12, 31)
            
            backtesting_start = datetime.strptime(params.get("start_date", "2023-01-01"), "%Y-%m-%d")
            backtesting_end = datetime.strptime(params.get("end_date", "2023-01-31"), "%Y-%m-%d")

            # Simple Strategy Backtest
            # Logic: YahooDataBacktesting
            
            from lumibot.backtesting import YahooDataBacktesting
            
            tm.log(f"Backtesting {backtesting_start} to {backtesting_end}")
            
            # Construct a new strategy instance for backtest to avoid conflict with live?
            # Yes. Define a subclass that mocks the TM logger or uses it.
            
            class BacktestStrategy(SimpleStrategy):
                def initialize(self):
                    self.sleeptime = "1D" 
                    self.tm = tm # Use same TM for logging
                    self.symbol = "SPY"
                    # We don't need polling in backtest as it drives iterations
                    
                def on_trading_iteration(self):
                    # SMA Crossover Strategy for Backtest
                    try:
                        # self.tm.log(f"Backtest Step: {self.get_datetime()}")
                        
                        # Get historical prices (enough for SMA30)
                        history = self.get_historical_prices(self.symbol, 40, "day")
                        if history is None or history.df.empty:
                            return

                        df = history.df
                        if len(df) < 30:
                            return
                            
                        # Calculate Indicators
                        sma10 = df['close'].rolling(window=10).mean().iloc[-1]
                        sma30 = df['close'].rolling(window=30).mean().iloc[-1]
                        price = self.get_last_price(self.symbol)
                        
                        # Trading Logic
                        position = self.get_position(self.symbol)
                        qty = float(position.quantity) if position else 0
                        
                        if sma10 > sma30:
                            # Buy signal
                            if qty == 0:
                                # Buy all in (simplified)
                                cash = self.get_cash()
                                # Reserve a little bit for safety
                                share_count = int((cash * 0.95) / price)
                                if share_count > 0:
                                    order = self.create_order(self.symbol, share_count, "buy")
                                    self.submit_order(order)
                                    self.tm.log(f"[{self.get_datetime()}] BUY {self.symbol} @ {price:.2f} (SMA10>SMA30)")
                        elif sma10 < sma30:
                            # Sell signal
                            if qty > 0:
                                order = self.create_order(self.symbol, qty, "sell")
                                self.submit_order(order)
                                self.tm.log(f"[{self.get_datetime()}] SELL {self.symbol} @ {price:.2f} (SMA10<SMA30)")
                                
                    except Exception as e:
                        # Log less frequently or just errors
                        # self.tm.log(f"BT Error: {e}")
                        pass

            # Run Backtest
            # Note: YahooDataBacktesting runs the strategy
            try:
                BacktestStrategy.backtest(
                    YahooDataBacktesting,
                    backtesting_start,
                    backtesting_end,
                    benchmark_asset="SPY",
                    parameters={}
                )
                tm.log("Backtest finished successfully.")
            except Exception as e:
                tm.log(f"Backtest execution failed: {e}")
                # Important: Print stack trace to console for debugging
                import traceback
                traceback.print_exc()
            
        except Exception as e:
            tm.log(f"Backtest error: {e}")

    thread = threading.Thread(target=_backtest_task)
    thread.start()
    
    return {"status": "backtest_started"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Poll state from TM
            state = tm.state
            await websocket.send_json(state)
            await asyncio.sleep(1) # 1Hz update
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WS Error: {e}")
