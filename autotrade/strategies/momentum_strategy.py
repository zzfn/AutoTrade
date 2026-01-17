import threading
import time
from datetime import datetime
import pandas as pd

from lumibot.strategies.strategy import Strategy
from autotrade.trade_manager import TradeManager


class MomentumStrategy(Strategy):
    """
    统一的策略类，目前实现了基于 SMA 的动量策略。
    后续可集成 Qlib/LightGBM 模型信号。
    """
    
    # 默认参数
    parameters = {
        "symbols": ["SPY"], 
        "sleeptime": "5S",
        "lookback_period": 40,
        "sma_fast": 10,
        "sma_slow": 30
    }

    def initialize(self):
        # 参数处理
        # 优先读取 symbols 列表，如果没有则回退到 compatible 单 symbol
        if "symbols" in self.parameters:
            self.symbols = self.parameters["symbols"]
        elif "symbol" in self.parameters:
            self.symbols = [self.parameters["symbol"]]
        else:
            self.symbols = ["SPY"]

        # 确保格式统一为列表
        if isinstance(self.symbols, str):
            # 处理逗号分隔字符串的情况 (e.g. "SPY,AAPL")
            if "," in self.symbols:
                self.symbols = [s.strip() for s in self.symbols.split(",")]
            else:
                self.symbols = [self.symbols]

        self.sleeptime = self.parameters.get("sleeptime", "5S")
        self.lookback_period = self.parameters.get("lookback_period", 40)
        self.sma_fast = self.parameters.get("sma_fast", 10)
        self.sma_slow = self.parameters.get("sma_slow", 30)

        # 初始化 TradeManager
        self.tm = TradeManager()
        self.tm.log(f"MomentumStrategy initialized for {len(self.symbols)} symbols: {self.symbols}")

        # 实盘专用：后台轮询资产状态
        # Lumibot backtest 模式下通常不需要这个自建的 poller，
        # 我们可以通过判断是否是回测环境来决定是否启动，但目前为了前端兼容性先保留
        # 简单的判断方式：如果 broker 是 backtesting 类型，可能不需要
        self._stop_poller = threading.Event()
        self._poller_thread = threading.Thread(
            target=self._portfolio_poller, daemon=True
        )
        self._poller_thread.start()

    def on_trading_iteration(self):
        """
        核心交易逻辑 - 支持多标的
        """
        try:
            self._update_portfolio_state()

            # --- 资金管理: 等权重分配 ---
            # 获取当前总资产价值 (Cash + Positions)
            total_portfolio_value = self.portfolio_value
            if total_portfolio_value is None or total_portfolio_value == 0:
                total_portfolio_value = self.get_cash()
            
            # 简单的等权重预算： 总资产 / 标的数量
            # 预留 5% 现金作为缓冲
            target_allocation = (total_portfolio_value * 0.95) / len(self.symbols)

            # 遍历每个标的执行逻辑
            for symbol in self.symbols:
                self._process_single_symbol(symbol, target_allocation)

        except Exception as e:
            self.tm.log(f"Error in trading iteration: {e}")
            import traceback
            traceback.print_exc()

    def _process_single_symbol(self, symbol, budget):
        """
        处理单个标的的信号与交易
        :param symbol: 标的代码
        :param budget: 分配给该标的的最大资金额度
        """
        try:
            # 1. 获取数据
            history = self.get_historical_prices(symbol, self.lookback_period, "day")
            if history is None or history.df.empty:
                return
            
            df = history.df
            if len(df) < self.sma_slow:
                return

            # 2. 计算信号 (SMA)
            sma_fast_val = df["close"].rolling(window=self.sma_fast).mean().iloc[-1]
            sma_slow_val = df["close"].rolling(window=self.sma_slow).mean().iloc[-1]
            price = self.get_last_price(symbol)
            
            if price is None or price <= 0:
                return

            # 3. 获取当前持仓
            position = self.get_position(symbol)
            qty = float(position.quantity) if position else 0
            current_market_value = qty * price

            # 4. 执行交易决策
            # 策略逻辑：金叉买入，死叉卖出

            if sma_fast_val > sma_slow_val:
                # -- Buy Signal --
                # 如果当前持仓不足预算，且还有现金，则买入
                if current_market_value < (budget * 0.9): 
                    # 计算还需要买多少金额
                    amount_to_buy = budget - current_market_value
                    shares_to_buy = int(amount_to_buy / price)
                    
                    # 再次检查账户实际剩余现金 (Global Cash Check)
                    cash = self.get_cash()
                    if shares_to_buy > 0 and (shares_to_buy * price) <= cash:
                        order = self.create_order(symbol, shares_to_buy, "buy")
                        self.submit_order(order)
                        
                        log_msg = f"BUY {symbol} +{shares_to_buy} @ {price:.2f} (SMA{self.sma_fast} > SMA{self.sma_slow})"
                        self.tm.log(f"[{self.get_datetime()}] {log_msg}")
                        self.tm.add_order({
                            "symbol": symbol, "action": "BUY", "quantity": shares_to_buy, 
                            "price": price, "status": "submitted", "timestamp": datetime.now().isoformat()
                        })

            elif sma_fast_val < sma_slow_val:
                # -- Sell Signal --
                # 目前逻辑是清仓
                if qty > 0:
                    order = self.create_order(symbol, qty, "sell")
                    self.submit_order(order)
                    
                    log_msg = f"SELL {symbol} -{qty} @ {price:.2f} (SMA{self.sma_fast} < SMA{self.sma_slow})"
                    self.tm.log(f"[{self.get_datetime()}] {log_msg}")
                    self.tm.add_order({
                        "symbol": symbol, "action": "SELL", "quantity": qty, 
                        "price": price, "status": "submitted", "timestamp": datetime.now().isoformat()
                    })
            
        except Exception as e:
            self.tm.log(f"Error processing {symbol}: {e}")

    def _portfolio_poller(self):
        """独立线程轮询资产状态，适配前端展示"""
        while not self._stop_poller.is_set():
            try:
                self._update_portfolio_state()
                time.sleep(5) # 5秒刷新一次
            except Exception:
                time.sleep(5)

    def _update_portfolio_state(self):
        """
        更新 TradeManager 的状态，供前端 WebSocket 使用
        """
        try:
            # 获取现金
            try:
                cash = float(self.broker.get_cash())
            except:
                cash = 0.0

            # 获取总资产
            try:
                portfolio_value = float(self.portfolio_value)
            except:
                portfolio_value = cash

            # 市场状态
            market_status = "unknown"
            try:
                if hasattr(self.broker, "get_clock"):
                    clock = self.broker.get_clock()
                    if clock:
                        market_status = "open" if clock.is_open else "closed"
            except:
                pass

            # 获取持仓详情
            positions_list = []
            try:
                # 获取所有持仓
                raw_positions = self.positions
                iterator = []
                if isinstance(raw_positions, dict):
                    iterator = raw_positions.values()
                elif isinstance(raw_positions, list):
                    iterator = raw_positions
                
                # 尝试从 Broker API 直接获取更详细的信息 (如 Alpaca)
                # 这里的逻辑主要是为了兼容性和获取更多字段
                # 简化处理：直接用力所能及的信息
                
                # DEBUG: 获取 Alpaca 原始数据
                try:
                    if hasattr(self.broker, "api"):
                        # 尝试调用 list_positions (alpaca-trade-api)
                        if hasattr(self.broker.api, "list_positions"):
                            raw_list = self.broker.api.list_positions()
                            self.tm.log(">>> DEBUG: RAW ALPACA API DATA <<<")
                            for p in raw_list:
                                # 尝试打印 _raw 字典，如果不存在则打印对象本身
                                content = getattr(p, "_raw", p)
                                self.tm.log(f"Symbol: {getattr(p, 'symbol', 'Unknown')} | Raw: {content}")
                            self.tm.log(">>> END RAW DATA <<<")
                except Exception as e:
                    # 避免在非 Alpaca broker 环境下报错
                    pass

                for position in iterator:
                    symbol = str(getattr(position, "asset", "UNKNOWN"))
                    quantity = float(getattr(position, "quantity", 0.0))
                    
                    # 尝试获取价格
                    current_price = 0.0
                    if hasattr(self, "get_last_price"):
                        try:
                            current_price = float(self.get_last_price(symbol))
                        except:
                            pass
                    
                    # 尝试获取成本
                    avg_price = float(getattr(position, "avg_entry_price", 0.0))
                    
                    # 计算盈亏
                    unrealized_pl = 0.0
                    unrealized_plpc = 0.0
                    
                    if hasattr(position, "unrealized_pl"):
                        unrealized_pl = float(position.unrealized_pl)
                    elif current_price > 0 and avg_price > 0:
                        unrealized_pl = (current_price - avg_price) * quantity
                        
                    if hasattr(position, "unrealized_plpc"):
                        unrealized_plpc = float(position.unrealized_plpc)
                    elif avg_price > 0:
                        unrealized_plpc = unrealized_pl / (avg_price * abs(quantity)) if quantity != 0 else 0

                    positions_list.append({
                        "symbol": symbol,
                        "quantity": quantity,
                        "average_price": avg_price,
                        "current_price": current_price,
                        "unrealized_pl": unrealized_pl,
                        "unrealized_plpc": unrealized_plpc,
                        "asset_class": "stock" # 简化
                    })

            except Exception as e:
                pass # 忽略持仓获取错误，避免刷屏

            # 更新到 TradeManager
            self.tm.update_portfolio(
                cash, portfolio_value, positions_list, market_status
            )

        except Exception as e:
            # self.tm.log(f"Portfolio update error: {e}") 
            pass

    def stop(self):
        if hasattr(self, "_stop_poller"):
            self._stop_poller.set()
        if hasattr(self, "_poller_thread"):
            self._poller_thread.join(timeout=1)
