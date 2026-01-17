from lumibot.strategies.strategy import Strategy


class MomentumStrategy(Strategy):
    """
    统一的策略类，实现了基于 SMA 的动量策略。
    此策略只负责核心交易逻辑，完全不感知前端或 TradeManager。
    
    所有状态更新和日志记录由外部 TradeManager 通过轮询策略实例状态来完成。
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
        if "symbols" in self.parameters:
            self.symbols = self.parameters["symbols"]
        elif "symbol" in self.parameters:
            self.symbols = [self.parameters["symbol"]]
        else:
            self.symbols = ["SPY"]

        # 确保格式统一为列表
        if isinstance(self.symbols, str):
            if "," in self.symbols:
                self.symbols = [s.strip() for s in self.symbols.split(",")]
            else:
                self.symbols = [self.symbols]

        self.sleeptime = self.parameters.get("sleeptime", "5S")
        self.lookback_period = self.parameters.get("lookback_period", 40)
        self.sma_fast = self.parameters.get("sma_fast", 10)
        self.sma_slow = self.parameters.get("sma_slow", 30)

        # 策略初始化完成日志（使用框架自带日志）
        self.log_message(f"Strategy initialized for symbols: {self.symbols}")

    def on_trading_iteration(self):
        """核心交易逻辑 - 支持多标的"""
        try:
            total_portfolio_value = self.portfolio_value
            if total_portfolio_value is None or total_portfolio_value == 0:
                total_portfolio_value = self.get_cash()
            
            # 等权重预算，预留 5% 现金缓冲
            target_allocation = (total_portfolio_value * 0.95) / len(self.symbols)

            for symbol in self.symbols:
                self._process_single_symbol(symbol, target_allocation)

        except Exception as e:
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
                # 数据不足，跳过
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

            # 4. 执行交易决策：金叉买入，死叉卖出
            if sma_fast_val > sma_slow_val:
                # Buy Signal - 如果仓位显著低于预算，则补仓
                if current_market_value < (budget * 0.9): 
                    amount_to_buy = budget - current_market_value
                    shares_to_buy = int(amount_to_buy / price)
                    
                    cash = self.get_cash()
                    if shares_to_buy > 0 and (shares_to_buy * price) <= cash:
                        self.log_message(f"BUY Signal for {symbol}: SMA Fast {sma_fast_val:.2f} > Slow {sma_slow_val:.2f}")
                        order = self.create_order(symbol, shares_to_buy, "buy")
                        self.submit_order(order)

            elif sma_fast_val < sma_slow_val:
                # Sell Signal - 死叉全额清仓
                if qty > 0:
                    self.log_message(f"SELL Signal for {symbol}: SMA Fast {sma_fast_val:.2f} < Slow {sma_slow_val:.2f}")
                    order = self.create_order(symbol, qty, "sell")
                    self.submit_order(order)
            
        except Exception as e:
            # 仅记录到框架日志
            self.log_message(f"Error processing {symbol}: {e}")
