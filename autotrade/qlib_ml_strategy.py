"""
Qlib ML Strategy - Machine learning driven trading strategy.

This strategy uses Qlib-trained ML models to predict returns and make trading decisions.
It has been refactored to use the new `ml` and `data` modules for cleaner architecture.
"""
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from lumibot.strategies.strategy import Strategy

# 应用 Lumibot 补丁：禁用云功能
from autotrade.lumibot_patches import patch_strategy_to_disable_cloud
patch_strategy_to_disable_cloud()

# Suppress SettingWithCopyWarning from lumibot's bars.py
# This is a third-party library issue, not affecting functionality
warnings.filterwarnings('ignore', message='.*SettingWithCopyWarning.*', module='lumibot.*')
pd.options.mode.chained_assignment = None
from lumibot.entities import Asset

# Import from new module locations
from autotrade.ml import QlibFeatureGenerator, LightGBMTrainer, ModelManager
from autotrade.ml.inference import ModelInference
from autotrade.data import lumibot_to_qlib

# Import patched backtesting classes (修复多分钟 timestep 支持)
from autotrade.lumibot_patches import MyAlpacaBacktesting


class QlibMLStrategy(Strategy):
    """
    Qlib ML Strategy.

    A fully ML-model driven trading strategy:
    1. Fetch historical data for candidate stocks
    2. Generate Qlib-compatible features
    3. Use ML model to predict returns
    4. Select Top-K stocks with highest predicted scores
    5. Allocate capital equally
    6. Execute trades

    Strategy Parameters:
        symbols: List of candidate stocks
        model_name: Model name to use (None = use current model from ModelManager)
        top_k: Number of stocks to hold
        lookback_period: Days of historical data to fetch
        sleeptime: Trading iteration interval
    """

    parameters = {
        "symbols": ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN"],
        "model_name": None,  # None means use ModelManager's current model
        "top_k": 4,
        "lookback_period": 2,  # Fetch 2 days of history (1min: ~780 bars)
        "sleeptime": "1D",
        "models_dir": "models",
        "position_sizing": "weighted",  # "equal" or "weighted"
    }

    def initialize(self):
        """Strategy initialization."""
        # Parse parameters
        self._parse_parameters()

        # Initialize components using new modules
        self.feature_generator = QlibFeatureGenerator(normalize=True)
        self.model_manager = ModelManager(self.models_dir)
        self.inference_engine: Optional[ModelInference] = None

        # Legacy compatibility
        self.trainer: Optional[LightGBMTrainer] = None

        # State tracking
        self.current_predictions: dict = {}

        # Load model
        self._load_model()

        # Initialize persistent state (self.vars)
        # This will be automatically backed up to DB by Lumibot
        self._init_persistent_state()

        self.log_message(f"QlibMLStrategy initialized")
        self.log_message(f"Stock pool: {self.symbols}")
        self.log_message(f"Top-K: {self.top_k}")

        # Log restored state if any
        if self.vars.get("trade_cycle_count", 0) > 0:
            self.log_message(f"Restored state: cycle={self.vars.trade_cycle_count}, "
                           f"positions={len(self.vars.get('entered_positions', {}))}")

    def _init_persistent_state(self):
        """
        Initialize persistent state variables.
        
        self.vars is automatically backed up to DB by Lumibot.
        On restart, previously saved values are restored.
        """
        # Define default state structure
        default_state = {
            "entered_positions": {},      # {symbol: {"entry_price": float, "entry_time": str}}
            "stop_loss_levels": {},       # {symbol: float}
            "take_profit_levels": {},     # {symbol: float}
            "trade_cycle_count": 0,       # Total trading iterations
            "last_rebalance_time": None,  # ISO format datetime string
            "current_predictions": {},    # Latest predictions {symbol: score}
            "last_top_k": [],             # Last selected Top-K symbols
        }
        
        # Merge defaults with restored state (restored values take precedence)
        # Use hasattr() to check if attribute exists on self.vars
        for key, default_value in default_state.items():
            if not hasattr(self.vars, key):
                setattr(self.vars, key, default_value)

    def _parse_parameters(self):
        """Parse strategy parameters."""
        # Stock list
        if "symbols" in self.parameters:
            self.symbols = self.parameters["symbols"]
        elif "symbol" in self.parameters:
            self.symbols = [self.parameters["symbol"]]
        else:
            self.symbols = ["SPY"]

        # Ensure format is list
        if isinstance(self.symbols, str):
            if "," in self.symbols:
                self.symbols = [s.strip() for s in self.symbols.split(",")]
            else:
                self.symbols = [self.symbols]

        # Other parameters
        self.model_name = self.parameters.get("model_name")
        self.top_k = self.parameters.get("top_k", 3)
        self.lookback_period = self.parameters.get("lookback_period", 60)
        self.sleeptime = self.parameters.get("sleeptime", "1D")
        self.models_dir = self.parameters.get("models_dir", "models")
        self.position_sizing = self.parameters.get("position_sizing", "equal")

        # Validate position_sizing
        if self.position_sizing not in ("equal", "weighted"):
            self.position_sizing = "equal"
        
        # Data frequency: Alpaca supports 'minute', '5 minutes', '15 minutes',
        # '30 minutes', 'hour', '2 hours', '4 hours', 'day'
        self.interval = self.parameters.get("interval", "minute")
        
        # Create TimeFrame object for 5-minute data (bypass Lumibot's buggy TIMESTEP_MAPPING)
        # Lumibot's "5 minutes" maps to "51Min" instead of "5Min" due to a bug
        self.timeframe = '5 minute'

        # Ensure top_k doesn't exceed stock count
        self.top_k = min(self.top_k, len(self.symbols))

    def _load_model(self):
        """Load ML model using new inference engine."""
        try:
            # Initialize inference engine
            self.inference_engine = ModelInference(
                model_name=self.model_name,
                models_dir=self.models_dir,
                auto_generate_features=False,  # We generate features ourselves
            )
            
            # Check if model loaded successfully
            if self.inference_engine.is_loaded:
                # Sync interval from model metadata if available
                metadata = self.inference_engine.metadata
                if 'interval' in metadata:
                    model_interval = metadata['interval']
                    if model_interval == "1min":
                        self.interval = "minute"
                    elif model_interval == "1h":
                        self.interval = "hour"
                    elif model_interval == "1d":
                        self.interval = "day"
                
                # Keep trainer reference for legacy compatibility
                self.trainer = self.inference_engine.trainer
                
                self.log_message(f"Model loaded: {self.inference_engine.model_path.name} (Interval: {self.interval})")
            else:
                self.log_message("Warning: No ML model found, will use momentum proxy")

        except Exception as e:
            self.log_message(f"Failed to load model: {e}")
            self.inference_engine = None
            self.trainer = None

    def _has_pending_orders(self) -> bool:
        """
        检查是否有待处理的订单。
        
        Returns:
            True 如果有未完成订单，否则 False
        """
        try:
            orders = self.get_orders()
            pending = [o for o in orders if o.status in ("open", "pending", "new")]
            if pending:
                self.log_message(f"有 {len(pending)} 个待处理订单")
                return True
            return False
        except Exception as e:
            self.log_message(f"检查订单状态失败: {e}")
            return False

    def on_trading_iteration(self):
        """
        Trading iteration - 核心逻辑。
        
        每次迭代按以下 5 个步骤执行：
        1. 获取时间
        2. 检查订单
        3. 获取数据
        4. 计算逻辑
        5. 执行下单
        """
        try:
            # ===== Step 1: 获取时间 =====
            current_time = self.get_datetime()
            self.log_message(f"=== Trading iteration at {current_time} ===")
            
            # ===== Step 2: 检查订单 =====
            if self._has_pending_orders():
                self.log_message("有待处理订单，跳过本次迭代")
                return
            
            # ===== Step 3: 获取数据 =====
            market_data = self._fetch_market_data()
            if market_data is None:
                self.log_message("无法获取市场数据，跳过本次迭代")
                return
            
            # ===== Step 4: 计算逻辑 =====
            predictions = self._compute_predictions(market_data)
            if not predictions:
                self.log_message("无法获取预测，跳过本次迭代")
                return
            
            self.log_message(f"预测结果: {predictions}")
            
            # ===== Step 5: 执行下单 =====
            top_symbols = self._select_top_k(predictions)
            self.log_message(f"Top-{self.top_k} 股票: {top_symbols}")
            
            self._execute_trades(top_symbols, predictions, current_time)
            
        except Exception as e:
            import traceback
            self.log_message(f"Trading iteration error: {e}")
            traceback.print_exc()
    
    def _update_persistent_state(self, predictions: dict, top_symbols: list, current_time):
        """
        Update self.vars with current state after each trading iteration.
        
        Lumibot automatically backs up self.vars to the database.
        """
        # Increment trade cycle count
        self.vars.trade_cycle_count = self.vars.get("trade_cycle_count", 0) + 1
        
        # Save current predictions
        self.vars.current_predictions = predictions
        self.vars.last_top_k = top_symbols
        self.vars.last_rebalance_time = current_time.isoformat() if current_time else None
        
        # Get current position dicts (or empty dicts if not set)
        entered_positions = self.vars.get("entered_positions", {})
        stop_loss_levels = self.vars.get("stop_loss_levels", {})
        take_profit_levels = self.vars.get("take_profit_levels", {})
        
        # Update entered positions with current prices
        for symbol in top_symbols:
            if symbol not in entered_positions:
                try:
                    price = self.get_last_price(symbol)
                    if price and price > 0:
                        entered_positions[symbol] = {
                            "entry_price": float(price),
                            "entry_time": current_time.isoformat() if current_time else None,
                        }
                        # Set default stop loss at 5% below entry
                        stop_loss_levels[symbol] = float(price) * 0.95
                        # Set default take profit at 10% above entry
                        take_profit_levels[symbol] = float(price) * 1.10
                except Exception:
                    pass
        
        # Remove positions no longer in top_k
        symbols_to_remove = [s for s in list(entered_positions.keys()) if s not in top_symbols]
        for symbol in symbols_to_remove:
            entered_positions.pop(symbol, None)
            stop_loss_levels.pop(symbol, None)
            take_profit_levels.pop(symbol, None)
        
        # Save back to vars
        self.vars.entered_positions = entered_positions
        self.vars.stop_loss_levels = stop_loss_levels
        self.vars.take_profit_levels = take_profit_levels

    def _fetch_market_data(self) -> Optional[dict]:
        """
        获取所有候选股票的历史数据。

        Returns:
            {symbol: DataFrame} 字典，如果失败返回 None
        """
        assets = [Asset(symbol=s) for s in self.symbols]

        # 计算需要的 K 线数量（5 分钟数据每天 78 根）
        bars_per_day = 78  # 390 / 5 = 78
        min_required_bars = 30  # 至少需要 30 根 5 分钟 K 线
        requested_bars = max(int(self.lookback_period * bars_per_day), min_required_bars)

        try:
            histories = self.get_historical_prices_for_assets文件(
                assets, requested_bars, "5 minutes"
            )
        except Exception as e:
            self.log_message(f"批量获取历史数据失败: {e}")
            return None

        if not histories:
            self.log_message("未返回历史数据")
            return None

        # 转换为标准格式
        market_data = {}
        for asset, history in histories.items():
            symbol = asset.symbol
            try:
                if history is None:
                    continue
                if hasattr(history, 'df') and history.df.empty:
                    continue
                if isinstance(history, pd.DataFrame) and history.empty:
                    continue

                # 使用适配器标准化数据格式
                df = lumibot_to_qlib(history, symbol=symbol)

                # 不再需要 resample，因为直接获取的就是 5 分钟数据
                if len(df) >= 30:
                    market_data[symbol] = df
                else:
                    self.log_message(f"{symbol} 数据不足: {len(df)} bars")

            except Exception as e:
                self.log_message(f"{symbol} 数据处理失败: {e}")

        return market_data if market_data else None

    def _compute_predictions(self, market_data: dict) -> dict:
        """
        基于市场数据计算预测分数。
        
        Args:
            market_data: {symbol: DataFrame} 字典
            
        Returns:
            {symbol: predicted_return} 字典
        """
        predictions = {}
        
        for symbol, df in market_data.items():
            try:
                # 生成特征
                features = self._generate_features(df)
                if features is None or features.empty:
                    continue
                
                # 获取最新特征
                latest_features = features.iloc[[-1]]
                
                # 预测
                if self.inference_engine is not None and self.inference_engine.is_loaded:
                    pred = self.trainer.predict(latest_features)[0]
                elif self.trainer is not None:
                    pred = self.trainer.predict(latest_features)[0]
                else:
                    # 无模型时使用动量代理
                    pred = df["close"].pct_change(5).iloc[-1]
                
                predictions[symbol] = float(pred)
                
            except Exception as e:
                self.log_message(f"{symbol} 预测失败: {e}")
        
        return predictions

    def _generate_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Generate features for a single stock."""
        try:
            # Use feature generator
            features = self.feature_generator._generate_single_symbol(df)
            return features

        except Exception as e:
            self.log_message(f"Feature generation failed: {e}")
            return None

    def _select_top_k(self, predictions: dict) -> list:
        """
        Select Top-K stocks with highest absolute prediction scores.

        Args:
            predictions: {symbol: score} dictionary

        Returns:
            List of Top-K stock symbols (sorted by absolute score, descending)
        """
        if not predictions:
            return []

        # Sort by absolute score
        sorted_symbols = sorted(
            predictions.keys(),
            key=lambda x: abs(predictions[x]),
            reverse=True
        )

        return sorted_symbols[: self.top_k]

    def _calculate_target_weights(self, predictions: dict, target_symbols: list) -> dict:
        """
        Calculate target weights for each symbol based on absolute prediction scores.

        Args:
            predictions: {symbol: predicted_return} dictionary
            target_symbols: List of target stock symbols

        Returns:
            {symbol: weight} dictionary where weights are positive and sum to 1.0
            The sign of the prediction determines long (positive) or short (negative)
        """
        if not target_symbols:
            return {}

        # Get absolute prediction scores for target symbols
        abs_scores = np.array([abs(predictions.get(s, 0.0)) for s in target_symbols])

        if self.position_sizing == "weighted":
            # Weight by absolute prediction scores
            # w_i = |s_i| / sum(|s_j|)
            total_abs_score = np.sum(abs_scores) + 1e-6  # Avoid division by zero
            weights = abs_scores / total_abs_score

            return {symbol: float(w) for symbol, w in zip(target_symbols, weights)}
        else:
            # Equal weighting
            weight = 1.0 / len(target_symbols)
            return {symbol: weight for symbol in target_symbols}

    def _get_current_positions(self) -> dict:
        """
        获取当前所有持仓。
        
        Returns:
            {symbol: quantity} 字典
        """
        all_positions = self.get_positions()
        positions = {}
        for pos in all_positions:
            symbol = pos.asset.symbol if hasattr(pos.asset, 'symbol') else str(pos.asset)
            positions[symbol] = float(pos.quantity)
        return positions

    def _calculate_order_plan(self, target_symbols: list, predictions: dict) -> list:
        """
        计算需要执行的订单列表。
        
        Returns:
            订单信息列表 [{"symbol": str, "qty": int, "side": str, "reason": str}, ...]
        """
        order_plan = []
        
        # 获取当前持仓
        current_positions = self._get_current_positions()
        
        # 1. 计算需要平仓的股票
        to_close = set(current_positions.keys()) - set(target_symbols)
        
        # 记录孤儿仓位
        orphans = to_close - set(self.symbols)
        if orphans:
            self.log_message(f"发现孤儿仓位需平仓: {orphans}")
        
        for symbol in to_close:
            qty = current_positions[symbol]
            if qty > 0:
                order_plan.append({
                    "symbol": symbol, "qty": qty, 
                    "side": "sell", "reason": "close_long"
                })
            elif qty < 0:
                order_plan.append({
                    "symbol": symbol, "qty": abs(qty), 
                    "side": "buy", "reason": "close_short"
                })
        
        # 2. 计算目标权重和仓位
        total_value = self.portfolio_value or self.get_cash()
        if total_value <= 0:
            return order_plan
            
        available_capital = total_value * 0.95  # 保留 5% 现金缓冲
        weights = self._calculate_target_weights(predictions, target_symbols)
        
        # 3. 计算每个目标股票的订单
        for symbol in target_symbols:
            try:
                price = self.get_last_price(symbol)
                if not price or price <= 0:
                    continue
                
                weight = weights.get(symbol, 0.0)
                target_value = available_capital * weight
                base_qty = int(target_value / price)
                
                # 根据预测方向决定多空
                prediction = predictions.get(symbol, 0.0)
                target_qty = base_qty if prediction >= 0 else -base_qty
                direction = "LONG" if prediction >= 0 else "SHORT"
                
                current_qty = current_positions.get(symbol, 0)
                diff = target_qty - current_qty
                
                if diff == 0:
                    continue
                
                # 处理仓位转换（从多转空或从空转多）
                if current_qty < 0 and target_qty > 0:
                    # 先平空再开多
                    order_plan.append({
                        "symbol": symbol, "qty": abs(current_qty),
                        "side": "buy", "reason": "cover_short"
                    })
                    order_plan.append({
                        "symbol": symbol, "qty": target_qty,
                        "side": "buy", "reason": f"open_long ({direction}, weight: {weight:.2%}, pred: {prediction:.4f})"
                    })
                elif current_qty > 0 and target_qty < 0:
                    # 先平多再开空
                    order_plan.append({
                        "symbol": symbol, "qty": current_qty,
                        "side": "sell", "reason": "close_long"
                    })
                    order_plan.append({
                        "symbol": symbol, "qty": abs(target_qty),
                        "side": "sell", "reason": f"open_short ({direction}, weight: {weight:.2%}, pred: {prediction:.4f})"
                    })
                elif diff > 0:
                    # 需要买入更多
                    if target_qty > 0:
                        # 增加多头仓位
                        cash = self.get_cash()
                        max_buyable = int(cash / price)
                        buy_qty = min(diff, max_buyable)
                        if buy_qty > 0:
                            order_plan.append({
                                "symbol": symbol, "qty": buy_qty,
                                "side": "buy", "reason": f"add_long ({direction}, weight: {weight:.2%}, pred: {prediction:.4f})"
                            })
                    else:
                        # 减少空头仓位
                        cover_qty = min(diff, abs(current_qty))
                        if cover_qty > 0:
                            order_plan.append({
                                "symbol": symbol, "qty": cover_qty,
                                "side": "buy", "reason": "reduce_short"
                            })
                elif diff < 0:
                    # 需要卖出更多
                    if target_qty >= 0:
                        # 减少多头仓位
                        sell_qty = abs(diff)
                        if sell_qty > 0:
                            order_plan.append({
                                "symbol": symbol, "qty": sell_qty,
                                "side": "sell", "reason": "reduce_long"
                            })
                    else:
                        # 增加空头仓位
                        short_qty = abs(diff)
                        if short_qty > 0:
                            order_plan.append({
                                "symbol": symbol, "qty": short_qty,
                                "side": "sell", "reason": f"add_short ({direction}, weight: {weight:.2%}, pred: {prediction:.4f})"
                            })
                            
            except Exception as e:
                self.log_message(f"计算 {symbol} 订单失败: {e}")
        
        return order_plan

    def _submit_single_order(self, order_info: dict):
        """
        提交单个订单并记录日志。
        """
        symbol = order_info["symbol"]
        qty = order_info["qty"]
        side = order_info["side"]
        reason = order_info["reason"]
        
        if qty <= 0:
            return
        
        try:
            price = self.get_last_price(symbol)
            self.log_message(f"{reason}: {side} {symbol} {qty} shares @ ${price:.2f}")
            
            order = self.create_order(symbol, qty, side)
            self.submit_order(order)
            
        except Exception as e:
            self.log_message(f"提交订单失败 {symbol}: {e}")

    def _execute_trades(self, target_symbols: list, predictions: dict, current_time):
        """
        执行交易并更新状态。
        
        Args:
            target_symbols: 目标持仓股票列表
            predictions: 预测分数字典
            current_time: 当前时间
        """
        # 计算订单计划
        order_plan = self._calculate_order_plan(target_symbols, predictions)
        
        self.log_message(f"订单计划: {len(order_plan)} 个订单")
        
        # 执行订单
        for order_info in order_plan:
            self._submit_single_order(order_info)
        
        # 更新持久化状态
        self.current_predictions = predictions
        self._update_persistent_state(predictions, target_symbols, current_time)

    def get_prediction_summary(self) -> dict:
        """Get current prediction summary (for frontend display)."""
        return {
            "predictions": self.current_predictions,
            "top_k": self.top_k,
            "model_loaded": self.inference_engine is not None and self.inference_engine.is_loaded,
        }
