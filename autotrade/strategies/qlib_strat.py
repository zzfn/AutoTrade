"""
Qlib ML Strategy - Machine learning driven trading strategy.

This strategy uses Qlib-trained ML models to predict returns and make trading decisions.
It has been refactored to use the new `ml` and `data` modules for cleaner architecture.
"""
from typing import Optional

import numpy as np
import pandas as pd
from lumibot.strategies.strategy import Strategy
from lumibot.entities import Asset

# Import from new module locations
from autotrade.ml import QlibFeatureGenerator, LightGBMTrainer, ModelManager
from autotrade.ml.inference import ModelInference, get_inference_engine
from autotrade.data import lumibot_to_qlib


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
        "top_k": 3,
        "lookback_period": 2,  # Fetch 2 days of history (1min: ~780 bars)
        "sleeptime": "1D",
        "models_dir": "models",
        "position_sizing": "equal",  # "equal" or "weighted"
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
        
        # Data frequency: 'day', 'hour', or 'minute'
        self.interval = self.parameters.get("interval", "minute")
        if self.interval == "1min":
            self.interval = "minute"
        elif self.interval == "1h":
            self.interval = "hour"
        elif self.interval == "1d":
            self.interval = "day"

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

    def on_trading_iteration(self):
        """
        Trading iteration - core logic.

        Each iteration:
        1. Fetch data and generate predictions
        2. Select Top-K stocks
        3. Rebalance portfolio
        4. Update and persist state
        """
        try:
            current_time = self.get_datetime()
            # Get predictions
            predictions = self._get_predictions()
            self.log_message(f"wwwwwwwww===Current predictions: {predictions}")

            if not predictions:
                self.log_message("Unable to get predictions, skipping this iteration")
                return

            # Select Top-K
            top_symbols = self._select_top_k(predictions)
            self.log_message(f"Top-{self.top_k} stocks: {top_symbols}")

            # Execute trades (pass predictions for weighted allocation)
            self._rebalance_portfolio(top_symbols, predictions)

            # Update state
            self.current_predictions = predictions
            
            # ========== Persist state to DB ==========
            self._update_persistent_state(predictions, top_symbols, current_time)

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

    def _get_predictions(self) -> dict:
        """
        Get prediction scores for all candidate stocks.

        Returns:
            {symbol: predicted_return} dictionary
        """
        predictions = {}

        # Create Asset objects for batch fetching
        assets = [Asset(symbol=s) for s in self.symbols]

        # Batch fetch historical data for all assets
        try:
            histories = self.get_historical_prices_for_assets(
                assets, length=self.lookback_period, timestep=self.interval
            )
        except Exception as e:
            self.log_message(f"Failed to fetch batch history: {e}")
            return predictions

        if not histories:
            self.log_message("No historical data returned")
            return predictions

        # Process each symbol
        # histories is expected to be a dict {asset: Bars/DataFrame}
        for asset, history in histories.items():
            symbol = asset.symbol
            try:
                if history is None:
                    continue

                # Check for empty data
                if hasattr(history, 'df') and history.df.empty:
                    continue
                if isinstance(history, pd.DataFrame) and history.empty:
                    continue

                # Use adapter to standardize data format
                df = lumibot_to_qlib(history, symbol=symbol)

                if len(df) < 30:  # Need enough data for features
                    self.log_message(f"Insufficient data for {symbol}: {len(df)} bars")
                    continue

                # Generate features
                features = self._generate_features(df)

                if features is None or features.empty:
                    continue

                # Get latest features
                latest_features = features.iloc[[-1]]

                # Predict
                if self.inference_engine is not None and self.inference_engine.is_loaded:
                    # Use inference engine (which uses the trainer internally)
                    pred = self.trainer.predict(latest_features)[0]
                elif self.trainer is not None:
                    # Legacy: direct trainer usage
                    pred = self.trainer.predict(latest_features)[0]
                else:
                    # No model - use momentum as proxy
                    pred = df["close"].pct_change(5).iloc[-1]

                predictions[symbol] = float(pred)

            except Exception as e:
                self.log_message(f"Prediction failed for {symbol}: {e}")

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
        Select Top-K stocks with highest prediction scores.

        Args:
            predictions: {symbol: score} dictionary

        Returns:
            List of Top-K stock symbols
        """
        if not predictions:
            return []

        # Sort by score
        sorted_symbols = sorted(
            predictions.keys(), key=lambda x: predictions[x], reverse=True
        )

        return sorted_symbols[: self.top_k]

    def _calculate_target_weights(self, predictions: dict, target_symbols: list) -> dict:
        """
        Calculate target weights for each symbol based on position_sizing mode.

        Args:
            predictions: {symbol: predicted_return} dictionary
            target_symbols: List of target stock symbols

        Returns:
            {symbol: weight} dictionary where weights sum to 1.0
        """
        if not target_symbols:
            return {}

        if self.position_sizing == "weighted":
            # Get prediction scores for target symbols
            scores = np.array([predictions.get(s, 0.0) for s in target_symbols])
            
            # Linear normalization: shift scores to be positive, then normalize
            # w_i = (s_i - min + epsilon) / sum(s_j - min + epsilon)
            min_score = np.min(scores)
            # Shift to make all scores positive (add small epsilon to avoid zero weights)
            shifted_scores = scores - min_score + 1e-6
            weights = shifted_scores / np.sum(shifted_scores)
            
            return {symbol: float(w) for symbol, w in zip(target_symbols, weights)}
        else:
            # Equal weighting (default)
            weight = 1.0 / len(target_symbols)
            return {symbol: weight for symbol in target_symbols}

    def _rebalance_portfolio(self, target_symbols: list, predictions: dict = None):
        """
        Rebalance portfolio to target holdings.

        This method uses get_positions() to identify ALL currently held positions,
        ensuring any "orphan" positions (positions not in self.symbols) are also
        handled and sold if not in the target list.

        Args:
            target_symbols: List of target stock symbols
            predictions: Optional predictions dict for weighted allocation
        """
        # Get ALL current positions using get_positions() to catch orphan positions
        all_positions = self.get_positions()
        current_positions = {}
        for pos in all_positions:
            symbol = pos.asset.symbol if hasattr(pos.asset, 'symbol') else str(pos.asset)
            qty = float(pos.quantity)
            if qty > 0:
                current_positions[symbol] = qty

        # Determine stocks to sell (includes orphan positions not in self.symbols)
        to_sell = set(current_positions.keys()) - set(target_symbols)

        # Log if we found orphan positions
        orphans = to_sell - set(self.symbols)
        if orphans:
            self.log_message(f"Found orphan positions to sell: {orphans}")

        # Sell positions not in target
        for symbol in to_sell:
            qty = current_positions[symbol]
            self.log_message(f"Selling {symbol}: {qty} shares")
            order = self.create_order(symbol, qty, "sell")
            self.submit_order(order)

        # Calculate target amounts
        total_value = self.portfolio_value or self.get_cash()
        if total_value <= 0:
            return

        # Reserve 5% cash buffer
        available_capital = total_value * 0.95

        # Calculate target weights
        if predictions is None:
            predictions = self.current_predictions or {}
        weights = self._calculate_target_weights(predictions, target_symbols)

        # Buy/adjust based on weights
        for symbol in target_symbols:
            try:
                price = self.get_last_price(symbol)
                if price is None or price <= 0:
                    continue

                # Calculate target allocation for this symbol
                weight = weights.get(symbol, 0.0)
                target_value = available_capital * weight
                target_qty = int(target_value / price)

                # Get current position
                current_qty = current_positions.get(symbol, 0)

                # Calculate difference
                diff = target_qty - current_qty

                if diff > 0:
                    # Need to buy
                    cash = self.get_cash()
                    max_buyable = int(cash / price)
                    buy_qty = min(diff, max_buyable)

                    if buy_qty > 0:
                        self.log_message(
                            f"Buying {symbol}: {buy_qty} shares @ ${price:.2f} (weight: {weight:.2%})"
                        )
                        order = self.create_order(symbol, buy_qty, "buy")
                        self.submit_order(order)

                elif diff < 0:
                    # Need to sell some
                    sell_qty = abs(diff)
                    if sell_qty > 0:
                        self.log_message(
                            f"Reducing {symbol}: {sell_qty} shares @ ${price:.2f}"
                        )
                        order = self.create_order(symbol, sell_qty, "sell")
                        self.submit_order(order)

            except Exception as e:
                self.log_message(f"Failed to adjust position for {symbol}: {e}")

    def get_prediction_summary(self) -> dict:
        """Get current prediction summary (for frontend display)."""
        return {
            "predictions": self.current_predictions,
            "top_k": self.top_k,
            "model_loaded": self.inference_engine is not None and self.inference_engine.is_loaded,
        }
