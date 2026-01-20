"""
Model Inference Module.

Provides a unified interface for loading Qlib-trained models 
and performing predictions on DataFrame inputs.

Task: Implement ml/inference.py - Class to load Qlib model and accept DataFrame input
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from loguru import logger

from .model_manager import ModelManager, get_model_manager
from .trainer import LightGBMTrainer
from .features import QlibFeatureGenerator


class ModelInference:
    """
    Model inference wrapper for Qlib models.
    
    Provides a clean interface for:
    1. Loading trained models from the model directory
    2. Accepting raw OHLCV DataFrames
    3. Generating features and returning predictions
    
    Example:
        >>> inferencer = ModelInference()
        >>> predictions = inferencer.predict(df)  # df is OHLCV DataFrame
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        models_dir: Union[str, Path] = "models",
        auto_generate_features: bool = True,
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_name: Specific model to load. If None, uses the current model from ModelManager.
            models_dir: Directory containing saved models.
            auto_generate_features: If True, automatically generate Qlib features from raw OHLCV.
        """
        self.models_dir = Path(models_dir)
        self.model_name = model_name
        self.auto_generate_features = auto_generate_features
        
        # Components
        self.model_manager = ModelManager(models_dir)
        self.feature_generator = QlibFeatureGenerator(normalize=True)
        self.trainer: Optional[LightGBMTrainer] = None
        
        # Metadata
        self.model_path: Optional[Path] = None
        self.is_loaded: bool = False
        self.metadata: dict = {}
        
        # Attempt to load model
        self._load_model()
    
    def _load_model(self) -> bool:
        """
        Load the model from disk.
        
        Returns:
            True if successfully loaded, False otherwise.
        """
        try:
            # Determine model path
            if self.model_name:
                self.model_path = self.models_dir / self.model_name
            else:
                self.model_path = self.model_manager.get_current_model_path()
            
            if not self.model_path or not self.model_path.exists():
                logger.warning(f"Model path not found: {self.model_path}")
                return False
            
            # Load the trainer with the model
            self.trainer = LightGBMTrainer(model_dir=str(self.models_dir))
            self.trainer.load(self.model_path)
            
            # Extract metadata if available
            if hasattr(self.trainer, 'metadata'):
                self.metadata = self.trainer.metadata
            
            self.is_loaded = True
            logger.info(f"Model loaded successfully: {self.model_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_loaded = False
            return False
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions from OHLCV DataFrame.
        
        If auto_generate_features is True, this accepts raw OHLCV data
        and generates Qlib-compatible features internally.
        
        Args:
            df: Input DataFrame. Can be either:
                - Raw OHLCV data (columns: open, high, low, close, volume)
                - Pre-computed features (if auto_generate_features is False)
        
        Returns:
            Array of predictions (one per row in input).
        
        Raises:
            ValueError: If model is not loaded or input is invalid.
        """
        if not self.is_loaded or self.trainer is None:
            raise ValueError("Model not loaded. Call _load_model() first or check model path.")
        
        # Ensure column names are lowercase
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })
        
        # Generate features if needed
        if self.auto_generate_features:
            features = self.feature_generator._generate_single_symbol(df)
        else:
            features = df
        
        if features.empty:
            raise ValueError("Feature generation resulted in empty DataFrame")
        
        # Get the latest features for prediction (most common use case)
        # If full predictions are needed, use predict_all()
        return self.trainer.predict(features)
    
    def predict_latest(self, df: pd.DataFrame) -> float:
        """
        Get prediction for the most recent data point only.
        
        Useful for live trading where only the current signal matters.
        
        Args:
            df: OHLCV DataFrame with historical data.
        
        Returns:
            Single prediction value for the latest timestamp.
        """
        predictions = self.predict(df)
        return float(predictions[-1])
    
    def predict_batch(self, data_dict: dict[str, pd.DataFrame]) -> dict[str, float]:
        """
        Get predictions for multiple symbols.
        
        Args:
            data_dict: Dictionary mapping symbol -> OHLCV DataFrame.
        
        Returns:
            Dictionary mapping symbol -> prediction score.
        """
        results = {}
        
        for symbol, df in data_dict.items():
            try:
                pred = self.predict_latest(df)
                results[symbol] = pred
            except Exception as e:
                logger.warning(f"Prediction failed for {symbol}: {e}")
        
        return results
    
    def reload(self, model_name: Optional[str] = None) -> bool:
        """
        Reload the model (useful when a new model has been trained).
        
        Args:
            model_name: Optional new model name to load.
        
        Returns:
            True if reload successful.
        """
        if model_name:
            self.model_name = model_name
        return self._load_model()
    
    def get_model_info(self) -> dict:
        """
        Get information about the currently loaded model.
        
        Returns:
            Dictionary with model metadata.
        """
        return {
            "model_name": self.model_path.name if self.model_path else None,
            "model_path": str(self.model_path) if self.model_path else None,
            "is_loaded": self.is_loaded,
            "metadata": self.metadata,
        }


# Singleton instance for convenience
_inference_instance: Optional[ModelInference] = None


def get_inference_engine(
    model_name: Optional[str] = None,
    models_dir: Union[str, Path] = "models",
    force_new: bool = False,
) -> ModelInference:
    """
    Get a singleton ModelInference instance.
    
    Args:
        model_name: Specific model to load.
        models_dir: Models directory.
        force_new: If True, create a new instance even if one exists.
    
    Returns:
        ModelInference instance.
    """
    global _inference_instance
    
    if _inference_instance is None or force_new:
        _inference_instance = ModelInference(
            model_name=model_name,
            models_dir=models_dir,
        )
    
    return _inference_instance
