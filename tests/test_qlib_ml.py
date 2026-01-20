"""
Qlib ML 策略模块测试

包含数据适配、特征工程、模型训练和策略的单元测试
"""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ==================== 数据适配层测试 ====================


class TestQlibDataAdapter:
    """QlibDataAdapter 测试"""

    @pytest.fixture
    def adapter(self, tmp_path):
        """创建临时数据目录的适配器"""
        from autotrade.research.data import QlibDataAdapter

        return QlibDataAdapter(data_dir=tmp_path / "qlib")

    @pytest.fixture
    def sample_df(self):
        """创建示例 OHLCV 数据"""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        data = {
            "open": np.random.uniform(100, 110, 100),
            "high": np.random.uniform(110, 120, 100),
            "low": np.random.uniform(90, 100, 100),
            "close": np.random.uniform(100, 110, 100),
            "volume": np.random.randint(1000000, 5000000, 100),
        }
        df = pd.DataFrame(data, index=dates)
        df["symbol"] = "AAPL"
        df = df.reset_index().rename(columns={"index": "timestamp"})
        df = df.set_index(["timestamp", "symbol"])
        return df

    def test_adapter_creates_directories(self, adapter):
        """测试适配器创建必要目录"""
        assert adapter.data_dir.exists()
        assert adapter.features_dir.exists()
        assert adapter.instruments_dir.exists()
        assert adapter.calendars_dir.exists()

    def test_store_and_load_data(self, adapter, sample_df):
        """测试数据存储和加载"""
        # 模拟 provider
        mock_provider = MagicMock()
        mock_provider.fetch_data.return_value = sample_df
        adapter._provider = mock_provider

        # 存储
        result = adapter.fetch_and_store(
            symbols=["AAPL"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 4, 10),
        )

        assert result["status"] == "success"
        assert "AAPL" in result["processed_symbols"]

        # 加载
        loaded = adapter.load_data(["AAPL"])
        assert not loaded.empty
        assert "close" in loaded.columns

    def test_get_available_symbols(self, adapter, sample_df):
        """测试获取可用股票列表"""
        mock_provider = MagicMock()
        mock_provider.fetch_data.return_value = sample_df
        adapter._provider = mock_provider

        adapter.fetch_and_store(
            symbols=["AAPL"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 4, 10),
        )

        symbols = adapter.get_available_symbols()
        assert "AAPL" in symbols


# ==================== 特征工程测试 ====================


class TestQlibFeatureGenerator:
    """QlibFeatureGenerator 测试"""

    @pytest.fixture
    def generator(self):
        from autotrade.research.features import QlibFeatureGenerator

        return QlibFeatureGenerator(normalize=False)

    @pytest.fixture
    def sample_ohlcv(self):
        """创建示例 OHLCV 数据"""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        return pd.DataFrame(
            {
                "open": close - np.random.uniform(0, 1, 100),
                "high": close + np.random.uniform(0, 2, 100),
                "low": close - np.random.uniform(0, 2, 100),
                "close": close,
                "volume": np.random.randint(1000000, 5000000, 100),
            },
            index=dates,
        )

    def test_generate_features(self, generator, sample_ohlcv):
        """测试特征生成"""
        features = generator.generate(sample_ohlcv)

        assert not features.empty
        assert len(features) == len(sample_ohlcv)

        # 检查关键特征存在
        assert "$close" in features.columns
        assert "$return_1d" in features.columns
        assert "$rsi_6" in features.columns
        assert "$macd" in features.columns

    def test_no_nan_in_output(self, generator, sample_ohlcv):
        """测试输出无 NaN（经过填充处理后）"""
        features = generator.generate(sample_ohlcv)
        assert not features.isna().any().any()

    def test_get_feature_names(self, generator):
        """测试获取特征名称"""
        names = generator.get_feature_names()
        assert len(names) > 0
        assert all(n.startswith("$") for n in names)


class TestFeaturePreprocessor:
    """FeaturePreprocessor 测试"""

    @pytest.fixture
    def preprocessor(self):
        from autotrade.research.features import FeaturePreprocessor

        return FeaturePreprocessor(normalize_method="zscore")

    def test_fit_transform(self, preprocessor):
        """测试拟合和转换"""
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [10, 20, 30, 40, 50],
            }
        )

        result = preprocessor.fit_transform(df)

        # Z-score 标准化后均值接近 0
        assert abs(result["a"].mean()) < 0.01
        assert abs(result["b"].mean()) < 0.01


# ==================== 模型训练测试 ====================


class TestLightGBMTrainer:
    """LightGBMTrainer 测试"""

    @pytest.fixture
    def trainer(self, tmp_path):
        from autotrade.research.models import LightGBMTrainer

        return LightGBMTrainer(
            model_dir=tmp_path / "models",
            model_name="test_model",
            num_boost_round=10,  # 快速测试
        )

    @pytest.fixture
    def sample_data(self):
        """创建示例训练数据"""
        np.random.seed(42)
        n_samples = 500

        X = pd.DataFrame(
            {
                "feature_1": np.random.randn(n_samples),
                "feature_2": np.random.randn(n_samples),
                "feature_3": np.random.randn(n_samples),
            }
        )

        # 目标变量与特征相关
        y = pd.Series(
            0.3 * X["feature_1"]
            + 0.5 * X["feature_2"]
            + np.random.randn(n_samples) * 0.1
        )

        return X, y

    def test_train_and_predict(self, trainer, sample_data):
        """测试训练和预测"""
        X, y = sample_data

        # 分割数据
        split = int(len(X) * 0.8)
        X_train, X_valid = X.iloc[:split], X.iloc[split:]
        y_train, y_valid = y.iloc[:split], y.iloc[split:]

        # 训练
        trainer.train(X_train, y_train, X_valid, y_valid)

        assert trainer.model is not None

        # 预测
        preds = trainer.predict(X_valid)
        assert len(preds) == len(X_valid)

    def test_save_and_load(self, trainer, sample_data, tmp_path):
        """测试模型保存和加载"""
        from autotrade.research.models import LightGBMTrainer

        X, y = sample_data
        trainer.train(X, y)

        # 保存
        model_path = trainer.save("v1")
        assert model_path.exists()
        assert (model_path / "model.pkl").exists()
        assert (model_path / "metadata.json").exists()

        # 加载
        new_trainer = LightGBMTrainer(model_dir=tmp_path / "models")
        new_trainer.load(model_path)

        # 验证加载后能预测
        preds = new_trainer.predict(X)
        assert len(preds) == len(X)

    def test_evaluate(self, trainer, sample_data):
        """测试模型评估"""
        X, y = sample_data
        trainer.train(X, y)

        metrics = trainer.evaluate(X, y)

        assert "ic" in metrics
        assert "mse" in metrics
        assert "mae" in metrics
        assert metrics["ic"] > 0  # 应该有正相关


class TestModelManager:
    """ModelManager 测试"""

    @pytest.fixture
    def manager(self, tmp_path):
        from autotrade.research.models import ModelManager

        return ModelManager(models_dir=tmp_path / "models")

    @pytest.fixture
    def create_mock_model(self, tmp_path):
        """创建模拟模型"""
        import json
        import pickle

        def _create(name):
            model_dir = tmp_path / "models" / name
            model_dir.mkdir(parents=True)

            # 创建模拟模型文件
            with open(model_dir / "model.pkl", "wb") as f:
                pickle.dump({"dummy": "model"}, f)

            # 创建元数据
            with open(model_dir / "metadata.json", "w") as f:
                json.dump(
                    {
                        "model_name": name,
                        "model_type": "LightGBMTrainer",
                        "saved_at": datetime.now().isoformat(),
                        "ic": 0.05,
                        "train_samples": 1000,
                    },
                    f,
                )

            return model_dir

        return _create

    def test_list_models(self, manager, create_mock_model):
        """测试列出模型"""
        create_mock_model("model_v1")
        create_mock_model("model_v2")

        models = manager.list_models()

        assert len(models) == 2
        names = [m["name"] for m in models]
        assert "model_v1" in names
        assert "model_v2" in names

    def test_set_and_get_current(self, manager, create_mock_model):
        """测试设置和获取当前模型"""
        create_mock_model("model_v1")

        # 设置
        success = manager.set_current_model("model_v1")
        assert success

        # 获取
        current = manager.get_current_model()
        assert current == "model_v1"

    def test_delete_model(self, manager, create_mock_model):
        """测试删除模型"""
        create_mock_model("model_to_delete")

        success = manager.delete_model("model_to_delete")
        assert success

        models = manager.list_models()
        names = [m["name"] for m in models]
        assert "model_to_delete" not in names


# ==================== 策略测试 ====================


class TestQlibMLStrategy:
    """QlibMLStrategy 测试"""

    def test_strategy_imports(self):
        """测试策略可以正确导入"""
        from autotrade.execution.strategies import QlibMLStrategy

        assert QlibMLStrategy is not None

    def test_default_parameters(self):
        """测试默认参数"""
        from autotrade.execution.strategies import QlibMLStrategy

        params = QlibMLStrategy.parameters

        assert "symbols" in params
        assert "top_k" in params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
