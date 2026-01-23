"""
模型训练器 - 统一训练接口

任务 3.2 - 3.4: 实现 ModelTrainer 基础类和 LightGBMTrainer
"""

import json
import os
import pickle
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM 未安装，LightGBMTrainer 不可用")


class ModelTrainer(ABC):
    """
    模型训练器基类

    定义统一的训练接口：
    - train(): 训练模型
    - predict(): 模型预测
    - save(): 保存模型
    - load(): 加载模型
    - evaluate(): 评估模型
    """

    def __init__(
        self,
        model_dir: str | Path = "models",
        model_name: str = "model",
    ):
        self.model_dir = Path(model_dir)
        self.model_name = model_name
        self.model: Any = None
        self.metadata: dict = {}

    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        **kwargs,
    ) -> "ModelTrainer":
        """训练模型"""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """模型预测"""
        pass

    def evaluate(
        self, X: pd.DataFrame, y: pd.Series, metrics: list[str] | None = None
    ) -> dict:
        """
        评估模型

        Args:
            X: 特征数据
            y: 目标变量
            metrics: 评估指标列表

        Returns:
            包含各指标值的字典
        """
        if metrics is None:
            metrics = ["ic", "icir", "mse", "mae"]

        predictions = self.predict(X)
        results = {}

        if "ic" in metrics:
            # IC (Information Coefficient) - 预测值与真实值的相关系数
            ic = np.corrcoef(predictions, y)[0, 1]
            results["ic"] = ic if not np.isnan(ic) else 0.0

        if "icir" in metrics:
            # ICIR 需要多期 IC，这里简化处理
            results["icir"] = results.get("ic", 0.0) * np.sqrt(len(y) / 20)

        if "mse" in metrics:
            results["mse"] = float(np.mean((predictions - y) ** 2))

        if "mae" in metrics:
            results["mae"] = float(np.mean(np.abs(predictions - y)))

        return results

    def save(self, version: str | None = None) -> Path:
        """
        保存模型

        Args:
            version: 版本号，默认使用时间戳

        Returns:
            模型保存目录
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_path = self.model_dir / f"{self.model_name}_{version}"
        model_path.mkdir(parents=True, exist_ok=True)

        # 保存模型
        model_file = model_path / "model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(self.model, f)

        # 更新元数据
        self.metadata.update(
            {
                "model_name": self.model_name,
                "version": version,
                "saved_at": datetime.now().isoformat(),
                "model_type": self.__class__.__name__,
            }
        )

        # 保存元数据
        metadata_file = model_path / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

        logger.info(f"模型已保存到: {model_path}")
        return model_path

    def load(self, model_path: str | Path) -> "ModelTrainer":
        """
        加载模型

        Args:
            model_path: 模型目录路径

        Returns:
            self
        """
        model_path = Path(model_path)

        # 加载模型
        model_file = model_path / "model.pkl"
        with open(model_file, "rb") as f:
            self.model = pickle.load(f)

        # 加载元数据
        metadata_file = model_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                self.metadata = json.load(f)

        logger.info(f"模型已从 {model_path} 加载")
        return self


class LightGBMTrainer(ModelTrainer):
    """
    LightGBM 模型训练器

    使用 LightGBM 进行回归预测（预测收益率）
    """

    DEFAULT_PARAMS = {
        "objective": "regression",
        "metric": "mse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    def __init__(
        self,
        model_dir: str | Path = "models",
        model_name: str = "lightgbm",
        params: dict | None = None,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50,
    ):
        super().__init__(model_dir, model_name)

        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM 未安装，请运行: pip install lightgbm")

        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.feature_importance_: Optional[pd.DataFrame] = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        **kwargs,
    ) -> "LightGBMTrainer":
        """
        训练 LightGBM 模型

        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_valid: 验证特征（可选）
            y_valid: 验证目标（可选）

        Returns:
            self
        """
        logger.info(
            f"开始训练 LightGBM: {len(X_train)} 样本, {len(X_train.columns)} 特征"
        )

        # 创建数据集
        train_data = lgb.Dataset(X_train, label=y_train)

        valid_sets = [train_data]
        valid_names = ["train"]

        if X_valid is not None and y_valid is not None:
            valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
            valid_sets.append(valid_data)
            valid_names.append("valid")

        # 训练回调
        callbacks = [
            lgb.log_evaluation(period=100),
        ]

        if X_valid is not None:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds))

        # 训练模型
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        # 记录特征重要性
        self.feature_importance_ = pd.DataFrame(
            {
                "feature": X_train.columns,
                "importance": self.model.feature_importance(importance_type="gain"),
            }
        ).sort_values("importance", ascending=False)

        # 更新元数据
        best_iteration = (
            self.model.best_iteration
            if self.model.best_iteration and self.model.best_iteration > 0
            else self.model.current_iteration()
        )
        self.metadata.update(
            {
                "train_samples": len(X_train),
                "num_features": len(X_train.columns),
                "best_iteration": best_iteration,
                "params": self.params,
                "feature_names": list(X_train.columns),
            }
        )

        logger.info(f"训练完成，最佳迭代: {best_iteration}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        模型预测

        Args:
            X: 特征数据

        Returns:
            预测值数组
        """
        if self.model is None:
            raise ValueError("模型未训练或未加载")

        best_iteration = (
            self.model.best_iteration
            if self.model.best_iteration and self.model.best_iteration > 0
            else self.model.current_iteration()
        )
        return self.model.predict(X, num_iteration=best_iteration)

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """获取 Top-N 特征重要性"""
        if self.feature_importance_ is None:
            return pd.DataFrame()
        return self.feature_importance_.head(top_n)


class WalkForwardValidator:
    """
    Walk-Forward 验证器

    任务 3.6: 实现滚动窗口训练和验证
    """

    def __init__(
        self,
        trainer_class: type = LightGBMTrainer,
        train_window: int = 252,  # 约 1 年交易日
        test_window: int = 21,  # 约 1 个月
        step_size: int = 21,  # 每次滚动 1 个月
        **trainer_kwargs,
    ):
        """
        初始化验证器

        Args:
            trainer_class: 训练器类
            train_window: 训练窗口大小（天数）
            test_window: 测试窗口大小（天数）
            step_size: 滚动步长（天数）
            trainer_kwargs: 传递给训练器的参数
        """
        self.trainer_class = trainer_class
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.trainer_kwargs = trainer_kwargs

    def validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_column: str = "target",
    ) -> dict:
        """
        执行 Walk-Forward 验证

        Args:
            X: 特征数据
            y: 目标变量
            target_column: 目标列名

        Returns:
            验证结果，包含每个窗口的指标
        """
        logger.info("开始 Walk-Forward 验证")

        results = []
        n_samples = len(X)

        start_idx = 0
        fold = 0

        while start_idx + self.train_window + self.test_window <= n_samples:
            train_end = start_idx + self.train_window
            test_end = train_end + self.test_window

            # 分割数据
            X_train = X.iloc[start_idx:train_end]
            y_train = y.iloc[start_idx:train_end]
            X_test = X.iloc[train_end:test_end]
            y_test = y.iloc[train_end:test_end]

            # 训练和评估
            trainer = self.trainer_class(**self.trainer_kwargs)
            trainer.train(X_train, y_train)
            metrics = trainer.evaluate(X_test, y_test)

            fold_result = {
                "fold": fold,
                "train_start": start_idx,
                "train_end": train_end,
                "test_start": train_end,
                "test_end": test_end,
                **metrics,
            }
            results.append(fold_result)

            logger.info(f"Fold {fold}: IC={metrics.get('ic', 0):.4f}")

            start_idx += self.step_size
            fold += 1

        # 汇总结果
        results_df = pd.DataFrame(results)

        summary = {
            "num_folds": len(results),
            "mean_ic": float(results_df["ic"].mean()),
            "std_ic": float(results_df["ic"].std()),
            "icir": float(
                results_df["ic"].mean()
                / results_df["ic"].std()
                * np.sqrt(len(results))
            )
            if len(results) > 1
            else 0.0,
            "mean_mse": float(results_df["mse"].mean()),
            "fold_results": results,
        }

        logger.info(
            f"验证完成: {summary['num_folds']} 折, "
            f"Mean IC={summary['mean_ic']:.4f}, ICIR={summary['icir']:.4f}"
        )

        return summary
