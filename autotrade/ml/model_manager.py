"""
模型管理器 - 管理模型版本和元数据

任务 3.4: 实现模型保存/加载功能（含元数据）
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger


class ModelManager:
    """
    模型管理器

    负责：
    1. 列出所有可用模型
    2. 获取模型元数据
    3. 选择当前使用的模型
    4. 删除旧模型
    """

    def __init__(self, models_dir: str | Path = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # 当前选择的模型
        self._current_model: Optional[str] = None
        self._config_file = self.models_dir / ".current_model"

        # 加载配置
        self._load_config()

    def _load_config(self):
        """加载配置"""
        if self._config_file.exists():
            with open(self._config_file, "r") as f:
                self._current_model = f.read().strip()

    def _save_config(self):
        """保存配置"""
        with open(self._config_file, "w") as f:
            f.write(self._current_model or "")

    def list_models(self) -> list[dict]:
        """
        列出所有可用模型

        Returns:
            模型信息列表，每个元素包含 name, version, metadata 等
        """
        models = []

        for model_dir in self.models_dir.iterdir():
            if not model_dir.is_dir():
                continue
            if model_dir.name.startswith("."):
                continue

            metadata_file = model_dir / "metadata.json"
            model_file = model_dir / "model.pkl"

            if not model_file.exists():
                continue

            # 读取元数据
            metadata = {}
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"读取 {model_dir.name} 元数据失败: {e}")

            # 获取文件信息
            stat = model_file.stat()

            model_info = {
                "name": model_dir.name,
                "path": str(model_dir),
                "model_type": metadata.get("model_type", "unknown"),
                "version": metadata.get("version", ""),
                "saved_at": metadata.get("saved_at", ""),
                "train_samples": metadata.get("train_samples", 0),
                "num_features": metadata.get("num_features", 0),
                "metrics": {
                    "ic": metadata.get("ic", 0),
                    "icir": metadata.get("icir", 0),
                    "mse": metadata.get("mse", 0),
                },
                "file_size_mb": stat.st_size / (1024 * 1024),
                "is_current": model_dir.name == self._current_model,
            }

            models.append(model_info)

        # 按保存时间排序（最新的在前）
        models.sort(key=lambda x: x.get("saved_at", ""), reverse=True)

        return models

    def get_model_info(self, model_name: str) -> Optional[dict]:
        """获取单个模型的详细信息"""
        model_dir = self.models_dir / model_name

        if not model_dir.exists():
            return None

        metadata_file = model_dir / "metadata.json"

        if not metadata_file.exists():
            return {"name": model_name, "path": str(model_dir)}

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        return {
            "name": model_name,
            "path": str(model_dir),
            **metadata,
            "is_current": model_name == self._current_model,
        }

    def set_current_model(self, model_name: str) -> bool:
        """
        设置当前使用的模型

        Args:
            model_name: 模型名称

        Returns:
            是否设置成功
        """
        model_dir = self.models_dir / model_name

        if not model_dir.exists():
            logger.error(f"模型不存在: {model_name}")
            return False

        self._current_model = model_name
        self._save_config()
        logger.info(f"当前模型设置为: {model_name}")
        return True

    def get_current_model(self) -> Optional[str]:
        """获取当前使用的模型名称"""
        return self._current_model

    def get_current_model_path(self) -> Optional[Path]:
        """获取当前使用的模型路径"""
        if not self._current_model:
            return None
        return self.models_dir / self._current_model

    def delete_model(self, model_name: str) -> bool:
        """
        删除模型

        Args:
            model_name: 模型名称

        Returns:
            是否删除成功
        """
        import shutil

        model_dir = self.models_dir / model_name

        if not model_dir.exists():
            logger.error(f"模型不存在: {model_name}")
            return False

        try:
            shutil.rmtree(model_dir)
            logger.info(f"模型已删除: {model_name}")

            # 如果删除的是当前模型，清除选择
            if self._current_model == model_name:
                self._current_model = None
                self._save_config()

            return True
        except Exception as e:
            logger.error(f"删除模型失败: {e}")
            return False

    def cleanup_old_models(self, keep_count: int = 5) -> list[str]:
        """
        清理旧模型，只保留最新的 N 个

        Args:
            keep_count: 保留的模型数量

        Returns:
            被删除的模型名称列表
        """
        models = self.list_models()

        if len(models) <= keep_count:
            return []

        # 跳过当前使用的模型
        to_delete = []
        kept = 0

        for model in models:
            if model["is_current"]:
                continue

            if kept < keep_count - (1 if self._current_model else 0):
                kept += 1
            else:
                to_delete.append(model["name"])

        # 执行删除
        deleted = []
        for name in to_delete:
            if self.delete_model(name):
                deleted.append(name)

        return deleted


def get_model_manager() -> ModelManager:
    """获取全局模型管理器实例"""
    # 使用项目根目录下的 models 目录
    base_dir = Path(__file__).parent.parent.parent.parent
    models_dir = base_dir / "models"
    return ModelManager(models_dir)
