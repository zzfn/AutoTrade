from pathlib import Path
from typing import Any

import yaml


class ConfigLoader:
    def __init__(self, config_path: str = "configs/universe.yaml"):
        # Resolve absolute path relative to project root if possible, or use as is
        # Assuming run from project root
        self.config_path = Path(config_path)
        self._config = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at {self.config_path}")

        with open(self.config_path) as f:
            return yaml.safe_load(f)

    @property
    def symbols(self) -> list[str]:
        return self._config.get("symbols", [])

    @property
    def timeframe(self) -> str:
        return self._config.get("timeframe", "1d")

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)


# Singleton instance for easy access
# Users can also instantiate their own if they need a different path
# but defaults are good for standard structure
try:
    default_config = ConfigLoader()
except Exception:
    # Allow import even if config is missing (e.g. during CI/tests setup)
    default_config = None
