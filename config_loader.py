"""
Configuration Loader for CANSLIM Analyzer
Loads YAML configuration files based on environment
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """Configuration singleton that loads and merges YAML configs"""

    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Load configuration from YAML files"""
        config_dir = Path(__file__).parent / "config"
        env = os.getenv("CANSLIM_ENV", "development")

        # Load default config
        default_path = config_dir / "default.yaml"
        if default_path.exists():
            with open(default_path, 'r') as f:
                self._config = yaml.safe_load(f) or {}

        # Load environment-specific config and merge
        env_path = config_dir / f"{env}.yaml"
        if env_path.exists():
            with open(env_path, 'r') as f:
                env_config = yaml.safe_load(f) or {}
                self._deep_merge(self._config, env_config)

        print(f"✓ Loaded configuration for environment: {env}")

    def _deep_merge(self, base: dict, override: dict):
        """Recursively merge override into base"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def get(self, path: str, default=None) -> Any:
        """
        Get config value using dot notation.
        Example: config.get('scanner.workers', 4)
        """
        keys = path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        return self.get(section, {})

    def reload(self):
        """Reload configuration from files"""
        self._config = {}
        self._load_config()

    # Convenience properties for common config sections
    @property
    def scanner(self) -> Dict[str, Any]:
        return self.get_section('scanner')

    @property
    def cache(self) -> Dict[str, Any]:
        return self.get_section('cache')

    @property
    def scoring(self) -> Dict[str, Any]:
        return self.get_section('scoring')

    @property
    def market(self) -> Dict[str, Any]:
        return self.get_section('market')

    @property
    def ai_trader(self) -> Dict[str, Any]:
        return self.get_section('ai_trader')

    @property
    def technical(self) -> Dict[str, Any]:
        return self.get_section('technical')

    @property
    def api(self) -> Dict[str, Any]:
        return self.get_section('api')

    @property
    def database(self) -> Dict[str, Any]:
        return self.get_section('database')


# Singleton instance
config = Config()


if __name__ == "__main__":
    # Test the config loader
    print("\n=== Configuration Test ===\n")

    print(f"Scanner workers: {config.get('scanner.workers')}")
    print(f"Cache enabled: {config.get('cache.redis.enabled')}")
    print(f"CANSLIM C max score: {config.get('scoring.canslim.max_scores.C')}")
    print(f"FMP rate limit: {config.get('api.fmp.rate_limit')}")

    print("\n=== Full Scanner Config ===")
    import json
    print(json.dumps(config.scanner, indent=2))

    print("\n=== Market Indexes ===")
    print(json.dumps(config.market.get('indexes'), indent=2))
