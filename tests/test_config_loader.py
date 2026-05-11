"""
Unit tests for configuration loader
"""

import pytest
import os
import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


class TestConfigLoader:
    """Test configuration loading and merging"""

    def test_config_loads_default(self):
        """Test that default config loads"""
        from config_loader import Config

        config = Config()

        assert config._config is not None
        assert len(config._config) > 0, "Config should have content"

    def test_get_scanner_workers(self):
        """Test getting scanner.workers config"""
        from config_loader import config

        workers = config.get('scanner.workers')

        assert workers is not None, "Scanner workers should be configured"
        assert isinstance(workers, int), "Workers should be an integer"
        assert workers > 0, "Workers should be positive"

    def test_get_with_default(self):
        """Test get() with default value"""
        from config_loader import config

        value = config.get('nonexistent.key', 'default_value')

        assert value == 'default_value', "Should return default for missing key"

    def test_get_section(self):
        """Test getting entire config section"""
        from config_loader import config

        scanner_config = config.get_section('scanner')

        assert isinstance(scanner_config, dict), "Section should be a dict"
        assert 'workers' in scanner_config, "Scanner section should have workers"
        assert 'delay_min' in scanner_config, "Scanner section should have delay_min"

    def test_cache_freshness_intervals(self):
        """Test cache freshness interval configuration"""
        from config_loader import config

        intervals = config.get('cache.freshness_intervals')

        assert isinstance(intervals, dict), "Freshness intervals should be dict"
        assert 'price' in intervals
        assert 'earnings' in intervals
        assert intervals['price'] == 0, "Price should always be fresh (0)"
        assert intervals['earnings'] > 0, "Earnings should have TTL"

    def test_canslim_max_scores(self):
        """Test CANSLIM max scores configuration"""
        from config_loader import config

        max_scores = config.get('scoring.canslim.max_scores')

        assert max_scores['C'] == 15
        assert max_scores['A'] == 15
        assert max_scores['N'] == 15
        assert max_scores['S'] == 15
        assert max_scores['L'] == 15
        assert max_scores['I'] == 10
        assert max_scores['M'] == 15

        total = sum(max_scores.values())
        assert total == 100, "CANSLIM scores should sum to 100"

    def test_growth_mode_max_scores(self):
        """Test Growth Mode max scores configuration"""
        from config_loader import config

        max_scores = config.get('scoring.growth_mode.max_scores')

        assert max_scores['R'] == 20
        assert max_scores['F'] == 15
        assert max_scores['N'] == 15
        assert max_scores['S'] == 15
        assert max_scores['L'] == 15
        assert max_scores['I'] == 10
        assert max_scores['M'] == 10

        total = sum(max_scores.values())
        assert total == 100, "Growth Mode scores should sum to 100"

    def test_market_index_weights(self):
        """Test market index weights sum to 1.0"""
        from config_loader import config

        indexes = config.get('market.indexes')

        weights = [idx['weight'] for idx in indexes.values()]
        total_weight = sum(weights)

        assert abs(total_weight - 1.0) < 0.01, "Market weights should sum to 1.0"

    def test_ai_trader_trailing_stops(self):
        """Test AI trader trailing stop configuration"""
        from config_loader import config

        stops = config.get('ai_trader.trailing_stops')

        assert 'gain_50_plus' in stops
        assert stops['gain_50_plus']['threshold'] == 0.50
        assert stops['gain_50_plus']['stop_pct'] == 0.15

    def test_config_properties(self):
        """Test convenience properties"""
        from config_loader import config

        assert isinstance(config.scanner, dict)
        assert isinstance(config.cache, dict)
        assert isinstance(config.scoring, dict)
        assert isinstance(config.market, dict)
        assert isinstance(config.ai_trader, dict)
        assert isinstance(config.technical, dict)
        assert isinstance(config.api, dict)

    def test_database_property_returns_dict(self):
        """database property mirrors the other section accessors — it must
        return a dict regardless of whether the database section exists
        in YAML (covers config_loader.py line 113)."""
        from config_loader import config

        result = config.database
        assert isinstance(result, dict), "database property must return a dict"

    def test_reload_happy_path_reloads_config(self):
        """reload() should re-read YAML from disk and restore the singleton's
        _config to a fully-populated dict (covers config_loader.py lines 74-77)."""
        from config_loader import config

        # Take a snapshot of a known-good value before reload.
        scanner_workers_before = config.get('scanner.workers')
        assert scanner_workers_before is not None

        # Mutate the in-memory config to prove reload() actually re-reads.
        config._config['scanner']['workers'] = 99999
        assert config.get('scanner.workers') == 99999

        config.reload()

        # After reload, the in-memory mutation should be gone.
        assert config.get('scanner.workers') == scanner_workers_before

    def test_reload_failure_rolls_back_to_old_config(self):
        """If _load_config raises mid-reload, the old config must be restored
        verbatim — never leave the singleton with a half-loaded or empty
        config (covers config_loader.py lines 78-80, the atomic-swap path)."""
        from config_loader import config
        from unittest.mock import patch

        # Snapshot the live config so we can verify byte-identical rollback.
        before = config._config
        assert before, "Config must be populated before this test runs"
        sentinel_workers = before.get('scanner', {}).get('workers')

        # Force the second _load_config call (the one inside reload) to raise.
        with patch.object(config, '_load_config', side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                config.reload()

        # The singleton's _config must be the EXACT same object as before —
        # the except block does `self._config = old_config` (not a copy),
        # so identity holds.
        assert config._config is before
        assert config.get('scanner.workers') == sentinel_workers
