"""
Unit tests for Redis cache layer
"""

import pytest
import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


class TestRedisCache:
    """Test Redis cache functionality"""

    def test_redis_cache_initialization(self):
        """Test that RedisCache initializes"""
        from redis_cache import RedisCache

        cache = RedisCache()

        assert cache is not None
        # Cache may or may not be enabled depending on Redis availability
        assert isinstance(cache.enabled, bool)

    def test_redis_cache_set_get(self):
        """Test basic set/get operations"""
        from redis_cache import redis_cache

        if not redis_cache.enabled:
            pytest.skip("Redis not available")

        test_data = {
            "quarterly_earnings": [1.0, 1.1, 1.2],
            "annual_earnings": [4.0, 3.5, 3.0]
        }

        # Set data
        success = redis_cache.set("TEST", "earnings", test_data, ttl=60)
        assert success, "Set should succeed"

        # Get data
        retrieved = redis_cache.get("TEST", "earnings")
        assert retrieved is not None, "Should retrieve cached data"
        assert retrieved == test_data, "Retrieved data should match"

        # Cleanup
        redis_cache.delete("TEST", "earnings")

    def test_redis_cache_ttl(self):
        """Test TTL (time-to-live) functionality"""
        from redis_cache import redis_cache

        if not redis_cache.enabled:
            pytest.skip("Redis not available")

        test_data = {"value": 123}

        # Set with 60 second TTL
        redis_cache.set("TEST", "temp_data", test_data, ttl=60)

        # Check TTL
        ttl = redis_cache.get_ttl("TEST", "temp_data")
        assert ttl > 0, "TTL should be positive"
        assert ttl <= 60, "TTL should be <= 60 seconds"

        # Cleanup
        redis_cache.delete("TEST", "temp_data")

    def test_redis_cache_exists(self):
        """Test exists() method"""
        from redis_cache import redis_cache

        if not redis_cache.enabled:
            pytest.skip("Redis not available")

        test_data = {"test": True}

        # Should not exist initially
        assert not redis_cache.exists("TEST", "exists_test")

        # Set data
        redis_cache.set("TEST", "exists_test", test_data, ttl=60)

        # Should exist now
        assert redis_cache.exists("TEST", "exists_test")

        # Cleanup
        redis_cache.delete("TEST", "exists_test")

    def test_redis_cache_delete(self):
        """Test delete functionality"""
        from redis_cache import redis_cache

        if not redis_cache.enabled:
            pytest.skip("Redis not available")

        test_data = {"value": "to_be_deleted"}

        # Set data
        redis_cache.set("TEST", "delete_test", test_data, ttl=60)
        assert redis_cache.exists("TEST", "delete_test")

        # Delete
        success = redis_cache.delete("TEST", "delete_test")
        assert success, "Delete should succeed"

        # Should not exist anymore
        assert not redis_cache.exists("TEST", "delete_test")

    def test_redis_cache_flush_ticker(self):
        """Test flushing all data for a ticker"""
        from redis_cache import redis_cache

        if not redis_cache.enabled:
            pytest.skip("Redis not available")

        # Set multiple data types for same ticker
        redis_cache.set("FLUSH_TEST", "earnings", {"q": [1, 2, 3]}, ttl=60)
        redis_cache.set("FLUSH_TEST", "revenue", {"q": [100, 200]}, ttl=60)
        redis_cache.set("FLUSH_TEST", "analyst", {"target": 150}, ttl=60)

        # Verify they exist
        assert redis_cache.exists("FLUSH_TEST", "earnings")
        assert redis_cache.exists("FLUSH_TEST", "revenue")
        assert redis_cache.exists("FLUSH_TEST", "analyst")

        # Flush ticker
        count = redis_cache.flush_ticker("FLUSH_TEST")
        assert count >= 3, f"Should delete at least 3 keys, deleted {count}"

        # Verify they're gone
        assert not redis_cache.exists("FLUSH_TEST", "earnings")
        assert not redis_cache.exists("FLUSH_TEST", "revenue")
        assert not redis_cache.exists("FLUSH_TEST", "analyst")

    def test_redis_cache_make_key(self):
        """Test key naming convention"""
        from redis_cache import redis_cache

        key = redis_cache._make_key("AAPL", "earnings")

        assert "canslim" in key, "Key should have namespace"
        assert "AAPL" in key, "Key should contain ticker"
        assert "earnings" in key, "Key should contain data type"

    def test_redis_fallback_when_disabled(self):
        """Test that cache operations work (return False/None) when Redis disabled"""
        from redis_cache import RedisCache

        # Create cache instance that will fail to connect
        import os
        os.environ['CANSLIM_ENV'] = 'test'

        cache = RedisCache()
        # Even if disabled, operations shouldn't crash
        result = cache.get("TEST", "data")
        assert result is None, "Get on disabled cache should return None"

        success = cache.set("TEST", "data", {"value": 1})
        # Success will be False if disabled
        assert isinstance(success, bool)
