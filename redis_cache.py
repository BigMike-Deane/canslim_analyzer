"""
Redis Cache Layer for CANSLIM Analyzer
Provides fast, persistent caching with automatic TTL management
"""

import json
import logging
from datetime import datetime
from typing import Any, Optional
from config_loader import config

logger = logging.getLogger(__name__)

# Redis client (lazy loaded)
_redis_client = None
_redis_available = False


def get_redis_client():
    """Get or create Redis client singleton"""
    global _redis_client, _redis_available

    if _redis_client is not None:
        return _redis_client

    # Check if Redis is enabled
    if not config.get('cache.redis.enabled', False):
        logger.info("Redis cache is disabled in configuration")
        _redis_available = False
        return None

    try:
        import redis

        redis_config = config.get_section('cache').get('redis', {})
        _redis_client = redis.Redis(
            host=redis_config.get('host', 'localhost'),
            port=redis_config.get('port', 6379),
            db=redis_config.get('db', 0),
            decode_responses=redis_config.get('decode_responses', True),
            socket_connect_timeout=2,
            socket_timeout=2,
        )

        # Test connection
        _redis_client.ping()
        _redis_available = True
        logger.info(f"✓ Redis cache connected: {redis_config.get('host')}:{redis_config.get('port')}")

    except ImportError:
        logger.warning("Redis library not installed. Install with: pip install redis")
        _redis_available = False

    except Exception as e:
        logger.warning(f"Redis connection failed: {e}. Falling back to in-memory cache.")
        _redis_client = None
        _redis_available = False

    return _redis_client


def is_redis_available() -> bool:
    """Check if Redis is available and connected"""
    global _redis_available
    if _redis_client is None:
        get_redis_client()
    return _redis_available


class RedisCache:
    """Redis cache manager with automatic serialization and TTL"""

    def __init__(self):
        self.client = get_redis_client()
        self.enabled = is_redis_available()
        self.freshness_intervals = config.get('cache.freshness_intervals', {})

    def get(self, ticker: str, data_type: str) -> Optional[Any]:
        """Get cached data from Redis"""
        if not self.enabled:
            return None

        try:
            key = self._make_key(ticker, data_type)
            cached = self.client.get(key)

            if cached:
                logger.debug(f"Redis HIT: {key}")
                return json.loads(cached)

            logger.debug(f"Redis MISS: {key}")
            return None

        except Exception as e:
            logger.debug(f"Redis get error for {ticker}:{data_type}: {e}")
            return None

    def set(self, ticker: str, data_type: str, data: Any, ttl: Optional[int] = None) -> bool:
        """
        Set cached data in Redis with automatic TTL.

        Args:
            ticker: Stock ticker
            data_type: Type of data (earnings, revenue, etc.)
            data: Data to cache (will be JSON serialized)
            ttl: Time-to-live in seconds (uses config if not specified)

        Returns:
            True if successfully cached
        """
        if not self.enabled:
            return False

        try:
            key = self._make_key(ticker, data_type)

            # Get TTL from config if not specified
            if ttl is None:
                ttl = self.freshness_intervals.get(data_type, 86400)  # Default 24h

            # Serialize and store
            serialized = json.dumps(data, default=str)

            if ttl > 0:
                self.client.setex(key, ttl, serialized)
            else:
                # TTL of 0 means always fresh (price data)
                self.client.set(key, serialized)

            logger.debug(f"Redis SET: {key} (TTL: {ttl}s)")
            return True

        except Exception as e:
            logger.debug(f"Redis set error for {ticker}:{data_type}: {e}")
            return False

    def delete(self, ticker: str, data_type: str) -> bool:
        """Delete cached data"""
        if not self.enabled:
            return False

        try:
            key = self._make_key(ticker, data_type)
            self.client.delete(key)
            logger.debug(f"Redis DELETE: {key}")
            return True

        except Exception as e:
            logger.debug(f"Redis delete error for {ticker}:{data_type}: {e}")
            return False

    def flush_ticker(self, ticker: str) -> int:
        """Delete all cached data for a ticker"""
        if not self.enabled:
            return 0

        try:
            pattern = f"canslim:{ticker}:*"
            keys = self.client.keys(pattern)
            if keys:
                count = self.client.delete(*keys)
                logger.info(f"Redis FLUSH: {ticker} ({count} keys)")
                return count
            return 0

        except Exception as e:
            logger.debug(f"Redis flush error for {ticker}: {e}")
            return 0

    def flush_all(self) -> bool:
        """Flush entire CANSLIM cache (keeps other Redis data)"""
        if not self.enabled:
            return False

        try:
            pattern = "canslim:*"
            keys = self.client.keys(pattern)
            if keys:
                count = self.client.delete(*keys)
                logger.info(f"Redis FLUSH ALL: Deleted {count} keys")
            return True

        except Exception as e:
            logger.warning(f"Redis flush all error: {e}")
            return False

    def get_ttl(self, ticker: str, data_type: str) -> int:
        """Get remaining TTL in seconds (-1 if no TTL, -2 if not exists)"""
        if not self.enabled:
            return -2

        try:
            key = self._make_key(ticker, data_type)
            return self.client.ttl(key)

        except Exception as e:
            logger.debug(f"Redis TTL error for {ticker}:{data_type}: {e}")
            return -2

    def exists(self, ticker: str, data_type: str) -> bool:
        """Check if key exists in cache"""
        if not self.enabled:
            return False

        try:
            key = self._make_key(ticker, data_type)
            return bool(self.client.exists(key))

        except Exception as e:
            logger.debug(f"Redis exists error for {ticker}:{data_type}: {e}")
            return False

    def get_stats(self) -> dict:
        """Get Redis cache statistics"""
        if not self.enabled:
            return {"enabled": False}

        try:
            info = self.client.info('stats')
            keyspace = self.client.info('keyspace')

            # Count CANSLIM-specific keys
            canslim_keys = len(self.client.keys("canslim:*"))

            return {
                "enabled": True,
                "connected": True,
                "total_keys": info.get('db0', {}).get('keys', 0),
                "canslim_keys": canslim_keys,
                "total_commands": info.get('total_commands_processed', 0),
                "keyspace_hits": info.get('keyspace_hits', 0),
                "keyspace_misses": info.get('keyspace_misses', 0),
                "hit_rate": self._calculate_hit_rate(info),
            }

        except Exception as e:
            logger.warning(f"Redis stats error: {e}")
            return {"enabled": True, "connected": False, "error": str(e)}

    def _make_key(self, ticker: str, data_type: str) -> str:
        """Create Redis key with namespace"""
        return f"canslim:{ticker}:{data_type}"

    def _calculate_hit_rate(self, info: dict) -> float:
        """Calculate cache hit rate percentage"""
        hits = info.get('keyspace_hits', 0)
        misses = info.get('keyspace_misses', 0)
        total = hits + misses

        if total == 0:
            return 0.0

        return round((hits / total) * 100, 2)


# Singleton instance
redis_cache = RedisCache()


if __name__ == "__main__":
    # Test the Redis cache
    print("\n=== Redis Cache Test ===\n")

    print(f"Redis enabled: {redis_cache.enabled}")

    if redis_cache.enabled:
        # Test set/get
        test_data = {
            "quarterly_earnings": [1.25, 1.15, 1.05, 0.95],
            "annual_earnings": [4.50, 4.00, 3.50]
        }

        print("\nTesting set/get:")
        redis_cache.set("AAPL", "earnings", test_data, ttl=60)
        retrieved = redis_cache.get("AAPL", "earnings")
        print(f"Stored: {test_data}")
        print(f"Retrieved: {retrieved}")
        print(f"Match: {test_data == retrieved}")

        # Test TTL
        ttl = redis_cache.get_ttl("AAPL", "earnings")
        print(f"\nTTL for AAPL earnings: {ttl} seconds")

        # Test stats
        stats = redis_cache.get_stats()
        print(f"\nRedis Stats:")
        import json
        print(json.dumps(stats, indent=2))

        # Cleanup
        redis_cache.delete("AAPL", "earnings")
        print("\n✓ Test completed and cleaned up")

    else:
        print("Redis not available. Skipping tests.")
