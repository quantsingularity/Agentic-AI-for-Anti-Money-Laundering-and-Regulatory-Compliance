"""
Redis Cache Manager for High-Performance AML Processing
Provides caching layer for sanctions lists, entity profiles, and ML model outputs
"""

import redis
import json
from typing import Any, Optional, Dict, List
from loguru import logger
import hashlib


class RedisCache:
    """
    Redis-based caching layer for AML system.
    Handles entity profiles, sanctions lists, and ML predictions.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        decode_responses: bool = False,
    ):
        """
        Initialize Redis connection.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Optional authentication password
            decode_responses: Whether to decode responses to strings
        """
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=decode_responses,
            socket_connect_timeout=5,
            socket_keepalive=True,
            health_check_interval=30,
        )

        # Test connection
        try:
            self.client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def set(
        self, key: str, value: Any, ttl: Optional[int] = None, namespace: str = "aml"
    ) -> bool:
        """
        Set a cache entry.

        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl: Time-to-live in seconds (None = no expiry, 0 = expire immediately)
            namespace: Key namespace prefix

        Returns:
            Success boolean
        """
        try:
            full_key = f"{namespace}:{key}"
            serialized = json.dumps(value)

            if ttl is not None:
                return self.client.setex(full_key, ttl, serialized)
            else:
                return self.client.set(full_key, serialized)

        except Exception as e:
            logger.error(f"Cache SET error for key '{key}': {e}")
            return False

    def get(self, key: str, namespace: str = "aml") -> Optional[Any]:
        """
        Get a cache entry.

        Args:
            key: Cache key
            namespace: Key namespace prefix

        Returns:
            Cached value or None if not found
        """
        try:
            full_key = f"{namespace}:{key}"
            value = self.client.get(full_key)

            if value:
                return json.loads(value)
            return None

        except Exception as e:
            logger.error(f"Cache GET error for key '{key}': {e}")
            return None

    def delete(self, key: str, namespace: str = "aml") -> bool:
        """Delete a cache entry."""
        try:
            full_key = f"{namespace}:{key}"
            return bool(self.client.delete(full_key))
        except Exception as e:
            logger.error(f"Cache DELETE error for key '{key}': {e}")
            return False

    def exists(self, key: str, namespace: str = "aml") -> bool:
        """Check if key exists."""
        full_key = f"{namespace}:{key}"
        return bool(self.client.exists(full_key))

    # Specialized AML caching methods

    def cache_entity_profile(self, entity_id: str, profile: Dict, ttl: int = 3600):
        """
        Cache entity profile with 1-hour default TTL.

        Args:
            entity_id: Entity identifier
            profile: Entity profile dict
            ttl: Cache duration in seconds
        """
        return self.set(f"entity:{entity_id}", profile, ttl, namespace="aml:entities")

    def get_entity_profile(self, entity_id: str) -> Optional[Dict]:
        """Retrieve cached entity profile."""
        return self.get(f"entity:{entity_id}", namespace="aml:entities")

    def cache_sanctions_check(
        self, entity_name: str, sanctions_data: Dict, ttl: int = 86400  # 24 hours
    ):
        """
        Cache sanctions list check result.

        Args:
            entity_name: Entity name
            sanctions_data: Sanctions check result
            ttl: Cache duration (default 24h)
        """
        # Create hash of name for consistent lookup
        name_hash = hashlib.sha256(entity_name.lower().encode()).hexdigest()
        return self.set(
            f"sanctions:{name_hash}", sanctions_data, ttl, namespace="aml:sanctions"
        )

    def get_sanctions_check(self, entity_name: str) -> Optional[Dict]:
        """Retrieve cached sanctions check."""
        name_hash = hashlib.sha256(entity_name.lower().encode()).hexdigest()
        return self.get(f"sanctions:{name_hash}", namespace="aml:sanctions")

    def cache_risk_score(
        self,
        transaction_id: str,
        risk_score: float,
        features: Dict,
        ttl: int = 7200,  # 2 hours
    ):
        """
        Cache ML model risk score.

        Args:
            transaction_id: Transaction identifier
            risk_score: Computed risk score
            features: Feature vector used
            ttl: Cache duration
        """
        cache_data = {
            "risk_score": risk_score,
            "features": features,
            "timestamp": self._get_timestamp(),
        }
        return self.set(
            f"risk:{transaction_id}", cache_data, ttl, namespace="aml:scores"
        )

    def get_risk_score(self, transaction_id: str) -> Optional[Dict]:
        """Retrieve cached risk score."""
        return self.get(f"risk:{transaction_id}", namespace="aml:scores")

    def increment_counter(
        self, counter_name: str, namespace: str = "aml:counters"
    ) -> int:
        """
        Increment a counter (e.g., for rate limiting).

        Returns:
            New counter value
        """
        full_key = f"{namespace}:{counter_name}"
        return self.client.incr(full_key)

    def get_counter(self, counter_name: str, namespace: str = "aml:counters") -> int:
        """Get counter value."""
        full_key = f"{namespace}:{counter_name}"
        value = self.client.get(full_key)
        return int(value) if value else 0

    def reset_counter(self, counter_name: str, namespace: str = "aml:counters"):
        """Reset counter to zero."""
        full_key = f"{namespace}:{counter_name}"
        self.client.delete(full_key)

    def set_with_expiry(
        self, key: str, value: Any, expiry_seconds: int, namespace: str = "aml"
    ):
        """Set key with expiry time."""
        return self.set(key, value, ttl=expiry_seconds, namespace=namespace)

    def get_many(self, keys: List[str], namespace: str = "aml") -> Dict[str, Any]:
        """
        Get multiple keys in a single operation.

        Args:
            keys: List of cache keys
            namespace: Namespace prefix

        Returns:
            Dict mapping keys to values
        """
        full_keys = [f"{namespace}:{key}" for key in keys]

        try:
            pipe = self.client.pipeline()
            for fk in full_keys:
                pipe.get(fk)

            values = pipe.execute()

            result = {}
            for key, value in zip(keys, values):
                if value:
                    try:
                        result[key] = json.loads(value)
                    except Exception:
                        result[key] = None

            return result

        except Exception as e:
            logger.error(f"Cache GET_MANY error: {e}")
            return {}

    def flush_namespace(self, namespace: str = "aml"):
        """Flush all keys in a namespace."""
        try:
            pattern = f"{namespace}:*"
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)
                logger.info(f"Flushed {len(keys)} keys from namespace '{namespace}'")
        except Exception as e:
            logger.error(f"Error flushing namespace '{namespace}': {e}")

    def get_stats(self) -> Dict:
        """Get Redis cache statistics."""
        try:
            info = self.client.info("stats")
            return {
                "total_connections": info.get("total_connections_received", 0),
                "total_commands": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(info),
                "used_memory_human": self.client.info("memory").get(
                    "used_memory_human", "N/A"
                ),
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}

    def _calculate_hit_rate(self, info: Dict) -> float:
        """Calculate cache hit rate."""
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        return (hits / total * 100) if total > 0 else 0.0

    def _get_timestamp(self) -> str:
        """Get current UTC timestamp."""
        from datetime import datetime

        return datetime.utcnow().isoformat()

    def close(self):
        """Close Redis connection."""
        try:
            self.client.close()
            logger.info("Closed Redis connection")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")
