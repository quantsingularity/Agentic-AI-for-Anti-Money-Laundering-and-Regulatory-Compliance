"""
Kafka Consumer for Real-time Transaction Streaming
Handles distributed transaction processing at scale (10M+ txns/day)
"""

import json
import time
from typing import Callable, Dict, List, Optional

try:
    from kafka import KafkaConsumer
except ImportError:  # pragma: no cover - optional dependency
    KafkaConsumer = None
from loguru import logger


class TransactionStreamConsumer:
    """
    High-throughput Kafka consumer for AML transaction processing.
    Designed for 10M+ transactions/day with fault tolerance.
    """

    def __init__(
        self,
        bootstrap_servers: List[str],
        topic: str,
        group_id: str,
        config: Optional[Dict] = None,
    ):
        """
        Initialize Kafka consumer for transaction streaming.

        Args:
            bootstrap_servers: List of Kafka broker addresses
            topic: Topic name for transactions
            group_id: Consumer group ID for load balancing
            config: Additional Kafka configuration
        """
        self.topic = topic
        self.group_id = group_id
        self.config = config or {}

        # Performance-optimized Kafka settings
        consumer_config = {
            "bootstrap_servers": bootstrap_servers,
            "group_id": group_id,
            "auto_offset_reset": "earliest",
            "enable_auto_commit": False,  # Manual commit for reliability
            "max_poll_records": 500,  # Batch size for throughput
            "session_timeout_ms": 30000,
            "heartbeat_interval_ms": 10000,
            "value_deserializer": lambda m: json.loads(m.decode("utf-8")),
            "key_deserializer": lambda m: m.decode("utf-8") if m else None,
            **self.config,
        }

        if KafkaConsumer is None:  # pragma: no cover - optional dependency
            raise ImportError(
                "kafka-python is required for the streaming consumer. "
                "Install it with `pip install kafka-python`."
            )
        self.consumer = KafkaConsumer(topic, **consumer_config)
        self.running = False
        self.processed_count = 0
        self.error_count = 0
        self.last_commit_time = time.time()

        logger.info(
            f"Initialized Kafka consumer for topic '{topic}' in group '{group_id}'"
        )

    def consume_stream(
        self,
        processor: Callable[[List[Dict]], None],
        batch_size: int = 100,
        commit_interval: int = 5,
    ):
        """
        Consume transaction stream and process in batches.

        Args:
            processor: Callback function to process transaction batches
            batch_size: Number of transactions per processing batch
            commit_interval: Seconds between offset commits
        """
        self.running = True
        batch = []

        logger.info(f"Starting stream consumption from topic '{self.topic}'")

        try:
            while self.running:
                # Poll for messages
                msg_pack = self.consumer.poll(timeout_ms=1000, max_records=batch_size)

                for topic_partition, messages in msg_pack.items():
                    for message in messages:
                        try:
                            # Extract transaction data
                            transaction = message.value
                            transaction["_kafka_offset"] = message.offset
                            transaction["_kafka_partition"] = message.partition
                            transaction["_kafka_timestamp"] = message.timestamp

                            batch.append(transaction)

                            # Process batch when full
                            if len(batch) >= batch_size:
                                self._process_batch(batch, processor)
                                batch = []

                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
                            self.error_count += 1

                # Periodic commit
                if time.time() - self.last_commit_time >= commit_interval:
                    self.consumer.commit()
                    self.last_commit_time = time.time()
                    logger.debug(
                        f"Committed offsets. Processed: {self.processed_count}, Errors: {self.error_count}"
                    )

            # Process remaining batch
            if batch:
                self._process_batch(batch, processor)

        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user")
        except Exception as e:
            logger.error(f"Fatal consumer error: {e}")
            raise
        finally:
            self.shutdown()

    def _process_batch(self, batch: List[Dict], processor: Callable):
        """Process a batch of transactions."""
        try:
            start_time = time.time()
            processor(batch)
            duration = time.time() - start_time

            self.processed_count += len(batch)
            throughput = len(batch) / duration if duration > 0 else 0

            logger.info(
                f"Processed batch of {len(batch)} txns in {duration:.2f}s ({throughput:.0f} txns/s)"
            )

        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            self.error_count += len(batch)

    def shutdown(self):
        """Gracefully shutdown consumer."""
        logger.info("Shutting down Kafka consumer...")
        self.running = False
        self.consumer.close()
        logger.info(
            f"Consumer closed. Total processed: {self.processed_count}, Errors: {self.error_count}"
        )

    def get_metrics(self) -> Dict:
        """Get consumer metrics."""
        return {
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(1, self.processed_count),
            "topic": self.topic,
            "group_id": self.group_id,
        }


class TransactionStreamProducer:
    """
    Kafka producer for publishing transaction events.
    """

    def __init__(self, bootstrap_servers: List[str], topic: str):
        from kafka import KafkaProducer

        self.topic = topic
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks="all",  # Wait for all replicas
            retries=3,
            max_in_flight_requests_per_connection=5,
            compression_type="gzip",
        )

        logger.info(f"Initialized Kafka producer for topic '{topic}'")

    def publish_transaction(self, transaction: Dict, key: Optional[str] = None):
        """
        Publish a transaction to Kafka.

        Args:
            transaction: Transaction data dict
            key: Optional partition key (e.g., entity_id for ordering)
        """
        try:
            future = self.producer.send(self.topic, value=transaction, key=key)
            future.get(timeout=10)  # Block until sent
        except Exception as e:
            logger.error(f"Failed to publish transaction: {e}")
            raise

    def publish_batch(self, transactions: List[Dict], key_field: str = "entity_id"):
        """
        Publish a batch of transactions.

        Args:
            transactions: List of transaction dicts
            key_field: Field to use as partition key
        """
        for txn in transactions:
            key = str(txn.get(key_field, ""))
            self.producer.send(self.topic, value=txn, key=key)

        self.producer.flush()
        logger.info(f"Published batch of {len(transactions)} transactions")

    def close(self):
        """Close producer."""
        self.producer.close()
