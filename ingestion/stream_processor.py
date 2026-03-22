"""
zelvryn/ingestion/stream_processor.py
========================================
StreamProcessor — high-throughput action event ingestion for Zelvryn.

Handles two ingestion pathways:
  1. Live Kafka streams  — real-time event-by-event processing at scale
  2. Batch log replay    — historical processing of existing audit log files

When to use each pathway
-------------------------
  Kafka streams (process_kafka_event / start_kafka_consumer):
    Use in production when the host application publishes action events to a
    Kafka topic. This is the recommended path for deployments with > 1,000
    concurrent users. Kafka provides built-in:
      - Message ordering per partition (preserving session sequence)
      - At-least-once delivery guarantees
      - Consumer group scaling (multiple Zelvryn workers share load)
      - Replay from offset for recovery after downtime

  Batch log replay (batch_ingest_file / batch_ingest_directory):
    Use to retroactively process historical access logs that predate Zelvryn's
    deployment. Also used by run_shadow_mode.py to replay captured logs through
    the shadow evaluation stack without requiring Kafka.

Message format
--------------
Both pathways expect action events in JSON format:

  {
    "user_id":     "user_001",
    "session_id":  "sess_abc123",
    "action_type": "read",
    "resource_id": "document_42",
    "timestamp":   1735000000.0,
    "metadata": {
      "ip": "10.0.0.1",
      "device": "chrome-win11"
    }
  }

Processing architecture
------------------------
The StreamProcessor maintains a per-session engine cache (session_engines dict).
Each unique session_id gets its own ABSAv2Engine instance. Engines are evicted
from cache after SESSION_ENGINE_TTL seconds of inactivity to bound memory.

Author : Vyrak
System : Zelvryn — Adaptive Behavioral Signature Authentication
Version: 2.0
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

from core.action import UserAction
from core.engine import ABSAv2Engine, DecisionResult
from core.profile import BehavioralProfile
from core.classifier import RandomForestClassifier
from core.constants import SHADOW_MODE_ENABLED

logger = logging.getLogger(__name__)

# Seconds of inactivity after which a session engine is evicted from cache
SESSION_ENGINE_TTL: int = 3600   # 1 hour

# Maximum events to process in a single batch before yielding control
DEFAULT_BATCH_SIZE: int = 500

# Supported log file extensions for batch ingestion
SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".jsonl", ".json", ".log", ".txt", ".ndjson")


@dataclass
class IngestResult:
    """
    Result of processing a single event through the StreamProcessor.

    Attributes
    ----------
    session_id : str
    user_id : str
    action_type : str
    timestamp : float
    decision : str
        ALLOW, CHALLENGE, or BLOCK.
    trust_score : float
    bot_prob : float
    anomalies : list of str
    processing_latency_ms : float
        Wall-clock milliseconds from event receipt to decision return.
    error : str or None
        Error message if processing failed, None on success.
    """
    session_id:            str
    user_id:               str
    action_type:           str
    timestamp:             float
    decision:              str              = "ALLOW"
    trust_score:           float            = 100.0
    bot_prob:              float            = 0.0
    anomalies:             List[str]        = field(default_factory=list)
    processing_latency_ms: float            = 0.0
    error:                 Optional[str]    = None

    def is_success(self) -> bool:
        """Return True if processing completed without error."""
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id":            self.session_id,
            "user_id":               self.user_id,
            "action_type":           self.action_type,
            "timestamp":             self.timestamp,
            "decision":              self.decision,
            "trust_score":           round(self.trust_score, 2),
            "bot_prob":              round(self.bot_prob, 4),
            "anomalies":             self.anomalies,
            "processing_latency_ms": round(self.processing_latency_ms, 2),
            "error":                 self.error,
        }


class StreamProcessor:
    """
    High-throughput action event processor supporting Kafka and batch ingestion.

    Maintains a cache of per-session ABSAv2Engine instances. New sessions are
    initialised with a profile from the profile store (Redis in production,
    or an in-memory dict in testing). Inactive sessions are evicted after
    SESSION_ENGINE_TTL seconds.

    Parameters
    ----------
    profile_loader : callable, optional
        Function that takes a user_id string and returns a BehavioralProfile.
        Signature: (user_id: str) -> BehavioralProfile
        If None, creates empty profiles for all users (suitable for testing).
    classifier : RandomForestClassifier, optional
        Shared classifier instance used for all sessions. In production this
        is the active model from the model registry.
    shadow_mode : bool, optional
        If True, all enforcement is suppressed. Overrides the global constant.
    engine_ttl : int, optional
        Seconds of inactivity before a session engine is evicted. Default 3600.
    on_decision_callback : callable, optional
        Called after every evaluation with the IngestResult. Use to write
        decisions to a database, publish to another Kafka topic, or trigger
        webhook alerts.
        Signature: (result: IngestResult) -> None
    on_block_callback : callable, optional
        Called when any session receives a BLOCK decision. Use for immediate
        security alerts.
        Signature: (result: IngestResult) -> None

    Attributes
    ----------
    _session_engines : dict
        {session_id: (ABSAv2Engine, last_active_timestamp)}
    processed_count : int
    error_count : int
    block_count : int
    """

    def __init__(
        self,
        profile_loader: Optional[Callable[[str], BehavioralProfile]] = None,
        classifier: Optional[RandomForestClassifier] = None,
        shadow_mode: Optional[bool] = None,
        engine_ttl: int = SESSION_ENGINE_TTL,
        on_decision_callback: Optional[Callable[["IngestResult"], None]] = None,
        on_block_callback:    Optional[Callable[["IngestResult"], None]] = None,
    ) -> None:
        self._profile_loader        = profile_loader or self._default_profile_loader
        self._classifier            = classifier
        self._shadow_mode           = (
            shadow_mode if shadow_mode is not None else SHADOW_MODE_ENABLED
        )
        self._engine_ttl            = engine_ttl
        self._on_decision_callback  = on_decision_callback
        self._on_block_callback     = on_block_callback

        # {session_id: (ABSAv2Engine, last_active_ts)}
        self._session_engines: Dict[str, Tuple[ABSAv2Engine, float]] = {}

        # Statistics
        self.processed_count: int = 0
        self.error_count:     int = 0
        self.block_count:     int = 0
        self.challenge_count: int = 0

        logger.info(
            "StreamProcessor initialised: shadow=%s ttl=%ds",
            self._shadow_mode, self._engine_ttl
        )

    # ─────────────────────────────────────────────────────────────────────────
    # SINGLE EVENT PROCESSING
    # ─────────────────────────────────────────────────────────────────────────

    def process_event(self, event: Dict[str, Any]) -> IngestResult:
        """
        Process a single action event JSON dict.

        This is the core processing method called by both the Kafka consumer
        loop and the batch ingestion pipeline.

        Parameters
        ----------
        event : dict
            Action event dict with keys: user_id, session_id, action_type,
            resource_id, timestamp (optional), metadata (optional).

        Returns
        -------
        IngestResult
        """
        t_start = time.time()

        # Parse required fields
        user_id    = str(event.get("user_id", "unknown"))
        session_id = str(event.get("session_id", f"{user_id}_session"))
        action_str = str(event.get("action_type", "read"))
        resource   = str(event.get("resource_id", "unknown"))
        timestamp  = float(event.get("timestamp") or t_start)
        metadata   = event.get("metadata", {}) or {}

        # Build result skeleton
        result = IngestResult(
            session_id  = session_id,
            user_id     = user_id,
            action_type = action_str,
            timestamp   = timestamp,
        )

        try:
            # Validate action type
            if action_str not in UserAction.ACTION_TYPES:
                action_str = "read"   # default fallback for unknown types
                logger.debug(
                    "Unknown action_type %r for user=%s — defaulted to 'read'",
                    event.get("action_type"), user_id
                )

            # Construct UserAction
            action = UserAction(
                action_type = action_str,
                resource_id = resource,
                session_id  = session_id,
                timestamp   = timestamp,
                metadata    = metadata,
            )

            # Get or create session engine
            engine = self._get_or_create_engine(session_id, user_id)

            # Evaluate
            decision_result: DecisionResult = engine.evaluate_action(action)

            # Populate result
            result.decision    = decision_result.decision
            result.trust_score = decision_result.trust_score
            result.bot_prob    = decision_result.bot_prob
            result.anomalies   = list(decision_result.anomalies)

            # Update statistics
            self.processed_count += 1
            if decision_result.decision == "BLOCK":
                self.block_count += 1
                if self._on_block_callback:
                    try:
                        self._on_block_callback(result)
                    except Exception as cb_err:
                        logger.warning("on_block_callback error: %s", cb_err)
            elif decision_result.decision == "CHALLENGE":
                self.challenge_count += 1

        except Exception as e:
            self.error_count += 1
            result.error = str(e)
            logger.error(
                "Event processing error: user=%s session=%s error=%s",
                user_id, session_id, e, exc_info=True
            )

        result.processing_latency_ms = (time.time() - t_start) * 1000

        # Fire decision callback
        if self._on_decision_callback:
            try:
                self._on_decision_callback(result)
            except Exception as cb_err:
                logger.warning("on_decision_callback error: %s", cb_err)

        return result

    def process_json_string(self, json_str: str) -> IngestResult:
        """
        Parse a JSON string and process it as an action event.

        Convenience wrapper for Kafka consumer message values which arrive
        as byte strings or decoded strings.

        Parameters
        ----------
        json_str : str
            JSON-encoded action event.

        Returns
        -------
        IngestResult
        """
        try:
            event = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON event: %s | error: %s", json_str[:200], e)
            return IngestResult(
                session_id  = "unknown",
                user_id     = "unknown",
                action_type = "unknown",
                timestamp   = time.time(),
                error       = f"JSON parse error: {e}",
            )
        return self.process_event(event)

    # ─────────────────────────────────────────────────────────────────────────
    # KAFKA STREAM INGESTION
    # ─────────────────────────────────────────────────────────────────────────

    def start_kafka_consumer(
        self,
        topic: str,
        bootstrap_servers: str,
        group_id: str = "zelvryn-consumer",
        auto_offset_reset: str = "latest",
        max_poll_records: int = 500,
        run_forever: bool = True,
    ) -> None:
        """
        Start consuming action events from a Kafka topic.

        Requires the `kafka-python` library (pip install kafka-python).
        Each message is processed synchronously in the consumer loop.
        For higher throughput, run multiple consumer instances in separate
        processes (Kafka's consumer group handles partition assignment).

        Parameters
        ----------
        topic : str
            Kafka topic name (e.g., "zelvryn.action.events").
        bootstrap_servers : str
            Comma-separated Kafka broker addresses (e.g., "localhost:9092").
        group_id : str, optional
            Consumer group identifier. All consumers with the same group_id
            share the topic partitions. Default "zelvryn-consumer".
        auto_offset_reset : str, optional
            "latest" (skip historical messages) or "earliest" (replay all).
            Default "latest" for live deployment.
        max_poll_records : int, optional
            Max records fetched per poll. Default 500.
        run_forever : bool, optional
            If True, loops indefinitely. Set False for testing. Default True.
        """
        try:
            from kafka import KafkaConsumer
        except ImportError:
            raise ImportError(
                "kafka-python required for Kafka ingestion. "
                "Install with: pip install kafka-python"
            )

        logger.info(
            "Starting Kafka consumer: topic=%s servers=%s group=%s",
            topic, bootstrap_servers, group_id
        )

        consumer = KafkaConsumer(
            topic,
            bootstrap_servers = bootstrap_servers.split(","),
            group_id          = group_id,
            auto_offset_reset = auto_offset_reset,
            max_poll_records  = max_poll_records,
            value_deserializer= lambda v: v.decode("utf-8", errors="replace"),
            enable_auto_commit = True,
        )

        try:
            while True:
                records = consumer.poll(timeout_ms=1000)
                for topic_partition, messages in records.items():
                    for message in messages:
                        self.process_json_string(message.value)

                self._evict_stale_engines()

                if not run_forever:
                    break

        except KeyboardInterrupt:
            logger.info("Kafka consumer stopped by KeyboardInterrupt.")
        finally:
            consumer.close()
            logger.info(
                "Kafka consumer closed. Processed=%d Errors=%d Blocks=%d",
                self.processed_count, self.error_count, self.block_count
            )

    def process_kafka_event(self, message_value: str) -> IngestResult:
        """
        Process a single Kafka message value string.

        Convenience method for testing Kafka integration without a running
        Kafka cluster.

        Parameters
        ----------
        message_value : str
            Raw message value (JSON string).

        Returns
        -------
        IngestResult
        """
        return self.process_json_string(message_value)

    # ─────────────────────────────────────────────────────────────────────────
    # BATCH LOG INGESTION
    # ─────────────────────────────────────────────────────────────────────────

    def batch_ingest_file(
        self,
        file_path: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Generator[List[IngestResult], None, None]:
        """
        Process a log file in batches, yielding results per batch.

        Supports .jsonl files (one JSON object per line) and JSON arrays.
        Large files are streamed line by line to avoid memory exhaustion.

        Parameters
        ----------
        file_path : str
            Path to the log file.
        batch_size : int, optional
            Number of events to process before yielding. Default 500.
        progress_callback : callable, optional
            Called after each batch with (events_processed, total_events).
            total_events is -1 if unknown.

        Yields
        ------
        list of IngestResult
            One list per batch.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Log file not found: {file_path}")

        ext = Path(file_path).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            logger.warning(
                "File extension %r may not be supported. Attempting JSONL parse.", ext
            )

        logger.info("Batch ingesting file: %s (batch_size=%d)", file_path, batch_size)
        batch        = []
        total_lines  = self._count_lines(file_path)
        processed    = 0

        for event in self._stream_jsonl(file_path):
            result = self.process_event(event)
            batch.append(result)
            processed += 1

            if len(batch) >= batch_size:
                if progress_callback:
                    progress_callback(processed, total_lines)
                yield batch
                batch = []

            # Periodic engine eviction during long batch jobs
            if processed % 10_000 == 0:
                self._evict_stale_engines()
                logger.info(
                    "Batch progress: %d/%s processed "
                    "(blocks=%d errors=%d)",
                    processed,
                    total_lines if total_lines > 0 else "?",
                    self.block_count, self.error_count
                )

        if batch:
            yield batch

        logger.info(
            "Batch ingest complete: %d events processed from %s "
            "(blocks=%d errors=%d)",
            processed, file_path, self.block_count, self.error_count
        )

    def batch_ingest_directory(
        self,
        directory: str,
        pattern: str = "*.jsonl",
        recursive: bool = False,
    ) -> Generator[IngestResult, None, None]:
        """
        Process all matching log files in a directory.

        Parameters
        ----------
        directory : str
            Directory to scan.
        pattern : str, optional
            Glob pattern for file matching. Default "*.jsonl".
        recursive : bool, optional
            If True, also scan subdirectories. Default False.

        Yields
        ------
        IngestResult
            One per processed event across all files.
        """
        import glob
        search = f"{'**/' if recursive else ''}{pattern}"
        files  = glob.glob(os.path.join(directory, search), recursive=recursive)

        if not files:
            logger.warning("No files matching %r found in %s", pattern, directory)
            return

        logger.info("Processing %d files from %s", len(files), directory)

        for file_path in sorted(files):
            logger.info("Processing file: %s", file_path)
            for batch in self.batch_ingest_file(file_path):
                for result in batch:
                    yield result

    def ingest_events_list(
        self,
        events: List[Dict[str, Any]],
    ) -> List[IngestResult]:
        """
        Process a list of event dicts in memory.

        Suitable for unit testing and small-scale offline processing.

        Parameters
        ----------
        events : list of dict

        Returns
        -------
        list of IngestResult
        """
        return [self.process_event(event) for event in events]

    # ─────────────────────────────────────────────────────────────────────────
    # SESSION ENGINE MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    def _get_or_create_engine(
        self,
        session_id: str,
        user_id: str,
    ) -> ABSAv2Engine:
        """
        Return the engine for a session, creating one if it does not exist.

        Parameters
        ----------
        session_id : str
        user_id : str

        Returns
        -------
        ABSAv2Engine
        """
        now = time.time()

        if session_id in self._session_engines:
            engine, _ = self._session_engines[session_id]
            # Update last-active timestamp
            self._session_engines[session_id] = (engine, now)
            return engine

        # New session — load profile and create engine
        profile = self._profile_loader(user_id)
        engine  = ABSAv2Engine(
            profile    = profile,
            classifier = self._classifier,
            session_id = session_id,
            shadow_mode = self._shadow_mode,
        )
        self._session_engines[session_id] = (engine, now)

        logger.debug(
            "New session engine created: session=%s user=%s "
            "(active_sessions=%d)",
            session_id, user_id, len(self._session_engines)
        )
        return engine

    def _evict_stale_engines(self) -> int:
        """
        Remove session engines that have been inactive longer than engine_ttl.

        Returns
        -------
        int
            Number of engines evicted.
        """
        now    = time.time()
        cutoff = now - self._engine_ttl

        stale_sessions = [
            sid for sid, (_, last_active) in self._session_engines.items()
            if last_active < cutoff
        ]

        for sid in stale_sessions:
            del self._session_engines[sid]

        if stale_sessions:
            logger.debug(
                "Evicted %d stale session engines (remaining=%d)",
                len(stale_sessions), len(self._session_engines)
            )

        return len(stale_sessions)

    def close_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Explicitly close and archive a session engine.

        Returns the session summary dict and removes the engine from cache.

        Parameters
        ----------
        session_id : str

        Returns
        -------
        dict or None
            Session summary from engine.get_session_summary(), or None if
            the session was not found.
        """
        if session_id not in self._session_engines:
            return None

        engine, _ = self._session_engines.pop(session_id)
        summary   = engine.get_session_summary()

        logger.info(
            "Session closed: session=%s user=%s final_trust=%.1f "
            "total_actions=%d",
            session_id, engine.profile.user_id,
            summary["final_trust"], summary["total_actions"]
        )
        return summary

    def get_active_session_count(self) -> int:
        """Return the number of currently active session engines in cache."""
        return len(self._session_engines)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Return overall processing statistics.

        Returns
        -------
        dict
        """
        return {
            "processed_count":    self.processed_count,
            "error_count":        self.error_count,
            "block_count":        self.block_count,
            "challenge_count":    self.challenge_count,
            "active_sessions":    len(self._session_engines),
            "error_rate_pct":     round(
                100 * self.error_count / max(self.processed_count, 1), 2
            ),
            "block_rate_pct":     round(
                100 * self.block_count / max(self.processed_count, 1), 2
            ),
            "shadow_mode":        self._shadow_mode,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _default_profile_loader(user_id: str) -> BehavioralProfile:
        """
        Default profile loader: creates an empty profile for any user.

        In production, override with a loader that fetches from Redis:
          async def redis_loader(user_id: str) -> BehavioralProfile:
              store = RedisProfileStore(redis_url)
              return await store.get(user_id) or BehavioralProfile(user_id)
        """
        logger.debug(
            "Default profile loader: creating empty profile for user=%s", user_id
        )
        return BehavioralProfile(user_id=user_id)

    @staticmethod
    def _stream_jsonl(path: str) -> Generator[Dict[str, Any], None, None]:
        """
        Stream JSON objects from a file, one per line (JSONL format).

        Also handles plain JSON arrays by loading the whole file.
        Skips blank lines and lines starting with '#'.

        Parameters
        ----------
        path : str

        Yields
        ------
        dict
        """
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            first_char = f.read(1)
            f.seek(0)

            if first_char == "[":
                # JSON array format
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                yield item
                except json.JSONDecodeError as e:
                    logger.error("Failed to parse JSON array from %s: %s", path, e)
                return

            # JSONL format
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        yield obj
                except json.JSONDecodeError as e:
                    logger.debug(
                        "Skipping malformed JSON on line %d of %s: %s",
                        line_num, path, e
                    )

    @staticmethod
    def _count_lines(path: str) -> int:
        """Count lines in a file efficiently. Returns -1 on error."""
        try:
            count = 0
            with open(path, "rb") as f:
                for _ in f:
                    count += 1
            return count
        except OSError:
            return -1

    # ─────────────────────────────────────────────────────────────────────────
    # Dunder methods
    # ─────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"StreamProcessor("
            f"processed={self.processed_count}, "
            f"active_sessions={len(self._session_engines)}, "
            f"shadow={self._shadow_mode})"
        )

    def __del__(self) -> None:
        """Log final statistics when the processor is garbage collected."""
        try:
            logger.info(
                "StreamProcessor shutting down: %s",
                self.get_statistics()
            )
        except Exception:
            pass   # avoid errors during interpreter shutdown
