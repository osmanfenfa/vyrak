"""
Data ingestion layer for the Zelvryn ABSA system.
"""

from ingestion.shadow_mode import ShadowModeLogger, ShadowLogEntry
from ingestion.consent_logger import ConsentGatedLogger, ConsentRecord
from ingestion.corpus_loader import AttackCorpusLoader, CorpusRecord
from ingestion.stream_processor import StreamProcessor, IngestResult

__all__ = [
    "ShadowModeLogger",
    "ShadowLogEntry",
    "ConsentGatedLogger",
    "ConsentRecord",
    "AttackCorpusLoader",
    "CorpusRecord",
    "StreamProcessor",
    "IngestResult",
]

__version__ = "2.0.0"
__author__  = "Vyrak"