"""
zelvryn/ingestion/corpus_loader.py
=====================================
AttackCorpusLoader — loads, processes, and converts real-world attack traffic
datasets into Zelvryn's 20-dimensional feature vector format for ML training.

Why real attack data matters
-----------------------------
The ABSA v2 Random Forest is initially seeded with synthetic bot data generated
by scripts/train_initial_model.py. Synthetic data captures obvious patterns
(regular timing, high export frequency) but misses the nuanced, evolving
evasion strategies used by real attackers. Real attack corpora provide:

  - Actual credential stuffing traffic (CICIDS2018 Brute Force datasets)
  - Real botnet C&C communication patterns (CTU-13 captures)
  - Time-correlated attack sequences that reveal true attacker pacing
  - Edge cases and anomalies that synthetic generators cannot anticipate

Supported datasets
------------------
  CICIDS2018
    Canadian Institute for Cybersecurity 2018 Intrusion Detection Dataset.
    CSV files with 80 network traffic features and LABEL column.
    Download: https://www.unb.ca/cic/datasets/ids-2018.html
    Relevant files: Friday-02-03-2018_TrafficForML_CICFlowMeter.csv
                    (contains BruteForce-Web and BruteForce-XSS attacks)

  CTU-13
    Czech Technical University botnet capture dataset.
    CSV files with binetflow format (src IP, dst IP, duration, packets, bytes,
    label). Scenarios 1-13 available.
    Download: https://www.stratosphereips.org/datasets-ctu13

Feature mapping strategy
------------------------
These datasets capture network-level traffic features, not application-level
action sequences. The conversion maps network traffic statistics to the ABSA
feature space using a principled heuristic:

  - High packet rate → low mean_interval, high burst_score
  - Repeated destination IP/port → low resource_div
  - Byte transfer volume → sensitive_prop proxy
  - Flow duration → max_interval proxy
  - Attack traffic → high entropy (many different connection patterns)

The mapping is approximate but sufficient to seed the classifier with a
directionally correct bot feature distribution. Once real shadow session data
accumulates, it supersedes the corpus-derived training samples.

Author : Vyrak
System : Zelvryn — Adaptive Behavioral Signature Authentication
Version: 2.0
"""

import csv
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple

from core.constants import ML_FEATURE_DIM
from core.features import FeatureExtractor

logger = logging.getLogger(__name__)

# CICIDS2018 column names (may vary slightly between CSV files)
_CICIDS_LABEL_COL      = "Label"
_CICIDS_BENIGN_LABELS  = {"Benign", "BENIGN", "benign"}
_CICIDS_MAX_ROWS       = 100_000   # cap to prevent memory exhaustion

# CTU-13 binetflow label column
_CTU13_LABEL_COL       = "Label"
_CTU13_BOTNET_PREFIX   = "Botnet"
_CTU13_NORMAL_LABELS   = {"LEGITIMATE", "Normal", "Background"}

# Chunk size for streaming large CSV files
_CSV_CHUNK_SIZE        = 1000


@dataclass
class CorpusRecord:
    """
    A single labeled record from an attack corpus, converted to ABSA format.

    Attributes
    ----------
    source_dataset : str
        Dataset name ("cicids2018" or "ctu13").
    original_label : str
        Original label string from the dataset.
    zelvryn_label : int
        0 = benign (human), 1 = attack (bot).
    feature_vector : list of float
        20-dimensional ABSA feature vector.
    metadata : dict
        Additional fields from the original record for reference.
    """
    source_dataset:  str
    original_label:  str
    zelvryn_label:   int
    feature_vector:  List[float]
    metadata:        Dict[str, Any]

    def to_training_pair(self) -> Tuple[List[float], int]:
        """Return (feature_vector, label) for ML training."""
        return (self.feature_vector, self.zelvryn_label)


class AttackCorpusLoader:
    """
    Loads and converts attack corpus datasets to Zelvryn feature vectors.

    All load methods return generators to avoid loading entire large
    datasets into memory at once.

    Parameters
    ----------
    max_records_per_file : int, optional
        Maximum records to load from any single file. Default 100,000.
    balance_classes : bool, optional
        If True, limit the benign class to the same count as the attack class
        to produce a balanced training dataset. Default True.
    verbose : bool, optional
        If True, log progress every 10,000 records. Default True.
    """

    def __init__(
        self,
        max_records_per_file: int = _CICIDS_MAX_ROWS,
        balance_classes: bool = True,
        verbose: bool = True,
    ) -> None:
        self.max_records_per_file = max_records_per_file
        self.balance_classes      = balance_classes
        self.verbose              = verbose

    # ─────────────────────────────────────────────────────────────────────────
    # CICIDS2018
    # ─────────────────────────────────────────────────────────────────────────

    def load_cicids2018(
        self,
        path: str,
        label_col: str = _CICIDS_LABEL_COL,
    ) -> List[CorpusRecord]:
        """
        Load and convert a CICIDS2018 CSV file to CorpusRecord objects.

        The CICIDS2018 dataset contains 80+ network flow features. This method
        extracts the subset relevant to ABSA's feature space and maps them to
        the 20-dimensional vector format.

        Parameters
        ----------
        path : str
            Path to a CICIDS2018 CSV file.
        label_col : str, optional
            Column name containing the traffic label.

        Returns
        -------
        list of CorpusRecord
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"CICIDS2018 file not found: {path}")

        logger.info("Loading CICIDS2018 from %s", path)
        records = []
        benign_count = 0
        attack_count = 0
        row_count = 0

        for row in self._stream_csv(path):
            if row_count >= self.max_records_per_file:
                break
            row_count += 1

            label_str = row.get(label_col, "").strip()
            if not label_str:
                continue

            zelvryn_label = 0 if label_str in _CICIDS_BENIGN_LABELS else 1

            try:
                feature_vector = self._cicids_to_feature_vector(row)
            except (ValueError, KeyError) as e:
                logger.debug("Skipping malformed CICIDS row %d: %s", row_count, e)
                continue

            if zelvryn_label == 0:
                benign_count += 1
            else:
                attack_count += 1

            records.append(CorpusRecord(
                source_dataset = "cicids2018",
                original_label = label_str,
                zelvryn_label  = zelvryn_label,
                feature_vector = feature_vector,
                metadata       = {
                    "row": row_count,
                    "original_label": label_str,
                },
            ))

            if self.verbose and row_count % 10_000 == 0:
                logger.info(
                    "CICIDS2018: %d rows processed (%d benign, %d attack)",
                    row_count, benign_count, attack_count
                )

        if self.balance_classes:
            records = self._balance(records)

        logger.info(
            "CICIDS2018 loaded: %d records (%d benign, %d attack) from %s",
            len(records),
            sum(1 for r in records if r.zelvryn_label == 0),
            sum(1 for r in records if r.zelvryn_label == 1),
            path
        )
        return records

    def _cicids_to_feature_vector(self, row: Dict[str, str]) -> List[float]:
        """
        Map CICIDS2018 network flow features to the 20-dim ABSA feature vector.

        Mapping rationale:
          Network flow statistics are not directly equivalent to user action
          sequences, but they share analogous structure:
            - Flow duration → mean_interval proxy
            - Packet rate → burst_score proxy
            - Destination port diversity → resource_div proxy
            - Protocol mix → entropy proxy
            - Flow bytes → sensitive_prop proxy (data transfer = export-like)

        All values are normalised to [0.0, 1.0] using domain-appropriate scales.

        Parameters
        ----------
        row : dict of str -> str
            A single CSV row from CICIDS2018.

        Returns
        -------
        list of float
            20-dimensional feature vector.
        """
        def safe_float(key: str, default: float = 0.0, scale: float = 1.0) -> float:
            try:
                val = float(row.get(key, default))
                return min(1.0, max(0.0, val / scale))
            except (ValueError, TypeError):
                return default

        # ── Action type frequency proxies [0-7] ───────────────────────────
        # Map protocol and traffic type to action type analogues
        protocol = str(row.get("Protocol", "0")).strip()
        fwd_pkts = safe_float("Total Fwd Packets", 0, 1000)
        bwd_pkts = safe_float("Total Backward Packets", 0, 1000)

        freq_login  = safe_float("Init_Win_bytes_forward", 0, 65535)   # initial handshake
        freq_read   = max(0.0, fwd_pkts - bwd_pkts)                    # reads > writes
        freq_write  = bwd_pkts
        freq_delete = 0.0   # no direct analogue in network flows
        freq_search = safe_float("Packet Length Variance", 0, 10000)   # diverse packets
        freq_export = safe_float("Total Length of Fwd Packets", 0, 1e6)  # large transfers
        freq_config = safe_float("Init_Win_bytes_backward", 0, 65535)
        freq_admin  = safe_float("Fwd URG Flags", 0, 10)               # urgent flags

        # Normalise so they sum to 1.0
        freq_vec = [freq_login, freq_read, freq_write, freq_delete,
                    freq_search, freq_export, freq_config, freq_admin]
        total = sum(freq_vec)
        if total > 0:
            freq_vec = [v / total for v in freq_vec]
        else:
            freq_vec = [1.0 / 8] * 8

        # ── Entropy [8] ────────────────────────────────────────────────────
        pkt_len_mean = safe_float("Packet Length Mean", 0, 1500)
        pkt_len_std  = safe_float("Packet Length Std", 0, 1500)
        entropy = min(1.0, pkt_len_std / max(pkt_len_mean, 1.0))

        # ── Interval statistics [9-13] ────────────────────────────────────
        flow_duration_us = float(row.get("Flow Duration", 1e6) or 1e6)
        fwd_iat_mean     = float(row.get("Fwd IAT Mean", 1e6) or 1e6)
        fwd_iat_std      = float(row.get("Fwd IAT Std", 0) or 0)

        # Convert microseconds to seconds and normalise
        mean_interval = min(30.0, fwd_iat_mean / 1e6) / 30.0
        std_interval  = min(30.0, fwd_iat_std / 1e6) / 30.0
        cv_interval   = std_interval / max(mean_interval, 0.001)
        min_interval  = min(1.0, float(row.get("Fwd IAT Min", 0) or 0) / 1e7)
        max_interval  = min(1.0, float(row.get("Fwd IAT Max", 0) or 0) / 1e8)

        # ── Sensitive proportion [14] ──────────────────────────────────────
        # High byte transfer = export-like behaviour
        sensitive_prop = safe_float("Total Length of Fwd Packets", 0, 1e6)

        # ── Autocorrelation [15] ───────────────────────────────────────────
        # Regular packet timing = high autocorrelation (bot signal)
        autocorr = max(-1.0, min(1.0, 1.0 - cv_interval))

        # ── Resource diversity [16] ────────────────────────────────────────
        # Different destination ports = high diversity (human-like browsing)
        dst_port = int(row.get("Dst Port", 80) or 80)
        # Well-known ports 80/443/22 → moderate diversity
        port_diversity = 0.5 if dst_port in (80, 443, 22, 8080) else min(1.0, dst_port / 65535)

        # ── Burst score [17] ───────────────────────────────────────────────
        pkt_rate = safe_float("Flow Pkts/s", 0, 10000)
        burst_score = min(1.0, pkt_rate / 1000.0)

        # ── Run-length [18] ────────────────────────────────────────────────
        # ACK flag rate as proxy for repeated same-type packets
        ack_rate = safe_float("ACK Flag Count", 0, 100)
        run_length = max(1.0, ack_rate * 10)

        # ── Quantile ratio [19] ────────────────────────────────────────────
        # Regular timing = high Q1/Q3 ratio
        quantile_ratio = max(0.0, min(1.0, autocorr))

        vector = (
            freq_vec
            + [entropy, mean_interval, std_interval, cv_interval,
               min_interval, max_interval,
               sensitive_prop, autocorr, port_diversity,
               burst_score, run_length, quantile_ratio]
        )

        assert len(vector) == ML_FEATURE_DIM, (
            f"Feature vector length mismatch: {len(vector)}"
        )
        # Sanitise
        return [float(v) if (not (v != v) and abs(v) != float("inf")) else 0.0
                for v in vector]

    # ─────────────────────────────────────────────────────────────────────────
    # CTU-13
    # ─────────────────────────────────────────────────────────────────────────

    def load_ctu13(
        self,
        path: str,
        label_col: str = _CTU13_LABEL_COL,
    ) -> List[CorpusRecord]:
        """
        Load and convert a CTU-13 binetflow CSV file to CorpusRecord objects.

        CTU-13 contains labeled botnet network flows. Each row represents
        one network flow with features like duration, protocol, bytes, packets.

        Parameters
        ----------
        path : str
            Path to a CTU-13 binetflow CSV file.
        label_col : str, optional
            Column name for the traffic label.

        Returns
        -------
        list of CorpusRecord
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"CTU-13 file not found: {path}")

        logger.info("Loading CTU-13 from %s", path)
        records   = []
        row_count = 0

        for row in self._stream_csv(path):
            if row_count >= self.max_records_per_file:
                break
            row_count += 1

            label_str = row.get(label_col, "").strip()
            if not label_str:
                continue

            # CTU-13: labels starting with "Botnet" are attacks
            zelvryn_label = (
                1 if label_str.startswith(_CTU13_BOTNET_PREFIX) else 0
            )

            try:
                feature_vector = self._ctu13_to_feature_vector(row)
            except (ValueError, KeyError) as e:
                logger.debug("Skipping malformed CTU-13 row %d: %s", row_count, e)
                continue

            records.append(CorpusRecord(
                source_dataset = "ctu13",
                original_label = label_str,
                zelvryn_label  = zelvryn_label,
                feature_vector = feature_vector,
                metadata       = {"row": row_count, "original_label": label_str},
            ))

            if self.verbose and row_count % 10_000 == 0:
                logger.info("CTU-13: %d rows processed", row_count)

        if self.balance_classes:
            records = self._balance(records)

        logger.info(
            "CTU-13 loaded: %d records (%d benign, %d botnet) from %s",
            len(records),
            sum(1 for r in records if r.zelvryn_label == 0),
            sum(1 for r in records if r.zelvryn_label == 1),
            path
        )
        return records

    def _ctu13_to_feature_vector(self, row: Dict[str, str]) -> List[float]:
        """
        Map CTU-13 binetflow features to the 20-dim ABSA feature vector.

        CTU-13 binetflow columns: StartTime, Dur, Proto, SrcAddr, Sport,
        Dir, DstAddr, Dport, State, sTos, dTos, TotPkts, TotBytes,
        SrcBytes, Label.

        Parameters
        ----------
        row : dict of str -> str

        Returns
        -------
        list of float
        """
        def safe_f(key: str, default: float = 0.0, scale: float = 1.0) -> float:
            try:
                return min(1.0, max(0.0, float(row.get(key, default) or default) / scale))
            except (ValueError, TypeError):
                return default

        proto    = str(row.get("Proto", "tcp")).lower()
        dur      = float(row.get("Dur", 1.0) or 1.0)
        tot_pkts = float(row.get("TotPkts", 1) or 1)
        tot_byt  = float(row.get("TotBytes", 0) or 0)
        src_byt  = float(row.get("SrcBytes", 0) or 0)

        # ── Frequency proxies ──────────────────────────────────────────────
        is_tcp  = 1.0 if proto == "tcp" else 0.0
        is_udp  = 1.0 if proto == "udp" else 0.0
        is_icmp = 1.0 if proto == "icmp" else 0.0
        byt_ratio = min(1.0, src_byt / max(tot_byt, 1))

        freq_vec = [
            is_tcp * 0.3,                          # login (connection setup)
            is_tcp * (1 - byt_ratio),              # read
            is_tcp * byt_ratio,                    # write
            is_icmp,                               # delete / probe
            is_udp,                                # search / DNS
            min(1.0, tot_byt / 1e6),               # export (large transfer)
            safe_f("sTos", 0, 255),                # config (TOS field)
            safe_f("dTos", 0, 255),                # admin
        ]
        total = sum(freq_vec)
        if total > 0:
            freq_vec = [v / total for v in freq_vec]
        else:
            freq_vec = [0.125] * 8

        # ── Derived features ───────────────────────────────────────────────
        pkt_rate    = tot_pkts / max(dur, 0.001)
        byt_per_pkt = tot_byt  / max(tot_pkts, 1)

        entropy       = min(1.0, abs(byt_ratio - 0.5) * 2)
        mean_interval = min(1.0, dur / (tot_pkts * 30))
        std_interval  = min(1.0, 1.0 / max(pkt_rate, 0.001) / 100)
        cv_interval   = std_interval / max(mean_interval, 0.001)
        min_interval  = min(1.0, 1.0 / max(pkt_rate * 2, 0.001) / 100)
        max_interval  = min(1.0, dur / 100)
        sensitive_prop = min(1.0, tot_byt / max(src_byt * 10, 1))
        autocorr      = min(1.0, 1.0 - std_interval)
        resource_div  = 0.5   # binetflow has single dst per record
        burst_score   = min(1.0, pkt_rate / 100)
        run_length    = min(10.0, max(1.0, tot_pkts / 10))
        q_ratio       = min(1.0, max(0.0, autocorr))

        vector = (
            freq_vec
            + [entropy, mean_interval, std_interval, cv_interval,
               min_interval, max_interval,
               sensitive_prop, autocorr, resource_div,
               burst_score, run_length, q_ratio]
        )

        assert len(vector) == ML_FEATURE_DIM
        return [float(v) if (not (v != v) and abs(v) != float("inf")) else 0.0
                for v in vector]

    # ─────────────────────────────────────────────────────────────────────────
    # COMBINED LOADING AND EXPORT
    # ─────────────────────────────────────────────────────────────────────────

    def load_all(
        self,
        cicids_paths: Optional[List[str]] = None,
        ctu13_paths:  Optional[List[str]] = None,
    ) -> List[CorpusRecord]:
        """
        Load from all specified corpus files and combine into a single list.

        Parameters
        ----------
        cicids_paths : list of str, optional
        ctu13_paths : list of str, optional

        Returns
        -------
        list of CorpusRecord
        """
        all_records: List[CorpusRecord] = []

        for path in (cicids_paths or []):
            try:
                all_records.extend(self.load_cicids2018(path))
            except FileNotFoundError as e:
                logger.warning("%s", e)

        for path in (ctu13_paths or []):
            try:
                all_records.extend(self.load_ctu13(path))
            except FileNotFoundError as e:
                logger.warning("%s", e)

        if self.balance_classes:
            all_records = self._balance(all_records)

        logger.info(
            "Total corpus records: %d (%d benign, %d attack)",
            len(all_records),
            sum(1 for r in all_records if r.zelvryn_label == 0),
            sum(1 for r in all_records if r.zelvryn_label == 1),
        )
        return all_records

    def to_training_arrays(
        self,
        records: List[CorpusRecord],
    ) -> Tuple[List[List[float]], List[int]]:
        """
        Convert a list of CorpusRecords to (X, y) training arrays.

        Parameters
        ----------
        records : list of CorpusRecord

        Returns
        -------
        tuple of (list of list, list of int)
        """
        X = [r.feature_vector for r in records]
        y = [r.zelvryn_label  for r in records]
        return (X, y)

    def save_processed(
        self,
        records: List[CorpusRecord],
        output_dir: str = "data/processed",
    ) -> Dict[str, str]:
        """
        Save processed records as JSON for reproducible reloading.

        Parameters
        ----------
        records : list of CorpusRecord
        output_dir : str

        Returns
        -------
        dict
            Paths of written files: {X: path, y: path, meta: path}
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        X, y = self.to_training_arrays(records)
        meta = {
            "n_samples":      len(records),
            "n_benign":       y.count(0),
            "n_attack":       y.count(1),
            "feature_names":  FeatureExtractor.feature_names(),
            "feature_dim":    ML_FEATURE_DIM,
            "sources":        list({r.source_dataset for r in records}),
            "created_at":     time.time(),
        }

        paths = {
            "X":    os.path.join(output_dir, "corpus_features.json"),
            "y":    os.path.join(output_dir, "corpus_labels.json"),
            "meta": os.path.join(output_dir, "corpus_meta.json"),
        }

        with open(paths["X"],    "w") as f: json.dump(X,    f)
        with open(paths["y"],    "w") as f: json.dump(y,    f)
        with open(paths["meta"], "w") as f: json.dump(meta, f, indent=2)

        logger.info("Processed corpus saved to %s", output_dir)
        return paths

    @classmethod
    def load_processed(
        cls,
        input_dir: str = "data/processed",
    ) -> Tuple[List[List[float]], List[int]]:
        """
        Load pre-processed corpus training arrays from disk.

        Parameters
        ----------
        input_dir : str

        Returns
        -------
        tuple of (list of list, list of int)
        """
        x_path = os.path.join(input_dir, "corpus_features.json")
        y_path = os.path.join(input_dir, "corpus_labels.json")

        if not os.path.exists(x_path) or not os.path.exists(y_path):
            raise FileNotFoundError(
                f"Processed corpus files not found in {input_dir}. "
                f"Run scripts/ingest_cicids.py first."
            )

        with open(x_path) as f: X = json.load(f)
        with open(y_path) as f: y = json.load(f)

        logger.info(
            "Processed corpus loaded from %s: %d samples (%d human, %d bot)",
            input_dir, len(X), y.count(0), y.count(1)
        )
        return (X, y)

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _stream_csv(path: str) -> Generator[Dict[str, str], None, None]:
        """
        Stream a CSV file row by row to avoid loading it entirely into memory.

        Handles common encoding issues (BOM, windows line endings).

        Parameters
        ----------
        path : str

        Yields
        ------
        dict of str -> str
        """
        encodings = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]
        for encoding in encodings:
            try:
                with open(path, "r", encoding=encoding, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Strip whitespace from keys and values
                        yield {k.strip(): v.strip() for k, v in row.items() if k}
                return   # success
            except (UnicodeDecodeError, csv.Error):
                continue
        raise IOError(f"Failed to read {path} with any supported encoding.")

    @staticmethod
    def _balance(records: List[CorpusRecord]) -> List[CorpusRecord]:
        """
        Downsample the majority class to balance benign vs attack counts.

        Parameters
        ----------
        records : list of CorpusRecord

        Returns
        -------
        list of CorpusRecord
        """
        import random
        benign  = [r for r in records if r.zelvryn_label == 0]
        attacks = [r for r in records if r.zelvryn_label == 1]

        if not benign or not attacks:
            return records

        min_count = min(len(benign), len(attacks))

        if len(benign) > min_count:
            benign = random.sample(benign, min_count)
        if len(attacks) > min_count:
            attacks = random.sample(attacks, min_count)

        combined = benign + attacks
        random.shuffle(combined)
        return combined

    # ─────────────────────────────────────────────────────────────────────────
    # Dunder methods
    # ─────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"AttackCorpusLoader("
            f"max_records={self.max_records_per_file}, "
            f"balance={self.balance_classes})"
        )
