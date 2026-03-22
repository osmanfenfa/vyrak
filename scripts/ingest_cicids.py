"""
scripts/ingest_cicids.py
=========================
Convert CICIDS2018 or CTU-13 CSV files into Zelvryn's 20-dim feature vectors
and save to data/processed/ for ML training.

Usage:
  python scripts/ingest_cicids.py --path data/raw/cicids2018/
  python scripts/ingest_cicids.py --path data/raw/ctu13/ --dataset ctu13
  python scripts/ingest_cicids.py --path data/raw/ --max-rows 50000 --verbose
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Ingest attack corpus datasets")
    p.add_argument("--path",      required=True,  help="Directory containing CSV/binetflow files")
    p.add_argument("--dataset",   default="cicids2018", choices=["cicids2018", "ctu13"])
    p.add_argument("--output-dir",default=str(ROOT / "data" / "processed"))
    p.add_argument("--max-rows",  type=int, default=100_000)
    p.add_argument("--no-balance",action="store_true", help="Skip class balancing")
    p.add_argument("--verbose",   action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not os.path.isdir(args.path):
        log.error("Path not found: %s", args.path)
        sys.exit(1)

    csv_files = [
        os.path.join(args.path, f)
        for f in os.listdir(args.path)
        if f.lower().endswith((".csv", ".binetflow", ".log", ".tsv"))
    ]
    if not csv_files:
        log.error("No CSV files found in %s", args.path)
        log.info("Download CICIDS2018 from: https://www.unb.ca/cic/datasets/ids-2018.html")
        log.info("Download CTU-13 from:     https://www.stratosphereips.org/datasets-ctu13")
        sys.exit(1)

    log.info("Found %d file(s) in %s", len(csv_files), args.path)

    try:
        from ingestion.corpus_loader import AttackCorpusLoader
        loader  = AttackCorpusLoader(max_records_per_file=args.max_rows,
                                     balance_classes=not args.no_balance)
        records = []
        if args.dataset == "cicids2018":
            for f in csv_files:
                log.info("Loading CICIDS2018: %s", f)
                records.extend(loader.load_cicids2018(f))
        else:
            for f in csv_files:
                log.info("Loading CTU-13: %s", f)
                records.extend(loader.load_ctu13(f))

        X, y = loader.to_training_arrays(records)

    except Exception as e:
        log.warning("corpus_loader unavailable (%s). Using fallback CSV parser.", e)
        X, y = _fallback_parse(csv_files, args.dataset, args.max_rows)

    if not X:
        log.error("No records loaded.")
        sys.exit(1)

    log.info("Total: %d samples (%d benign, %d attack)",
             len(X), y.count(0), y.count(1))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    x_path = os.path.join(args.output_dir, "corpus_features.json")
    y_path = os.path.join(args.output_dir, "corpus_labels.json")
    m_path = os.path.join(args.output_dir, "corpus_meta.json")

    with open(x_path, "w") as f: json.dump(X, f)
    with open(y_path, "w") as f: json.dump(y, f)
    with open(m_path, "w") as f:
        import time
        json.dump({"n_samples": len(X), "n_benign": y.count(0),
                   "n_attack": y.count(1), "dataset": args.dataset,
                   "source_dir": args.path, "created_at": time.time()}, f, indent=2)

    log.info("Saved corpus_features.json / corpus_labels.json → %s", args.output_dir)
    log.info("Next step: python scripts/train_initial_model.py --corpus-only")


def _fallback_parse(csv_files, dataset, max_rows):
    """Minimal CSV parser that maps network rows to a plausible 20-dim vector."""
    import csv, math, random

    X, y = [], []
    benign_labels = {"Benign", "BENIGN", "benign", "LEGITIMATE", "Normal", "Background"}

    def safe(v, scale=1.0):
        try:
            return min(1.0, max(0.0, float(v) / scale))
        except (ValueError, TypeError):
            return 0.0

    for path in csv_files:
        if len(X) >= max_rows:
            break
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                with open(path, encoding=encoding, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if len(X) >= max_rows:
                            break
                        row = {k.strip(): v.strip() for k, v in row.items() if k}
                        # Determine label
                        label_key = next((k for k in row if "label" in k.lower()), None)
                        label_str = row.get(label_key, "Benign") if label_key else "Benign"
                        label = 0 if label_str in benign_labels else 1

                        # Build a very rough 20-dim vector from whatever columns exist
                        fp  = safe(row.get("Total Fwd Packets",    row.get("TotPkts", 1)),   1000)
                        bp  = safe(row.get("Total Backward Packets",row.get("SrcBytes", 0)), 1000)
                        tb  = safe(row.get("Total Length of Fwd Packets", row.get("TotBytes", 0)), 1e6)
                        dur = safe(row.get("Flow Duration", row.get("Dur", 1)), 1e7)
                        pkt_r = safe(row.get("Flow Pkts/s", 0), 10000)
                        iat_m = safe(row.get("Fwd IAT Mean", 1e6), 1e7)
                        iat_s = safe(row.get("Fwd IAT Std",  0),   1e6)
                        ack   = safe(row.get("ACK Flag Count", 0), 100)

                        freq = [0.1]*8
                        total = sum(freq)
                        freq = [v/total for v in freq]
                        entropy = math.log2(8) / math.log2(8)  # uniform
                        cv = iat_s / max(iat_m, 0.001)
                        sens = min(1.0, tb * 2)

                        vec = freq + [entropy, iat_m, iat_s, cv, 0.0, dur,
                                      sens, min(1.0, 1-cv), 0.5, pkt_r,
                                      min(1.0, ack/10), cv]
                        X.append(vec)
                        y.append(label)
                break
            except UnicodeDecodeError:
                continue

    return X, y


if __name__ == "__main__":
    main()
