"""
scripts/train_from_labels.py
=============================
Train the supervised layer from analyst-labelled alerts and register it as a
new model version (state "trained" — promote via canary, never directly).

Input is the JSONL from  GET /alerts/training-export  (one or more files):
    {"features": [20 floats], "label": 0|1, "identity_kind": "...", ...}

  label 1 = analyst confirmed malicious
  label 0 = false positive / expected behaviour

Labelled alerts are hard examples by construction (every one of them raised
an anomaly), which is exactly what the supervised layer is for: cutting false
positives the label-free detectors produce.

Only human-identity samples train the automation classifier, because the
engine applies it to humans only.

Usage:
  python scripts/train_from_labels.py labels.jsonl [more.jsonl ...]
  python scripts/train_from_labels.py labels.jsonl --with-synthetic   # mix in data/processed/*.npy
  python scripts/train_from_labels.py labels.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.classifier import CalibratedGBMClassifier       # noqa: E402
from core.constants import ML_FEATURE_DIM, ML_MIN_TRAIN_SAMPLES  # noqa: E402


def load_jsonl(paths):
    X, y = [], []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("identity_kind", "human") != "human":
                continue
            f = rec.get("features")
            if isinstance(f, list) and len(f) == ML_FEATURE_DIM and rec.get("label") in (0, 1):
                X.append([float(v) for v in f])
                y.append(int(rec["label"]))
    return X, y


def load_synthetic():
    import numpy as np
    base = Path("data/processed")
    h, b = base / "human_features.npy", base / "bot_features.npy"
    if not (h.exists() and b.exists()):
        return [], []
    hx, bx = np.load(h).tolist(), np.load(b).tolist()
    return hx + bx, [0] * len(hx) + [1] * len(bx)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", nargs="+", help="JSONL file(s) from /alerts/training-export")
    ap.add_argument("--with-synthetic", action="store_true", help="also use data/processed synthetic vectors")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--version", default=None)
    ap.add_argument("--dry-run", action="store_true", help="train and report, but don't save or register")
    args = ap.parse_args()

    X, y = load_jsonl(args.labels)
    n_real = len(X)
    if args.with_synthetic:
        sx, sy = load_synthetic()
        X, y = X + sx, y + sy

    n_pos, n_neg = sum(y), len(y) - sum(y)
    print(f"Samples: {len(y)} ({n_real} analyst-labelled) — malicious={n_pos} benign={n_neg}")
    if len(y) < ML_MIN_TRAIN_SAMPLES or n_pos < 5 or n_neg < 5:
        print(
            f"Not enough labelled data yet (need ≥{ML_MIN_TRAIN_SAMPLES} samples and ≥5 of each class). "
            "Keep running shadow mode and labelling alerts."
        )
        return 1

    clf = CalibratedGBMClassifier()
    cv = clf.cross_validate(X, y, k=min(5, max(2, min(n_pos, n_neg))))
    clf.fit(X, y)
    print(
        f"CV  F1={cv['f1']:.3f}  precision={cv['precision']:.3f}  recall={cv['recall']:.3f}  "
        f"missed-attack rate={cv['far']:.3f}  false-alarm rate={cv['frr']:.3f}"
    )
    print("Top features:", ", ".join(f"{n} ({v:.2f})" for n, v in clf.get_top_features(5)))

    if args.dry_run:
        return 0

    from ml.registry import ModelRegistry
    version = args.version or f"gbm_{time.strftime('%Y%m%d_%H%M%S')}"
    path = str(Path(args.models_dir) / f"{version}.joblib")
    clf.save_model(path)
    reg = ModelRegistry(registry_dir=args.models_dir)
    reg.register(
        version, path, n_samples=len(y), n_human=n_neg, n_bot=n_pos,
        cv_f1=cv["f1"], cv_far=cv["far"], cv_frr=cv["frr"],
        notes=f"Calibrated GBM; {n_real} analyst-labelled samples"
              + (" + synthetic" if args.with_synthetic else ""),
    )
    print(f"Registered {version} → {path}. Roll out with POST /admin-gate/models/canary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
