"""
scripts/train_initial_model.py
================================
Train the initial Random Forest classifier and register it as the active model.

Loads training data from (in priority order):
  1. data/processed/human_features.npy + bot_features.npy  (synthetic)
  2. data/processed/corpus_features.json                   (CICIDS/CTU-13 corpus)
  3. data/shadow_exports/*.jsonl                           (shadow session exports)

After training, saves the model to models/ and updates model_registry.json.

Usage:
  python scripts/train_initial_model.py
  python scripts/train_initial_model.py --samples 2000 --trees 20 --folds 5
  python scripts/train_initial_model.py --corpus-only --verbose
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ── data loaders ────────────────────────────────────────────────────────────

def load_numpy(data_dir: str):
    """Load human/bot .npy files from data/processed/."""
    try:
        import numpy as np
        h_path = os.path.join(data_dir, "human_features.npy")
        b_path = os.path.join(data_dir, "bot_features.npy")
        if not (os.path.exists(h_path) and os.path.exists(b_path)):
            return [], []
        X_h = np.load(h_path).tolist()
        X_b = np.load(b_path).tolist()
        y   = [0]*len(X_h) + [1]*len(X_b)
        log.info("Loaded numpy: %d human + %d bot", len(X_h), len(X_b))
        return X_h + X_b, y
    except Exception as e:
        log.warning("Numpy load failed: %s", e)
        return [], []


def load_corpus_json(data_dir: str):
    """Load pre-processed corpus JSON from data/processed/."""
    path = os.path.join(data_dir, "corpus_features.json")
    labels_path = os.path.join(data_dir, "corpus_labels.json")
    if not (os.path.exists(path) and os.path.exists(labels_path)):
        return [], []
    try:
        with open(path) as f:  X = json.load(f)
        with open(labels_path) as f: y = json.load(f)
        log.info("Loaded corpus JSON: %d samples", len(X))
        return X, y
    except Exception as e:
        log.warning("Corpus JSON load failed: %s", e)
        return [], []


def load_shadow_exports(shadow_dir: str, limit: int = 5000):
    """Load labeled feature vectors from shadow export JSONL files."""
    X, y = [], []
    for path in glob.glob(os.path.join(shadow_dir, "*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    fv  = rec.get("feature_vector", [])
                    lbl = rec.get("verified_label") if rec.get("verified_label") is not None \
                          else rec.get("heuristic_label", 0)
                    if isinstance(fv, list) and len(fv) == 20:
                        X.append(fv)
                        y.append(int(lbl))
                        if len(X) >= limit:
                            break
                except Exception:
                    continue
        if len(X) >= limit:
            break
    if X:
        log.info("Loaded shadow exports: %d samples", len(X))
    return X, y


def combine(*datasets):
    """Merge multiple (X, y) pairs into one."""
    X_all, y_all = [], []
    for X, y in datasets:
        X_all.extend(X)
        y_all.extend(y)
    return X_all, y_all


# ── training ────────────────────────────────────────────────────────────────

def train(X, y, n_trees: int, max_depth: int, seed: int):
    """Train a sklearn RandomForestClassifier. Returns (classifier, metrics_dict)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    import random, numpy as np

    # Balance classes
    idx0 = [i for i, v in enumerate(y) if v == 0]
    idx1 = [i for i, v in enumerate(y) if v == 1]
    n    = min(len(idx0), len(idx1))
    rng  = random.Random(seed)
    idx  = rng.sample(idx0, n) + rng.sample(idx1, n)
    X_b  = [X[i] for i in idx]
    y_b  = [y[i] for i in idx]
    log.info("Balanced dataset: %d samples (%d each class)", len(X_b), n)

    clf = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth,
                                  max_features="sqrt", class_weight="balanced",
                                  n_jobs=-1, random_state=seed)
    # 5-fold CV
    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    X_np   = np.array(X_b, dtype=np.float32)
    y_np   = np.array(y_b, dtype=int)
    f1s, fars, frrs = [], [], []
    for tr, te in skf.split(X_np, y_np):
        clf.fit(X_np[tr], y_np[tr])
        pred = clf.predict(X_np[te])
        f1s.append(f1_score(y_np[te], pred, zero_division=0))
        bots   = (y_np[te] == 1).sum()
        humans = (y_np[te] == 0).sum()
        fn  = ((y_np[te] == 1) & (pred == 0)).sum()
        fp  = ((y_np[te] == 0) & (pred == 1)).sum()
        fars.append(fn / bots   if bots   > 0 else 0)
        frrs.append(fp / humans if humans > 0 else 0)
    # Final fit on all balanced data
    clf.fit(X_np, y_np)
    metrics = {"f1":  round(sum(f1s)/5,  4),
               "far": round(sum(fars)/5, 4),
               "frr": round(sum(frrs)/5, 4)}
    log.info("CV metrics — F1=%.3f  FAR=%.3f  FRR=%.3f",
             metrics["f1"], metrics["far"], metrics["frr"])
    return clf, metrics


# ── save & register ─────────────────────────────────────────────────────────

def save_and_register(clf, metrics: dict, n_samples: int, n_human: int, n_bot: int,
                      model_dir: str, version: str):
    import joblib
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(model_dir, f"{version}.joblib")
    joblib.dump({"sklearn_model": clf, "n_trees": clf.n_estimators,
                 "max_depth": clf.max_depth, "n_samples": n_samples,
                 "training_timestamp": time.time(),
                 "feature_importances": list(clf.feature_importances_)}, path)
    log.info("Model saved → %s", path)

    reg_path = os.path.join(model_dir, "model_registry.json")
    try:
        with open(reg_path) as f:
            reg = json.load(f)
    except Exception:
        reg = {"active_version": None, "canary_version": None,
               "canary_traffic_percent": 5, "versions": {}}

    # Retire any current active
    for v, mv in reg.get("versions", {}).items():
        if mv.get("state") == "active":
            mv["state"] = "retired"
            mv["retired_at"] = time.time()

    reg["versions"][version] = {
        "version": version, "model_path": path, "state": "active",
        "n_samples": n_samples, "n_human": n_human, "n_bot": n_bot,
        "cv_f1": metrics["f1"], "cv_far": metrics["far"], "cv_frr": metrics["frr"],
        "trained_at": time.time(), "promoted_at": time.time(),
        "retired_at": None, "notes": "",
    }
    reg["active_version"] = version
    reg["saved_at"]       = time.time()
    with open(reg_path, "w") as f:
        json.dump(reg, f, indent=2)
    log.info("Registry updated  → active=%s", version)
    return path


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train initial Zelvryn RF classifier")
    p.add_argument("--data-dir",    default=str(ROOT / "data" / "processed"))
    p.add_argument("--shadow-dir",  default=str(ROOT / "data" / "shadow_exports"))
    p.add_argument("--model-dir",   default=str(ROOT / "models"))
    p.add_argument("--version",     default="rf_v1")
    p.add_argument("--trees",       type=int,   default=15)
    p.add_argument("--depth",       type=int,   default=6)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--corpus-only", action="store_true",
                   help="Use corpus JSON only, skip .npy files")
    p.add_argument("--shadow-only", action="store_true",
                   help="Use shadow exports only")
    p.add_argument("--verbose",     action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load data
    datasets = []
    if not args.corpus_only and not args.shadow_only:
        datasets.append(load_numpy(args.data_dir))
    if not args.shadow_only:
        datasets.append(load_corpus_json(args.data_dir))
    if not args.corpus_only:
        datasets.append(load_shadow_exports(args.shadow_dir))

    X, y = combine(*datasets)
    if len(X) < 20:
        log.error("Insufficient training data (%d samples). "
                  "Run: python data/synthetic/generate_synthetic.py", len(X))
        sys.exit(1)

    if len(set(y)) < 2:
        log.error("Need both class 0 (human) and class 1 (bot) samples.")
        sys.exit(1)

    log.info("Total training samples: %d (%d human, %d bot)",
             len(X), y.count(0), y.count(1))

    clf, metrics = train(X, y, args.trees, args.depth, args.seed)
    save_and_register(clf, metrics, len(X), y.count(0), y.count(1),
                      args.model_dir, args.version)
    log.info("Done. Active model: %s", args.version)


if __name__ == "__main__":
    main()
