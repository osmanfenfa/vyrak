"""
scripts/run_shadow_mode.py
===========================
Run the Zelvryn engine in shadow mode against a log file or generate a
synthetic session stream, then export the results as labeled training data.

Usage:
  # Shadow-evaluate a JSONL event log:
  python scripts/run_shadow_mode.py --input logs/events.jsonl

  # Generate a synthetic stream and shadow-evaluate it:
  python scripts/run_shadow_mode.py --synthetic --sessions 50

  # Shadow-evaluate and immediately export + retrain:
  python scripts/run_shadow_mode.py --synthetic --export --retrain

  # Export already-accumulated shadow logs:
  python scripts/run_shadow_mode.py --export-only
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ACTION_TYPES = ["login","read","write","delete","search","export","config","admin"]


def parse_args():
    p = argparse.ArgumentParser(description="Run Zelvryn shadow mode evaluation")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--input",     help="JSONL event log file to shadow-evaluate")
    g.add_argument("--synthetic", action="store_true",
                   help="Generate and evaluate synthetic session stream")
    p.add_argument("--sessions",  type=int, default=30, help="Synthetic sessions to generate")
    p.add_argument("--actions",   type=int, default=40, help="Actions per synthetic session")
    p.add_argument("--bot-ratio", type=float, default=0.3,
                   help="Fraction of synthetic sessions to be bot-like (0–1)")
    p.add_argument("--export",    action="store_true", help="Export shadow logs after evaluation")
    p.add_argument("--export-only",action="store_true", help="Only export; skip evaluation")
    p.add_argument("--retrain",   action="store_true", help="Retrain after export")
    p.add_argument("--output-dir",default=str(ROOT / "data" / "shadow_exports"))
    p.add_argument("--model-dir", default=str(ROOT / "models"))
    p.add_argument("--format",    default="jsonl", choices=["jsonl","csv"])
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--verbose",   action="store_true")
    return p.parse_args()


# ── synthetic stream generator ───────────────────────────────────────────────

def generate_events(n_sessions: int, actions_per: int, bot_ratio: float, seed: int):
    """Yield action event dicts simulating a mixed human/bot stream."""
    rng = random.Random(seed)
    for i in range(n_sessions):
        is_bot  = rng.random() < bot_ratio
        uid     = f"user_{i:04d}"
        sid     = f"sess_{i:04d}_{int(time.time())}"
        ts      = time.time() - rng.uniform(0, 3600)

        for _ in range(actions_per):
            if is_bot:
                action = rng.choices(
                    ACTION_TYPES,
                    weights=[2,20,8,4,8,12,4,4], k=1
                )[0]
                ts += rng.uniform(0.05, 0.4)        # machine-fast
            else:
                action = rng.choices(
                    ACTION_TYPES,
                    weights=[3,28,14,2,14,3,3,2], k=1
                )[0]
                ts += rng.gauss(5.0, 3.0)           # human pace

            yield {
                "user_id":    uid,
                "session_id": sid,
                "action_type": action,
                "resource_id": f"res_{rng.randint(1, 50)}",
                "timestamp":  round(max(ts, 0), 3),
                "_is_bot":    is_bot,   # ground truth (not part of real API payload)
            }


# ── shadow evaluation ────────────────────────────────────────────────────────

def shadow_evaluate(events, output_dir: str, model_dir: str, fmt: str):
    """Run all events through the StreamProcessor in shadow mode."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        from ingestion.stream_processor import StreamProcessor
        processor = StreamProcessor(shadow_mode=True)

        processed = errors = blocks = challenges = 0
        for ev in events:
            result = processor.process_event({k: v for k, v in ev.items()
                                               if not k.startswith("_")})
            if result.is_success():
                processed += 1
                if result.decision == "BLOCK":     blocks += 1
                if result.decision == "CHALLENGE": challenges += 1
            else:
                errors += 1

        stats = processor.get_statistics()
        log.info("Shadow evaluation: processed=%d errors=%d BLOCK=%d CHALLENGE=%d",
                 processed, errors, blocks, challenges)
        log.info("Active sessions in cache: %d", stats["active_sessions"])
        return processor

    except Exception as e:
        log.warning("StreamProcessor unavailable (%s). Running direct evaluation.", e)
        return _direct_evaluate(events, output_dir)


def _direct_evaluate(events, output_dir: str):
    """Fallback: evaluate directly using core engine without infrastructure."""
    from core.action import UserAction
    from core.engine import ABSAv2Engine
    from core.profile import BehavioralProfile

    engines = {}
    log_entries = []
    processed = errors = blocks = challenges = 0

    for ev in events:
        sid = ev.get("session_id", "default")
        uid = ev.get("user_id", "unknown")

        if sid not in engines:
            engines[sid] = ABSAv2Engine(
                profile     = BehavioralProfile(user_id=uid),
                session_id  = sid,
                shadow_mode = True,
            )
        engine = engines[sid]

        try:
            action = UserAction(
                action_type = ev.get("action_type", "read"),
                resource_id = ev.get("resource_id", "unknown"),
                session_id  = sid,
                timestamp   = ev.get("timestamp", time.time()),
            )
            result = engine.evaluate_action(action)
            processed += 1
            if result.decision == "BLOCK":     blocks += 1
            if result.decision == "CHALLENGE": challenges += 1
            log_entries.append({
                "user_id": uid, "session_id": sid,
                "action_type": ev.get("action_type"),
                "timestamp": ev.get("timestamp"),
                "real_decision": result.decision,
                "trust_score": result.trust_score,
                "bot_prob": result.bot_prob,
                "anomalies": result.anomalies,
                "heuristic_label": 1 if result.anomalies else 0,
                "feature_vector": [],  # no extractor in fallback
            })
        except Exception as e:
            errors += 1
            log.debug("Event error: %s", e)

    log.info("Direct evaluation: %d processed %d errors BLOCK=%d CHALLENGE=%d",
             processed, errors, blocks, challenges)

    # Write shadow log
    if log_entries:
        out_path = os.path.join(
            output_dir,
            f"shadow_direct_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        with open(out_path, "w") as f:
            for entry in log_entries:
                f.write(json.dumps(entry) + "\n")
        log.info("Shadow log written → %s (%d entries)", out_path, len(log_entries))

    return None


# ── export ───────────────────────────────────────────────────────────────────

def export_shadow_logs(output_dir: str, fmt: str):
    """Aggregate all shadow JSONL files and return (X, y)."""
    import glob
    X, y = [], []
    files = glob.glob(os.path.join(output_dir, f"*.{fmt}"))
    if not files:
        log.warning("No shadow export files found in %s", output_dir)
        return X, y

    for path in files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    fv  = rec.get("feature_vector", [])
                    lbl = rec.get("verified_label")
                    if lbl is None:
                        lbl = rec.get("heuristic_label", 0)
                    if isinstance(fv, list) and len(fv) == 20:
                        X.append(fv)
                        y.append(int(lbl))
                except Exception:
                    continue

    log.info("Exported %d labeled samples from %d files", len(X), len(files))
    return X, y


# ── retrain ──────────────────────────────────────────────────────────────────

def retrain(X, y, model_dir: str):
    """Quick retrain using exported shadow data."""
    if len(X) < 20 or len(set(y)) < 2:
        log.warning("Insufficient data for retraining (%d samples, classes=%s).",
                    len(X), set(y))
        return

    import subprocess, sys
    cmd = [sys.executable, str(ROOT / "scripts" / "train_initial_model.py"),
           "--shadow-only", "--version", f"rf_shadow_{int(time.time())}",
           "--model-dir", model_dir]
    log.info("Launching: %s", " ".join(cmd))
    subprocess.run(cmd, check=False)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.export_only:
        X, y = export_shadow_logs(args.output_dir, args.format)
        if args.retrain:
            retrain(X, y, args.model_dir)
        return

    # Build event stream
    if args.input:
        if not os.path.exists(args.input):
            log.error("Input file not found: %s", args.input)
            sys.exit(1)
        log.info("Loading events from %s", args.input)
        def events():
            with open(args.input, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try: yield json.loads(line)
                        except: pass
        event_iter = events()
    elif args.synthetic:
        log.info("Generating synthetic stream: %d sessions × %d actions "
                 "(bot_ratio=%.2f)", args.sessions, args.actions, args.bot_ratio)
        event_iter = generate_events(args.sessions, args.actions,
                                     args.bot_ratio, args.seed)
    else:
        log.error("Specify --input <file> or --synthetic")
        sys.exit(1)

    # Shadow evaluate
    events_list = list(event_iter)
    log.info("Processing %d events in shadow mode...", len(events_list))
    shadow_evaluate(events_list, args.output_dir, args.model_dir, args.format)

    # Export + retrain if requested
    if args.export or args.retrain:
        X, y = export_shadow_logs(args.output_dir, "jsonl")
        if args.retrain:
            retrain(X, y, args.model_dir)

    log.info("Shadow mode run complete.")


if __name__ == "__main__":
    main()
