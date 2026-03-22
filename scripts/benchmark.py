"""
scripts/benchmark.py
=====================
Benchmark the Zelvryn evaluation engine for latency and throughput.

Measures:
  - evaluate_action() latency: p50, p95, p99, max (target: p95 < 50ms)
  - Sessions per second throughput
  - Rule-only vs rule+ML latency comparison

Usage:
  python scripts/benchmark.py
  python scripts/benchmark.py --actions 500 --sessions 20 --no-ml
  python scripts/benchmark.py --api http://localhost:8000 --token mytoken
"""

import argparse
import json
import logging
import os
import random
import statistics
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
    p = argparse.ArgumentParser(description="Benchmark Zelvryn evaluation latency")
    p.add_argument("--actions",  type=int, default=200, help="Actions per session")
    p.add_argument("--sessions", type=int, default=10,  help="Number of sessions")
    p.add_argument("--model",    default=str(ROOT / "models" / "rf_v1.joblib"))
    p.add_argument("--no-ml",    action="store_true",   help="Skip ML classifier")
    p.add_argument("--api",      default=None,          help="API base URL for HTTP benchmark")
    p.add_argument("--token",    default="bench_token", help="JWT token for API benchmark")
    p.add_argument("--warmup",   type=int, default=20,  help="Warmup actions before timing")
    p.add_argument("--verbose",  action="store_true")
    return p.parse_args()


# ── in-process benchmark ─────────────────────────────────────────────────────

def bench_in_process(n_sessions: int, n_actions: int, model_path: str,
                     use_ml: bool, n_warmup: int, verbose: bool):
    from core.action import UserAction
    from core.engine import ABSAv2Engine
    from core.profile import BehavioralProfile

    clf = None
    if use_ml and os.path.exists(model_path):
        try:
            from core.classifier import RandomForestClassifier
            clf = RandomForestClassifier.load_model(model_path)
            log.info("Classifier loaded: %s", model_path)
        except Exception as e:
            log.warning("Classifier load failed (%s) — rule-based only.", e)

    rng       = random.Random(42)
    latencies = []
    total_ts  = time.perf_counter()

    for s in range(n_sessions):
        uid     = f"bench_user_{s}"
        profile = BehavioralProfile(user_id=uid)

        # Warm up profile with synthetic history
        ts = time.time() - (n_warmup + n_actions) * 5.0
        for _ in range(n_warmup):
            ts += rng.gauss(5.0, 2.0)
            profile.record_action(
                UserAction(rng.choices(ACTION_TYPES, weights=[3,28,14,2,14,3,3,2], k=1)[0],
                           f"res_{rng.randint(1,30)}", timestamp=ts)
            )

        engine = ABSAv2Engine(profile=profile, classifier=clf,
                              session_id=uid, shadow_mode=True)

        # Timed loop
        for _ in range(n_actions):
            ts += rng.gauss(5.0, 2.0)
            action = UserAction(
                rng.choices(ACTION_TYPES, weights=[3,28,14,2,14,3,3,2], k=1)[0],
                f"res_{rng.randint(1,30)}", timestamp=ts
            )
            t0 = time.perf_counter()
            engine.evaluate_action(action)
            latencies.append((time.perf_counter() - t0) * 1000)

        if verbose:
            log.debug("Session %d/%d complete", s+1, n_sessions)

    total_s = time.perf_counter() - total_ts
    return latencies, total_s


def percentile(data, pct):
    data = sorted(data)
    k    = (len(data) - 1) * pct / 100
    lo   = int(k)
    hi   = lo + 1
    if hi >= len(data):
        return data[-1]
    return data[lo] + (k - lo) * (data[hi] - data[lo])


def print_report(latencies, total_s, label, target_p95=50.0):
    n       = len(latencies)
    p50     = percentile(latencies, 50)
    p95     = percentile(latencies, 95)
    p99     = percentile(latencies, 99)
    p_max   = max(latencies)
    mean    = statistics.mean(latencies)
    tps     = n / total_s
    ok      = "✓ PASS" if p95 < target_p95 else "✗ FAIL"

    print(f"\n{'='*54}")
    print(f"  {label}")
    print(f"{'='*54}")
    print(f"  Samples        : {n:,}")
    print(f"  Duration       : {total_s:.2f}s")
    print(f"  Throughput     : {tps:,.0f} actions/s")
    print(f"  Mean latency   : {mean:.2f} ms")
    print(f"  p50  latency   : {p50:.2f} ms")
    print(f"  p95  latency   : {p95:.2f} ms   (target <{target_p95}ms)  {ok}")
    print(f"  p99  latency   : {p99:.2f} ms")
    print(f"  Max  latency   : {p_max:.2f} ms")
    print(f"{'='*54}\n")

    return p95 < target_p95


# ── HTTP API benchmark ───────────────────────────────────────────────────────

def bench_api(base_url: str, token: str, n_sessions: int, n_actions: int):
    """Benchmark the live API via HTTP. Requires requests library."""
    try:
        import requests
    except ImportError:
        log.error("requests not installed. pip install requests")
        return

    headers  = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    rng      = random.Random(42)
    latencies = []
    errors    = 0

    for s in range(n_sessions):
        # Start session
        try:
            resp = requests.post(f"{base_url}/session/start",
                                 json={"user_id": f"bench_{s}"},
                                 headers=headers, timeout=5)
            if resp.status_code != 201:
                log.warning("Session start failed: %d %s", resp.status_code, resp.text[:100])
                errors += 1
                continue
            session_id = resp.json()["session_id"]
        except Exception as e:
            log.warning("Session start error: %s", e)
            errors += 1
            continue

        # Action loop
        for _ in range(n_actions):
            action = rng.choices(ACTION_TYPES, weights=[3,28,14,2,14,3,3,2], k=1)[0]
            t0 = time.perf_counter()
            try:
                resp = requests.post(
                    f"{base_url}/session/{session_id}/action",
                    json={"action_type": action, "resource_id": f"res_{rng.randint(1,30)}"},
                    headers=headers, timeout=2,
                )
                latencies.append((time.perf_counter() - t0) * 1000)
                if resp.status_code not in (200, 201):
                    errors += 1
            except Exception:
                errors += 1

        # Close session
        try:
            requests.delete(f"{base_url}/session/{session_id}",
                            headers=headers, timeout=2)
        except Exception:
            pass

    log.info("API benchmark complete. Errors: %d/%d", errors, n_sessions * n_actions)
    return latencies


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    all_pass = True

    if args.api:
        log.info("HTTP API benchmark: %s (%d sessions × %d actions)",
                 args.api, args.sessions, args.actions)
        lats = bench_api(args.api, args.token, args.sessions, args.actions)
        if lats:
            ok = print_report(lats,
                              sum(lats)/1000,    # rough total estimate
                              f"HTTP API — {args.api}",
                              target_p95=100.0)  # looser target for HTTP
            all_pass = all_pass and ok
    else:
        # Rule-based only
        log.info("In-process benchmark (rule-based): %d sessions × %d actions",
                 args.sessions, args.actions)
        lats, total_s = bench_in_process(args.sessions, args.actions,
                                          args.model, use_ml=False,
                                          n_warmup=args.warmup, verbose=args.verbose)
        ok = print_report(lats, total_s, "In-process — Rule-based only", target_p95=20.0)
        all_pass = all_pass and ok

        if not args.no_ml and os.path.exists(args.model):
            # Rule + ML
            log.info("In-process benchmark (rule + ML): %d sessions × %d actions",
                     args.sessions, args.actions)
            lats, total_s = bench_in_process(args.sessions, args.actions,
                                              args.model, use_ml=True,
                                              n_warmup=args.warmup, verbose=args.verbose)
            ok = print_report(lats, total_s, "In-process — Rule + ML classifier",
                               target_p95=50.0)
            all_pass = all_pass and ok

    print("Overall result:", "ALL PASS ✓" if all_pass else "SOME TARGETS MISSED ✗")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
