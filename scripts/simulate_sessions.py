"""
scripts/simulate_sessions.py
==============================
Simulate realistic session streams to validate the Zelvryn engine in isolation
(no API, no Redis, no database). Runs three scenarios and prints a pass/fail report.

Scenarios:
  1. Normal user    — 60 actions matching trained profile → all ALLOW expected
  2. AI mimic bot   — machine-precise timing, export-heavy → CHALLENGE/BLOCK expected
  3. Exfil attack   — burst of delete+export → BLOCK expected

Usage:
  python scripts/simulate_sessions.py
  python scripts/simulate_sessions.py --model models/rf_v1.joblib --sessions 5
  python scripts/simulate_sessions.py --no-ml --verbose
"""

import argparse
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


def parse_args():
    p = argparse.ArgumentParser(description="Simulate Zelvryn sessions end-to-end")
    p.add_argument("--model",    default=str(ROOT / "models" / "rf_v1.joblib"))
    p.add_argument("--sessions", type=int, default=3, help="Repetitions per scenario")
    p.add_argument("--actions",  type=int, default=60, help="Actions per session")
    p.add_argument("--no-ml",    action="store_true", help="Skip ML classifier")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--verbose",  action="store_true")
    return p.parse_args()


# ── action generators ────────────────────────────────────────────────────────

ACTION_TYPES = ["login","read","write","delete","search","export","config","admin"]


def _actions(rng, n, weights, interval_fn, resource_pool=20):
    """Yield n UserAction objects."""
    from core.action import UserAction
    ts = time.time() - n * 6.0
    for _ in range(n):
        ts += max(0.01, interval_fn(rng))
        act = rng.choices(ACTION_TYPES, weights=weights, k=1)[0]
        rid = f"res_{rng.randint(1, resource_pool)}"
        yield UserAction(act, rid, timestamp=ts)


def normal_user(rng, n):
    return _actions(rng, n,
                    weights=[3,28,14,2,14,3,3,2],
                    interval_fn=lambda r: r.gauss(5.0, 2.5),
                    resource_pool=30)


def ai_mimic_bot(rng, n):
    return _actions(rng, n,
                    weights=[2,26,10,3,10,10,4,4],
                    interval_fn=lambda r: 1.8 + r.gauss(0, 0.05),   # too uniform
                    resource_pool=8)                                   # low diversity


def exfil_bot(rng, n):
    return _actions(rng, n,
                    weights=[1,10,4,15,4,25,8,8],                     # heavy export/delete
                    interval_fn=lambda r: r.uniform(0.08, 0.35),      # fast burst
                    resource_pool=4)                                    # targeting few resources


# ── session runner ───────────────────────────────────────────────────────────

def run_session(actions, classifier, label: str, verbose: bool):
    from core.engine import ABSAv2Engine
    from core.profile import BehavioralProfile

    uid     = f"sim_{label}_{int(time.time())}"
    profile = BehavioralProfile(user_id=uid)

    # Warm up profile with 50 normal actions before testing
    _warm_up(profile)

    engine  = ABSAv2Engine(profile=profile, classifier=classifier,
                           session_id=uid, shadow_mode=False)
    decisions = []
    for action in actions:
        result = engine.evaluate_action(action)
        decisions.append(result.decision)
        if verbose:
            log.debug("  [%s] %s → %s  trust=%.1f  P(bot)=%.3f  anomalies=%s",
                      label, action.action_type, result.decision,
                      result.trust_score, result.bot_prob, result.anomalies)

    summary = engine.get_session_summary()
    return decisions, summary


def _warm_up(profile, n=50, seed=0):
    """Seed the profile with n normal human actions."""
    from core.action import UserAction
    rng = random.Random(seed)
    ts  = time.time() - n * 6.0
    for _ in range(n):
        ts += rng.gauss(5.0, 2.0)
        act = rng.choices(ACTION_TYPES, weights=[3,28,14,2,14,3,3,2], k=1)[0]
        profile.record_action(UserAction(act, f"res_{rng.randint(1,30)}", timestamp=ts))


def load_classifier(model_path: str):
    """Load the RF classifier. Returns None gracefully if unavailable."""
    if not os.path.exists(model_path):
        log.warning("Model not found at %s — running rule-based only.", model_path)
        return None
    try:
        from core.classifier import RandomForestClassifier
        return RandomForestClassifier.load_model(model_path)
    except Exception as e:
        log.warning("Could not load classifier (%s) — rule-based only.", e)
        return None


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    clf = None if args.no_ml else load_classifier(args.model)
    if clf:
        log.info("Classifier loaded: %s", args.model)
    else:
        log.info("Running rule-based detection only (no classifier).")

    SCENARIOS = [
        ("normal_user",  normal_user,  "ALLOW"),
        ("ai_mimic_bot", ai_mimic_bot, "CHALLENGE_OR_BLOCK"),
        ("exfil_attack", exfil_bot,    "CHALLENGE_OR_BLOCK"),
    ]

    results = []
    rng     = random.Random(args.seed)

    for name, generator, expected in SCENARIOS:
        log.info("─── Scenario: %s (expected: %s) ───", name, expected)
        session_results = []
        for rep in range(args.sessions):
            acts = list(generator(rng, args.actions))
            decisions, summary = run_session(acts, clf, f"{name}_{rep}", args.verbose)
            allow_pct     = decisions.count("ALLOW")    / len(decisions) * 100
            challenge_pct = decisions.count("CHALLENGE")/ len(decisions) * 100
            block_pct     = decisions.count("BLOCK")    / len(decisions) * 100
            flagged       = challenge_pct + block_pct

            log.info("  Session %d/%d: ALLOW=%.0f%%  CHALLENGE=%.0f%%  BLOCK=%.0f%%  "
                     "trust=%.1f  risk=%s",
                     rep+1, args.sessions, allow_pct, challenge_pct, block_pct,
                     summary["final_trust"], summary["risk_rating"])
            session_results.append(flagged)

        avg_flagged = sum(session_results) / len(session_results)
        if expected == "ALLOW":
            passed = avg_flagged < 15.0   # allow up to 15% false positive
        else:
            passed = avg_flagged > 20.0   # expect at least 20% flagged for bots

        status = "PASS ✓" if passed else "FAIL ✗"
        log.info("  → %s  (avg flagged=%.1f%%)", status, avg_flagged)
        results.append((name, passed, avg_flagged))

    # Summary report
    print("\n" + "="*60)
    print("  ZELVRYN SIMULATION REPORT")
    print("="*60)
    for name, passed, flagged in results:
        mark  = "PASS" if passed else "FAIL"
        print(f"  {mark:4s}  {name:<20s}  flagged={flagged:.1f}%")
    print("="*60)
    all_pass = all(p for _, p, _ in results)
    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    print("="*60 + "\n")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
