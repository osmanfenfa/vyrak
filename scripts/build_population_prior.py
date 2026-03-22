"""
scripts/build_population_prior.py
===================================
Rebuild the population prior (data/processed/population_prior.json) from all
mature user profiles stored in Redis.

Run weekly via cron or `make rebuild-prior`.

Usage:
  python scripts/build_population_prior.py
  python scripts/build_population_prior.py --redis-url redis://localhost:6379/0
  python scripts/build_population_prior.py --dry-run --verbose
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Rebuild Zelvryn population prior from Redis profiles")
    p.add_argument("--redis-url",  default=os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    p.add_argument("--output",     default=str(ROOT / "data" / "processed" / "population_prior.json"))
    p.add_argument("--context",    default="default", help="Deployment context label")
    p.add_argument("--min-actions",type=int, default=20, help="Min actions for a mature profile")
    p.add_argument("--dry-run",    action="store_true", help="Compute but do not save")
    p.add_argument("--verbose",    action="store_true")
    return p.parse_args()


def load_profiles_from_redis(redis_url: str, min_actions: int):
    """Load all mature BehavioralProfile objects from Redis."""
    from profiles.redis_store import RedisProfileStore
    from core.profile import BehavioralProfile
    import redis

    pool   = redis.ConnectionPool.from_url(redis_url, decode_responses=True,
                                            socket_connect_timeout=5)
    client = redis.Redis(connection_pool=pool)
    client.ping()

    store    = RedisProfileStore(redis_client=client)
    user_ids = store.list_active_users(max_results=10_000)
    log.info("Found %d users in Redis", len(user_ids))

    profiles = []
    for uid in user_ids:
        try:
            p = store.get(uid)
            if p and p.vector_count >= min_actions:
                profiles.append(p)
        except Exception as e:
            log.debug("Skipping user %s: %s", uid, e)

    return profiles


def build_from_profiles(profiles, context: str):
    """Delegate to PopulationPriorProfile.build_from_profiles()."""
    from coldstart.population_prior import PopulationPriorProfile
    return PopulationPriorProfile.build_from_profiles(profiles, deployment_context=context)


def build_default_prior(context: str):
    """Return a sensible default prior when no profiles are available."""
    return {
        "mean_action_vector": [0.052, 0.481, 0.197, 0.021, 0.168, 0.028, 0.033, 0.020],
        "std_action_vector":  [0.031, 0.092, 0.058, 0.018, 0.061, 0.024, 0.028, 0.017],
        "mean_interval_s":    4.8,
        "std_interval_s":     3.2,
        "median_adaptive_threshold": 0.74,
        "n_profiles":         0,
        "built_at":           time.time(),
        "deployment_context": context,
        "action_types": ["login","read","write","delete","search","export","config","admin"],
    }


def main():
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    profiles = []

    # Try loading from Redis
    try:
        log.info("Connecting to Redis: %s", args.redis_url.split("@")[-1])
        profiles = load_profiles_from_redis(args.redis_url, args.min_actions)
        log.info("Loaded %d mature profiles (min_actions=%d)", len(profiles), args.min_actions)
    except Exception as e:
        log.warning("Redis unavailable (%s). Using synthetic profiles for prior.", e)

    # If no real profiles, generate synthetic ones as a bootstrap
    if len(profiles) < 10:
        log.info("Fewer than 10 mature profiles — generating synthetic bootstrap prior.")
        prior_data = _build_synthetic_prior(args)
    else:
        try:
            prior = build_from_profiles(profiles, args.context)
            prior_data = prior.to_dict() if hasattr(prior, "to_dict") else _prior_to_dict(prior, profiles)
        except Exception as e:
            log.warning("build_from_profiles failed (%s). Falling back to default.", e)
            prior_data = build_default_prior(args.context)

    # Report
    log.info("Prior summary:")
    log.info("  n_profiles    = %d",  prior_data.get("n_profiles", 0))
    log.info("  mean_interval = %.2fs", prior_data.get("mean_interval_s", 0))
    log.info("  threshold     = %.3f", prior_data.get("median_adaptive_threshold", 0))
    log.info("  vector        = %s",
             [round(v, 3) for v in prior_data.get("mean_action_vector", [])])

    if args.dry_run:
        log.info("Dry run — not saving.")
        return

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(prior_data, f, indent=2)
    log.info("Saved → %s", args.output)


def _build_synthetic_prior(args):
    """Build a prior from synthetic profile statistics."""
    import statistics, random

    rng = random.Random(42)
    action_types = ["login","read","write","delete","search","export","config","admin"]
    base = [0.052, 0.481, 0.197, 0.021, 0.168, 0.028, 0.033, 0.020]

    # Simulate 50 synthetic profiles with mild variation
    all_vecs, all_intervals, all_thresholds = [], [], []
    for _ in range(50):
        noise   = [rng.gauss(0, 0.015) for _ in range(8)]
        raw     = [max(0, b + n) for b, n in zip(base, noise)]
        total   = sum(raw)
        vec     = [v / total for v in raw]
        mean_iv = rng.gauss(4.8, 1.2)
        cv      = rng.gauss(0.75, 0.2)
        thresh  = max(0.65, min(0.85, 0.75 - cv * 0.1))
        all_vecs.append(vec)
        all_intervals.append(max(0.5, mean_iv))
        all_thresholds.append(thresh)

    mean_vec = [statistics.mean(v[i] for v in all_vecs) for i in range(8)]
    total    = sum(mean_vec)
    mean_vec = [v / total for v in mean_vec]
    std_vec  = [statistics.stdev(v[i] for v in all_vecs) for i in range(8)]

    return {
        "mean_action_vector":        [round(v, 6) for v in mean_vec],
        "std_action_vector":         [round(v, 6) for v in std_vec],
        "mean_interval_s":           round(statistics.mean(all_intervals), 4),
        "std_interval_s":            round(statistics.stdev(all_intervals), 4),
        "median_adaptive_threshold": round(statistics.median(all_thresholds), 4),
        "n_profiles":                0,
        "built_at":                  time.time(),
        "deployment_context":        args.context,
        "action_types":              action_types,
        "_note": "Built from synthetic bootstrap (no real profiles available).",
    }


def _prior_to_dict(prior, profiles):
    """Extract dict from a PopulationPriorProfile that may lack to_dict()."""
    import statistics
    thresholds = [getattr(p, "adaptive_threshold", 0.75) for p in profiles]
    return {
        "mean_action_vector":        getattr(prior, "mean_vector", [0.125]*8),
        "std_action_vector":         [0.05]*8,
        "mean_interval_s":           getattr(prior, "mean_interval", 4.8),
        "std_interval_s":            getattr(prior, "std_interval", 3.2),
        "median_adaptive_threshold": statistics.median(thresholds) if thresholds else 0.75,
        "n_profiles":                getattr(prior, "source_profile_count", len(profiles)),
        "built_at":                  time.time(),
        "deployment_context":        getattr(prior, "deployment_context", "default"),
        "action_types": ["login","read","write","delete","search","export","config","admin"],
    }


if __name__ == "__main__":
    main()
