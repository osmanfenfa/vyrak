"""
zelvryn/data/synthetic/generate_synthetic.py
==============================================
Synthetic training data generator for the Zelvryn ABSA ML classifier.

Generates realistic labeled feature vectors for three classes of session
behaviour to seed the Random Forest classifier before real behavioral data
accumulates via shadow mode.

Why synthetic data is necessary
---------------------------------
A new Zelvryn deployment has zero historical sessions. The ML classifier
cannot be trained on an empty dataset. This script generates a plausible
initial training set that:
  1. Gives the classifier a starting decision boundary
  2. Captures the most obvious bot archetypes (timing regularity, burst
     patterns, exfiltration patterns) so the rule-based engine has ML
     support from the first session
  3. Is quickly superseded by real shadow mode data — the classifier is
     retrained every ML_RETRAIN_INTERVAL new labeled samples

Generated populations
----------------------
  Human users (label=0)
    Three human archetypes reflecting different legitimate usage patterns:
      - Document readers:   heavy read/search, slow natural timing, high variance
      - Content creators:   balanced read/write, moderate timing
      - System admins:      varied action mix including config/admin, faster pace

  Bot archetypes (label=1)
    Four bot archetypes targeting different attack scenarios:
      - Regular-timing bot: machine-precise intervals (low CV), read-heavy
      - Burst bot:          rapid-fire actions in bursts (high burst_score)
      - Exfiltration bot:   elevated export/delete proportion, resource targeting
      - AI mimic bot:       human-like timing but unnaturally consistent pattern

Feature vector structure (20 dimensions)
------------------------------------------
  [0-7]  : action type frequencies
  [8]    : normalised Shannon entropy
  [9]    : mean inter-action interval (seconds, normalised to [0,1])
  [10]   : std dev of intervals (normalised)
  [11]   : coefficient of variation (stdev/mean)
  [12]   : minimum interval (seconds, normalised)
  [13]   : maximum interval (seconds, normalised)
  [14]   : sensitive action proportion (delete+export+config+admin)
  [15]   : lag-1 autocorrelation of action indices
  [16]   : resource diversity (unique resources / total actions)
  [17]   : burst score (fraction of intervals < 0.5s)
  [18]   : run-length regularity (mean consecutive same-action streak)
  [19]   : interval quantile ratio (Q1/Q3)

Output files
-------------
  data/processed/human_features.npy   — float32 array shape (N_HUMAN, 20)
  data/processed/bot_features.npy     — float32 array shape (N_BOT,   20)
  data/processed/labels.npy           — int8    array shape (N_TOTAL,)  [0/1]
  data/synthetic/synthetic_features.npy — combined float32 (N_TOTAL, 20)

Run with:
  python data/synthetic/generate_synthetic.py
  python data/synthetic/generate_synthetic.py --samples 2000 --seed 123
  python data/synthetic/generate_synthetic.py --output-dir data/processed/ --verbose

Author : Vyrak
System : Zelvryn — Adaptive Behavioral Signature Authentication
Version: 2.0
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# PATH SETUP — allow running from any working directory
# ─────────────────────────────────────────────────────────────────────────────

_script_dir   = Path(__file__).resolve().parent          # data/synthetic/
_data_dir     = _script_dir.parent                       # data/
_project_root = _data_dir.parent                         # vyrak/

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ─────────────────────────────────────────────────────────────────────────────
# NUMPY IMPORT
# ─────────────────────────────────────────────────────────────────────────────

try:
    import numpy as np
    NP_AVAILABLE = True
except ImportError:
    NP_AVAILABLE = False

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS (mirroring core/constants.py to avoid circular import)
# ─────────────────────────────────────────────────────────────────────────────

N_FEATURES         = 20
ACTION_TYPES       = ["login","read","write","delete","search","export","config","admin"]
SENSITIVE_INDICES  = {3, 5, 6, 7}   # delete, export, config, admin
N_HUMAN_DEFAULT    = 600
N_BOT_DEFAULT      = 600


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE INDEX NAMES (for documentation and logging)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "freq_login",      # 0
    "freq_read",       # 1
    "freq_write",      # 2
    "freq_delete",     # 3
    "freq_search",     # 4
    "freq_export",     # 5
    "freq_config",     # 6
    "freq_admin",      # 7
    "entropy",         # 8
    "mean_interval",   # 9
    "std_interval",    # 10
    "cv_interval",     # 11
    "min_interval",    # 12
    "max_interval",    # 13
    "sensitive_prop",  # 14
    "autocorrelation", # 15
    "resource_div",    # 16
    "burst_score",     # 17
    "run_length",      # 18
    "quantile_ratio",  # 19
]


# ─────────────────────────────────────────────────────────────────────────────
# ARCHETYPE DEFINITIONS
# Each archetype is a dict of parameter ranges. The generator samples
# uniformly within each range and applies Gaussian noise on top.
# ─────────────────────────────────────────────────────────────────────────────

HUMAN_ARCHETYPES = {

    "document_reader": {
        "description": "Enterprise user who primarily reads and searches documents.",
        "weight":       0.40,   # fraction of human samples from this archetype
        # Action type frequency ranges [min, max]
        "freq_login":  [0.03, 0.07],
        "freq_read":   [0.48, 0.62],
        "freq_write":  [0.05, 0.14],
        "freq_delete": [0.01, 0.03],
        "freq_search": [0.16, 0.28],
        "freq_export": [0.02, 0.05],
        "freq_config": [0.01, 0.03],
        "freq_admin":  [0.01, 0.02],
        # Timing parameters (seconds)
        "mean_interval_s": [4.0, 12.0],
        "cv_interval":     [0.50, 1.20],   # high variability — humans pause to think
        "burst_score":     [0.00, 0.04],   # very few bursts
        # Behavioural parameters
        "resource_div":    [0.55, 0.92],   # visits many different documents
        "autocorr":        [-0.15, 0.35],
        "run_length":      [1.0, 2.2],     # rarely does same action more than twice
        "q_ratio":         [0.08, 0.35],   # high interval spread → low Q1/Q3
        "sensitive_prop":  [0.02, 0.08],
    },

    "content_creator": {
        "description": "User who creates, edits, and manages content daily.",
        "weight":       0.35,
        "freq_login":  [0.04, 0.07],
        "freq_read":   [0.28, 0.42],
        "freq_write":  [0.22, 0.35],
        "freq_delete": [0.02, 0.05],
        "freq_search": [0.14, 0.22],
        "freq_export": [0.03, 0.07],
        "freq_config": [0.02, 0.04],
        "freq_admin":  [0.01, 0.03],
        "mean_interval_s": [2.5, 8.0],
        "cv_interval":     [0.45, 1.10],
        "burst_score":     [0.01, 0.06],
        "resource_div":    [0.40, 0.75],
        "autocorr":        [0.05, 0.45],   # moderate repetition (write, write, write)
        "run_length":      [1.2, 3.0],
        "q_ratio":         [0.10, 0.40],
        "sensitive_prop":  [0.04, 0.12],
    },

    "system_admin": {
        "description": "Admin user with elevated access performing system operations.",
        "weight":       0.25,
        "freq_login":  [0.05, 0.09],
        "freq_read":   [0.22, 0.38],
        "freq_write":  [0.14, 0.24],
        "freq_delete": [0.03, 0.08],
        "freq_search": [0.11, 0.20],
        "freq_export": [0.03, 0.08],
        "freq_config": [0.07, 0.15],
        "freq_admin":  [0.04, 0.11],
        "mean_interval_s": [1.8, 6.0],
        "cv_interval":     [0.38, 0.90],
        "burst_score":     [0.02, 0.09],
        "resource_div":    [0.35, 0.70],
        "autocorr":        [0.10, 0.50],
        "run_length":      [1.3, 3.5],
        "q_ratio":         [0.12, 0.42],
        "sensitive_prop":  [0.12, 0.30],
    },
}


BOT_ARCHETYPES = {

    "regular_timing_bot": {
        "description": "Scripted bot with machine-precise regular timing. Classic automation signal.",
        "weight":       0.25,
        "freq_login":  [0.02, 0.06],
        "freq_read":   [0.55, 0.78],
        "freq_write":  [0.08, 0.18],
        "freq_delete": [0.01, 0.03],
        "freq_search": [0.08, 0.18],
        "freq_export": [0.01, 0.04],
        "freq_config": [0.01, 0.02],
        "freq_admin":  [0.00, 0.02],
        # Suspicious timing: very uniform, low CV
        "mean_interval_s": [1.2, 3.5],
        "cv_interval":     [0.02, 0.12],   # ← KEY: very low CV = machine precision
        "burst_score":     [0.00, 0.03],
        "resource_div":    [0.15, 0.40],   # ← KEY: low diversity, targeting same resources
        "autocorr":        [0.60, 0.92],   # ← KEY: high autocorrelation = scripted loop
        "run_length":      [3.5, 7.0],     # ← KEY: long repetitive action runs
        "q_ratio":         [0.75, 0.96],   # ← KEY: high Q1/Q3 = uniform timing
        "sensitive_prop":  [0.01, 0.05],
    },

    "burst_bot": {
        "description": "Bot that fires rapid bursts of actions, rests, then bursts again.",
        "weight":       0.25,
        "freq_login":  [0.03, 0.08],
        "freq_read":   [0.40, 0.62],
        "freq_write":  [0.15, 0.28],
        "freq_delete": [0.02, 0.06],
        "freq_search": [0.10, 0.20],
        "freq_export": [0.02, 0.06],
        "freq_config": [0.01, 0.04],
        "freq_admin":  [0.01, 0.03],
        # Burst timing: mix of very fast bursts and pauses
        "mean_interval_s": [2.0, 6.0],    # mean looks human
        "cv_interval":     [1.8, 4.0],    # ← KEY: very high CV from burst/pause mix
        "burst_score":     [0.35, 0.72],  # ← KEY: high fraction of sub-0.5s intervals
        "resource_div":    [0.25, 0.55],
        "autocorr":        [0.40, 0.80],
        "run_length":      [2.0, 5.0],
        "q_ratio":         [0.01, 0.15],  # low Q1/Q3 from extreme burst/pause contrast
        "sensitive_prop":  [0.03, 0.12],
    },

    "exfiltration_bot": {
        "description": "Bot performing data exfiltration: high export/delete, targeting same resources.",
        "weight":       0.25,
        "freq_login":  [0.02, 0.05],
        "freq_read":   [0.18, 0.35],
        "freq_write":  [0.08, 0.15],
        "freq_delete": [0.08, 0.18],  # ← KEY: elevated delete
        "freq_search": [0.05, 0.12],
        "freq_export": [0.18, 0.35],  # ← KEY: very elevated export
        "freq_config": [0.04, 0.10],
        "freq_admin":  [0.04, 0.10],
        "mean_interval_s": [0.8, 2.5],   # fast
        "cv_interval":     [0.05, 0.18], # fairly uniform
        "burst_score":     [0.10, 0.35],
        "resource_div":    [0.08, 0.22], # ← KEY: very low diversity = targeting same files
        "autocorr":        [0.55, 0.88],
        "run_length":      [4.0, 9.0],   # ← KEY: long export/export/export runs
        "q_ratio":         [0.65, 0.90],
        "sensitive_prop":  [0.28, 0.62], # ← KEY: very high sensitive proportion
    },

    "ai_mimic_bot": {
        "description": "Sophisticated AI mimic: human-like action mix but subtly too consistent.",
        "weight":       0.25,
        "freq_login":  [0.04, 0.07],
        "freq_read":   [0.38, 0.52],    # plausible read proportion
        "freq_write":  [0.14, 0.24],
        "freq_delete": [0.02, 0.05],
        "freq_search": [0.14, 0.22],
        "freq_export": [0.02, 0.06],
        "freq_config": [0.02, 0.05],
        "freq_admin":  [0.01, 0.03],
        # Timing: looks human on average but unnaturally consistent
        "mean_interval_s": [3.5, 8.0],
        "cv_interval":     [0.08, 0.22],  # ← KEY: too low for a human
        "burst_score":     [0.00, 0.03],
        "resource_div":    [0.30, 0.60],
        "autocorr":        [0.45, 0.78],  # ← KEY: moderate-high = subtle scripting
        "run_length":      [2.0, 4.5],
        "q_ratio":         [0.55, 0.82],  # ← KEY: too uniform
        "sensitive_prop":  [0.04, 0.14],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE VECTOR GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _uniform(rng, lo: float, hi: float, noise: float = 0.0) -> float:
    """
    Sample uniformly from [lo, hi] and add optional Gaussian noise.

    Parameters
    ----------
    rng : numpy.random.Generator
    lo, hi : float
        Range bounds.
    noise : float
        Gaussian noise standard deviation (as fraction of range width).

    Returns
    -------
    float
        Clipped to [0.0, 1.0].
    """
    val = rng.uniform(lo, hi)
    if noise > 0:
        val += rng.normal(0, noise * (hi - lo))
    return float(np.clip(val, 0.0, 1.0))


def _sample_freq_vector(rng, archetype: dict) -> List[float]:
    """
    Sample an action type frequency vector from an archetype's parameter ranges.

    Samples raw counts, adds noise, then normalises to sum to 1.0.

    Parameters
    ----------
    rng : numpy.random.Generator
    archetype : dict

    Returns
    -------
    list of float (length 8)
    """
    raw = np.array([
        rng.uniform(*archetype[f"freq_{t}"]) for t in ACTION_TYPES
    ], dtype=np.float64)

    # Add mild Dirichlet noise to break exact uniformity
    noise = rng.dirichlet(np.ones(8) * 2.0)
    raw = raw * 0.85 + noise * 0.15

    # Normalise to sum to 1.0
    total = raw.sum()
    if total > 0:
        raw /= total
    else:
        raw = np.ones(8) / 8.0

    return raw.tolist()


def _compute_entropy(freq_vec: List[float]) -> float:
    """
    Compute normalised Shannon entropy from a frequency vector.

    H_norm = -sum(p * log2(p)) / log2(N)

    Parameters
    ----------
    freq_vec : list of float

    Returns
    -------
    float in [0, 1]
    """
    import math
    entropy = 0.0
    for p in freq_vec:
        if p > 1e-9:
            entropy -= p * math.log2(p)
    max_entropy = math.log2(8)
    return float(np.clip(entropy / max_entropy, 0.0, 1.0))


def _compute_sensitive_prop(freq_vec: List[float]) -> float:
    """
    Compute the sensitive action proportion from a frequency vector.

    Sensitive indices: 3 (delete), 5 (export), 6 (config), 7 (admin).
    """
    return float(sum(freq_vec[i] for i in SENSITIVE_INDICES))


def generate_feature_vector(
    rng,
    archetype: dict,
    label: int,
    noise_level: float = 0.08,
) -> List[float]:
    """
    Generate a single 20-dimensional feature vector from an archetype.

    Parameters
    ----------
    rng : numpy.random.Generator
    archetype : dict
        One of the archetype parameter dicts from HUMAN_ARCHETYPES or BOT_ARCHETYPES.
    label : int
        0 = human, 1 = bot (used for final sanity check only).
    noise_level : float
        Fraction of range width to use as Gaussian noise std dev. Default 0.08.

    Returns
    -------
    list of float (length 20)
    """
    # ── Features 0–7: Action type frequencies ─────────────────────────────
    freq_vec     = _sample_freq_vector(rng, archetype)

    # ── Feature 8: Shannon entropy ─────────────────────────────────────────
    entropy      = _compute_entropy(freq_vec)

    # ── Features 9–13: Interval statistics ────────────────────────────────
    mean_iv_s    = rng.uniform(*archetype["mean_interval_s"])
    cv_iv        = _uniform(rng, *archetype["cv_interval"], noise=noise_level)
    std_iv_s     = mean_iv_s * cv_iv

    # min/max interval derived from mean and std
    min_iv_s     = max(0.01, mean_iv_s - 2.0 * std_iv_s + rng.uniform(-0.1, 0.1))
    max_iv_s     = mean_iv_s + 2.5 * std_iv_s + rng.uniform(0.0, 2.0)

    # Normalise intervals to [0, 1] using domain scale (0–60 seconds)
    _INTERVAL_SCALE = 60.0
    mean_iv      = float(np.clip(mean_iv_s / _INTERVAL_SCALE, 0.0, 1.0))
    std_iv       = float(np.clip(std_iv_s  / _INTERVAL_SCALE, 0.0, 1.0))
    min_iv       = float(np.clip(min_iv_s  / _INTERVAL_SCALE, 0.0, 1.0))
    max_iv       = float(np.clip(max_iv_s  / _INTERVAL_SCALE, 0.0, 1.0))

    # ── Feature 14: Sensitive action proportion ────────────────────────────
    # Let archetype's freq_vec determine this naturally, then blend with range
    freq_sensitive = _compute_sensitive_prop(freq_vec)
    arch_sensitive = _uniform(rng, *archetype["sensitive_prop"], noise=noise_level)
    sensitive_prop = float(np.clip(0.6 * freq_sensitive + 0.4 * arch_sensitive, 0.0, 1.0))

    # ── Feature 15: Lag-1 autocorrelation ─────────────────────────────────
    autocorr     = float(np.clip(
        _uniform(rng, *archetype["autocorr"], noise=noise_level * 1.5),
        -1.0, 1.0
    ))

    # ── Feature 16: Resource diversity ────────────────────────────────────
    resource_div = _uniform(rng, *archetype["resource_div"], noise=noise_level)

    # ── Feature 17: Burst score ───────────────────────────────────────────
    burst_score  = _uniform(rng, *archetype["burst_score"], noise=noise_level)

    # ── Feature 18: Run-length regularity ─────────────────────────────────
    run_length   = float(np.clip(
        rng.uniform(*archetype["run_length"]) + rng.normal(0, 0.3),
        1.0, 15.0
    ))
    # Normalise run length to [0, 1] using scale of 1–15
    run_length_norm = float(np.clip((run_length - 1.0) / 14.0, 0.0, 1.0))

    # ── Feature 19: Interval quantile ratio (Q1/Q3) ───────────────────────
    q_ratio      = _uniform(rng, *archetype["q_ratio"], noise=noise_level)

    # ── Assemble vector ───────────────────────────────────────────────────
    vector = (
        freq_vec                                       # 0-7  (8)
        + [                                            # 8-19 (12)
            entropy,
            mean_iv,
            std_iv,
            float(np.clip(cv_iv, 0.0, 10.0)),         # cv can exceed 1.0
            min_iv,
            max_iv,
            sensitive_prop,
            autocorr,
            resource_div,
            burst_score,
            run_length_norm,
            q_ratio,
        ]
    )

    assert len(vector) == N_FEATURES, (
        f"Vector length error: expected {N_FEATURES}, got {len(vector)}"
    )

    # Sanitise: replace NaN/inf
    vector = [
        float(v) if (not (v != v) and abs(v) != float("inf")) else 0.0
        for v in vector
    ]

    return vector


def generate_population(
    rng,
    archetypes: dict,
    n_samples:  int,
    label:      int,
    noise:      float = 0.08,
) -> "np.ndarray":
    """
    Generate n_samples feature vectors from a population of archetypes.

    Distributes samples across archetypes according to their weight fields.

    Parameters
    ----------
    rng : numpy.random.Generator
    archetypes : dict
        HUMAN_ARCHETYPES or BOT_ARCHETYPES.
    n_samples : int
    label : int
        0 = human, 1 = bot.
    noise : float
        Per-sample Gaussian noise level.

    Returns
    -------
    numpy.ndarray shape (n_samples, 20), dtype float32
    """
    archetype_names  = list(archetypes.keys())
    weights          = np.array([archetypes[n]["weight"] for n in archetype_names])
    weights         /= weights.sum()   # normalise to sum to 1.0

    # Distribute samples proportionally
    counts = np.round(weights * n_samples).astype(int)
    # Adjust rounding errors so total == n_samples
    while counts.sum() < n_samples:
        counts[counts.argmin()] += 1
    while counts.sum() > n_samples:
        counts[counts.argmax()] -= 1

    vectors = []
    for arch_name, count in zip(archetype_names, counts):
        arch = archetypes[arch_name]
        for _ in range(int(count)):
            vec = generate_feature_vector(rng, arch, label=label, noise_level=noise)
            vectors.append(vec)

    # Shuffle to mix archetypes
    vectors_arr = np.array(vectors, dtype=np.float32)
    rng.shuffle(vectors_arr)
    return vectors_arr


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION AND STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def validate_dataset(
    X_human: "np.ndarray",
    X_bot:   "np.ndarray",
    labels:  "np.ndarray",
) -> bool:
    """
    Validate the generated dataset for correctness and plausibility.

    Checks:
      - Correct shapes and dtypes
      - All values in [0, 1] for normalised features
      - No NaN or infinite values
      - Human/bot frequency vectors sum to ~1.0
      - Mean P(human bot_prob) << Mean P(human human_prob) — distributions are separable

    Parameters
    ----------
    X_human : ndarray shape (N, 20)
    X_bot   : ndarray shape (N, 20)
    labels  : ndarray shape (2N,)

    Returns
    -------
    bool
        True if all checks pass.
    """
    passed = True

    # Shape checks
    if X_human.ndim != 2 or X_human.shape[1] != N_FEATURES:
        logger.error("X_human has wrong shape: %s", X_human.shape)
        passed = False
    if X_bot.ndim != 2 or X_bot.shape[1] != N_FEATURES:
        logger.error("X_bot has wrong shape: %s", X_bot.shape)
        passed = False

    # NaN / inf checks
    if np.any(~np.isfinite(X_human)):
        logger.error("X_human contains NaN or infinite values.")
        passed = False
    if np.any(~np.isfinite(X_bot)):
        logger.error("X_bot contains NaN or infinite values.")
        passed = False

    # Frequency vector sum checks (columns 0-7 should sum to ~1.0)
    freq_sums_human = X_human[:, :8].sum(axis=1)
    freq_sums_bot   = X_bot[:, :8].sum(axis=1)
    if not np.allclose(freq_sums_human, 1.0, atol=0.05):
        logger.warning(
            "Human freq vectors: mean sum=%.4f (expected 1.0)",
            freq_sums_human.mean()
        )
    if not np.allclose(freq_sums_bot, 1.0, atol=0.05):
        logger.warning(
            "Bot freq vectors: mean sum=%.4f (expected 1.0)",
            freq_sums_bot.mean()
        )

    # Label distribution
    n_human = (labels == 0).sum()
    n_bot   = (labels == 1).sum()
    if n_human == 0 or n_bot == 0:
        logger.error("Labels are all one class — cannot train a binary classifier.")
        passed = False

    # Feature separability check: CV (col 11) should be lower for bots
    cv_human = X_human[:, 11].mean()
    cv_bot   = X_bot[:, 11].mean()
    if cv_human <= cv_bot:
        logger.warning(
            "CV separability check failed: mean_cv_human=%.3f >= mean_cv_bot=%.3f. "
            "Expected bots to have lower CV (more regular timing).",
            cv_human, cv_bot
        )

    # Burst score check: should be higher for bots
    burst_human = X_human[:, 17].mean()
    burst_bot   = X_bot[:, 17].mean()
    if burst_human >= burst_bot:
        logger.warning(
            "Burst score check: mean_burst_human=%.3f >= mean_burst_bot=%.3f.",
            burst_human, burst_bot
        )

    return passed


def print_statistics(
    X_human: "np.ndarray",
    X_bot:   "np.ndarray",
) -> None:
    """
    Print per-feature statistics comparing human vs bot populations.

    Parameters
    ----------
    X_human : ndarray
    X_bot : ndarray
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("DATASET STATISTICS — Human vs Bot feature comparison")
    logger.info("=" * 70)
    logger.info(
        "%-20s  %8s  %8s  %8s  %8s",
        "Feature", "H_mean", "B_mean", "H_std", "B_std"
    )
    logger.info("-" * 70)

    for i, name in enumerate(FEATURE_NAMES):
        h_mean = X_human[:, i].mean()
        b_mean = X_bot[:,   i].mean()
        h_std  = X_human[:, i].std()
        b_std  = X_bot[:,   i].std()

        # Flag features with good separation (|H_mean - B_mean| > 0.1)
        sep_flag = " *" if abs(h_mean - b_mean) > 0.08 else ""

        logger.info(
            "%-20s  %8.4f  %8.4f  %8.4f  %8.4f%s",
            name, h_mean, b_mean, h_std, b_std, sep_flag
        )

    logger.info("=" * 70)
    logger.info("* = Feature with good human/bot separation (|delta| > 0.08)")
    logger.info("")


# ─────────────────────────────────────────────────────────────────────────────
# FILE OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def save_numpy_files(
    X_human:     "np.ndarray",
    X_bot:       "np.ndarray",
    labels:      "np.ndarray",
    X_combined:  "np.ndarray",
    output_dir:  str,
    synthetic_dir: str,
) -> dict:
    """
    Save all generated arrays to .npy files.

    Parameters
    ----------
    X_human, X_bot, labels, X_combined : ndarrays
    output_dir : str
        data/processed/ — for production-use files
    synthetic_dir : str
        data/synthetic/ — for the combined synthetic file

    Returns
    -------
    dict
        {filename: path} for all saved files.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(synthetic_dir).mkdir(parents=True, exist_ok=True)

    paths = {
        "human_features":    os.path.join(output_dir,    "human_features.npy"),
        "bot_features":      os.path.join(output_dir,    "bot_features.npy"),
        "labels":            os.path.join(output_dir,    "labels.npy"),
        "synthetic_features": os.path.join(synthetic_dir, "synthetic_features.npy"),
    }

    np.save(paths["human_features"],     X_human)
    np.save(paths["bot_features"],       X_bot)
    np.save(paths["labels"],             labels)
    np.save(paths["synthetic_features"], X_combined)

    for name, path in paths.items():
        size_kb = os.path.getsize(path) / 1024
        logger.info(
            "Saved %-25s → %s (%.1f KB)",
            name, path, size_kb
        )

    return paths


def save_metadata(
    paths:          dict,
    n_human:        int,
    n_bot:          int,
    random_seed:    int,
    output_dir:     str,
) -> None:
    """
    Save a metadata JSON file alongside the .npy files describing their contents.

    Parameters
    ----------
    paths : dict
    n_human, n_bot, random_seed : int
    output_dir : str
    """
    meta = {
        "generated_at":   time.time(),
        "generator":      "data/synthetic/generate_synthetic.py",
        "system":         "Zelvryn — Adaptive Behavioral Signature Authentication",
        "random_seed":    random_seed,
        "n_human":        n_human,
        "n_bot":          n_bot,
        "n_total":        n_human + n_bot,
        "n_features":     N_FEATURES,
        "feature_names":  FEATURE_NAMES,
        "action_types":   ACTION_TYPES,
        "files":          paths,
        "human_archetypes": list(HUMAN_ARCHETYPES.keys()),
        "bot_archetypes":   list(BOT_ARCHETYPES.keys()),
        "label_encoding": {"0": "human (legitimate)", "1": "bot (anomalous)"},
        "load_example": (
            "import numpy as np\n"
            "X_human = np.load('data/processed/human_features.npy')\n"
            "X_bot   = np.load('data/processed/bot_features.npy')\n"
            "labels  = np.load('data/processed/labels.npy')\n"
            "# For training: combine X_human and X_bot, use labels as y\n"
            "X = np.vstack([X_human, X_bot])"
        ),
    }
    meta_path = os.path.join(output_dir, "synthetic_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("Metadata saved → %s", meta_path)


# ─────────────────────────────────────────────────────────────────────────────
# NUMPY-FREE FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def generate_without_numpy(
    n_human:     int,
    n_bot:       int,
    seed:        int,
    output_dir:  str,
    synthetic_dir: str,
) -> None:
    """
    Generate and save data using pure Python (no numpy) as JSON fallback.

    Used when numpy is not installed. Saves JSON files instead of .npy.
    The JSON files can be loaded by FeatureStore.import_jsonl() and
    ModelTrainer.train_from_files().

    Parameters
    ----------
    n_human, n_bot, seed : int
    output_dir, synthetic_dir : str
    """
    import random
    import math

    logger.warning(
        "NumPy not installed. Generating JSON fallback files instead of .npy. "
        "Install numpy for binary format: pip install numpy"
    )

    rng_py = random.Random(seed)

    def _u(lo, hi):
        return rng_py.uniform(lo, hi)

    def _gauss_clip(mu, sigma, lo=0.0, hi=1.0):
        return max(lo, min(hi, rng_py.gauss(mu, sigma)))

    all_X   = []
    all_y   = []

    # Simplified archetype sampling — just uses means of ranges
    def _sample_simple(arch, label):
        freq = [_u(*arch[f"freq_{t}"]) for t in ACTION_TYPES]
        total = sum(freq)
        freq  = [v / total for v in freq]

        ent   = 0.0
        for p in freq:
            if p > 1e-9:
                ent -= p * math.log2(p)
        ent /= math.log2(8)

        mean_iv  = _u(*arch["mean_interval_s"]) / 60.0
        cv       = _gauss_clip(_u(*arch["cv_interval"]), 0.05, 0.0, 5.0)
        std_iv   = mean_iv * cv
        min_iv   = max(0.0, mean_iv - 2 * std_iv)
        max_iv   = min(1.0, mean_iv + 2.5 * std_iv)
        sens     = sum(freq[i] for i in SENSITIVE_INDICES)
        autocorr = _gauss_clip(_u(*arch["autocorr"]), 0.1, -1.0, 1.0)
        res_div  = _u(*arch["resource_div"])
        burst    = _u(*arch["burst_score"])
        run      = (_u(*arch["run_length"]) - 1.0) / 14.0
        q_ratio  = _u(*arch["q_ratio"])

        return (freq + [ent, mean_iv, std_iv, cv, min_iv, max_iv,
                        sens, autocorr, res_div, burst, run, q_ratio],
                label)

    archetypes_with_labels = [
        (HUMAN_ARCHETYPES, 0, n_human),
        (BOT_ARCHETYPES,   1, n_bot),
    ]

    for archs, label, n in archetypes_with_labels:
        arch_names = list(archs.keys())
        weights    = [archs[a]["weight"] for a in arch_names]
        w_total    = sum(weights)
        weights    = [w / w_total for w in weights]

        for i in range(n):
            # Weighted random archetype selection
            r = rng_py.random()
            cumsum = 0.0
            selected = arch_names[0]
            for name, w in zip(arch_names, weights):
                cumsum += w
                if r <= cumsum:
                    selected = name
                    break
            vec, lbl = _sample_simple(archs[selected], label)
            all_X.append(vec)
            all_y.append(lbl)

    # Shuffle
    combined = list(zip(all_X, all_y))
    rng_py.shuffle(combined)
    all_X, all_y = zip(*combined)

    # Save as JSON
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(synthetic_dir).mkdir(parents=True, exist_ok=True)

    human_X  = [x for x, y in zip(all_X, all_y) if y == 0]
    bot_X    = [x for x, y in zip(all_X, all_y) if y == 1]

    files = {
        "human_features.json":    (os.path.join(output_dir, "human_features.json"),    human_X),
        "bot_features.json":      (os.path.join(output_dir, "bot_features.json"),      bot_X),
        "labels.json":            (os.path.join(output_dir, "labels.json"),            list(all_y)),
        "synthetic_features.json":(os.path.join(synthetic_dir, "synthetic_features.json"), list(all_X)),
    }

    for fname, (path, data) in files.items():
        with open(path, "w") as f:
            json.dump(data, f)
        logger.info("Saved (JSON) %-30s → %s", fname, path)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def generate(
    n_human:       int  = N_HUMAN_DEFAULT,
    n_bot:         int  = N_BOT_DEFAULT,
    random_seed:   int  = 42,
    noise_level:   float = 0.08,
    output_dir:    str  = None,
    synthetic_dir: str  = None,
    verbose:       bool = False,
) -> None:
    """
    Run the full synthetic data generation pipeline.

    Parameters
    ----------
    n_human : int
        Number of human (label=0) samples to generate.
    n_bot : int
        Number of bot (label=1) samples to generate.
    random_seed : int
        Random seed for reproducibility.
    noise_level : float
        Per-sample Gaussian noise magnitude.
    output_dir : str
        Output directory for human_features.npy, bot_features.npy, labels.npy.
        Defaults to data/processed/ relative to this script.
    synthetic_dir : str
        Output directory for synthetic_features.npy.
        Defaults to data/synthetic/ (same directory as this script).
    verbose : bool
        If True, print per-feature statistics table.
    """
    # Resolve output directories
    if output_dir is None:
        output_dir = str(_data_dir / "processed")
    if synthetic_dir is None:
        synthetic_dir = str(_script_dir)

    logger.info("Zelvryn Synthetic Data Generator v2.0")
    logger.info("  Random seed  : %d", random_seed)
    logger.info("  Human samples: %d (%s)", n_human, ", ".join(
        f"{k}: {int(v['weight']*100)}%%" for k, v in HUMAN_ARCHETYPES.items()
    ))
    logger.info("  Bot samples  : %d (%s)", n_bot, ", ".join(
        f"{k}: {int(v['weight']*100)}%%" for k, v in BOT_ARCHETYPES.items()
    ))
    logger.info("  Noise level  : %.2f", noise_level)
    logger.info("  Output dir   : %s", output_dir)

    if not NP_AVAILABLE:
        generate_without_numpy(n_human, n_bot, random_seed, output_dir, synthetic_dir)
        logger.info("Generation complete (JSON fallback mode).")
        return

    t_start = time.time()
    rng = np.random.default_rng(random_seed)

    # ── Generate human population ─────────────────────────────────────────
    logger.info("Generating %d human feature vectors...", n_human)
    X_human = generate_population(rng, HUMAN_ARCHETYPES, n_human, label=0,
                                   noise=noise_level)

    # ── Generate bot population ───────────────────────────────────────────
    logger.info("Generating %d bot feature vectors...", n_bot)
    X_bot = generate_population(rng, BOT_ARCHETYPES, n_bot, label=1,
                                 noise=noise_level)

    # ── Build combined arrays ─────────────────────────────────────────────
    X_combined = np.vstack([X_human, X_bot]).astype(np.float32)
    labels     = np.concatenate([
        np.zeros(len(X_human), dtype=np.int8),
        np.ones(len(X_bot),    dtype=np.int8),
    ])

    # Shuffle combined
    shuffle_idx = rng.permutation(len(labels))
    X_combined  = X_combined[shuffle_idx]
    labels      = labels[shuffle_idx]

    # ── Validate ──────────────────────────────────────────────────────────
    logger.info("Validating generated dataset...")
    is_valid = validate_dataset(X_human, X_bot, labels)
    if is_valid:
        logger.info("Validation PASSED.")
    else:
        logger.warning("Validation completed with warnings — review output above.")

    # ── Statistics table ──────────────────────────────────────────────────
    if verbose:
        print_statistics(X_human, X_bot)

    # ── Save files ────────────────────────────────────────────────────────
    logger.info("Saving .npy files...")
    paths = save_numpy_files(
        X_human, X_bot, labels, X_combined,
        output_dir, synthetic_dir
    )
    save_metadata(paths, len(X_human), len(X_bot), random_seed, output_dir)

    elapsed = time.time() - t_start
    logger.info(
        "Generation complete: %d human + %d bot = %d total samples in %.2fs",
        len(X_human), len(X_bot), len(X_human) + len(X_bot), elapsed
    )
    logger.info("")
    logger.info("To train the initial model:")
    logger.info("  python scripts/train_initial_model.py")
    logger.info("")
    logger.info("To verify the data:")
    logger.info("  python -c \"import numpy as np; print(np.load('data/processed/labels.npy').shape)\"")


# ─────────────────────────────────────────────────────────────────────────────
# CLI ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic behavioral feature vectors for Zelvryn ML training. "
            "Outputs human_features.npy, bot_features.npy, labels.npy to "
            "data/processed/ and synthetic_features.npy to data/synthetic/."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--samples",
        type    = int,
        default = N_HUMAN_DEFAULT,
        help    = "Number of samples PER CLASS (human and bot each). Total = 2 * samples.",
    )
    parser.add_argument(
        "--human",
        type    = int,
        default = None,
        help    = "Override number of human samples independently.",
    )
    parser.add_argument(
        "--bot",
        type    = int,
        default = None,
        help    = "Override number of bot samples independently.",
    )
    parser.add_argument(
        "--seed",
        type    = int,
        default = 42,
        help    = "Random seed for reproducibility.",
    )
    parser.add_argument(
        "--noise",
        type    = float,
        default = 0.08,
        help    = "Per-sample Gaussian noise level (fraction of range width).",
    )
    parser.add_argument(
        "--output-dir",
        type    = str,
        default = None,
        help    = "Output directory for processed .npy files (default: data/processed/).",
    )
    parser.add_argument(
        "--synthetic-dir",
        type    = str,
        default = None,
        help    = "Output directory for synthetic_features.npy (default: data/synthetic/).",
    )
    parser.add_argument(
        "--verbose",
        action  = "store_true",
        help    = "Print per-feature statistics table after generation.",
    )

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    n_h = args.human if args.human is not None else args.samples
    n_b = args.bot   if args.bot   is not None else args.samples

    generate(
        n_human       = n_h,
        n_bot         = n_b,
        random_seed   = args.seed,
        noise_level   = args.noise,
        output_dir    = args.output_dir,
        synthetic_dir = args.synthetic_dir,
        verbose       = args.verbose,
    )