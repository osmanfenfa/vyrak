"""
AdversarialTrainer — GAN-based adversarial hardening for Zelvryn's Random Forest.
"""

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from core.classifier import RandomForestClassifier
from core.constants import ML_FEATURE_DIM, ML_BOT_CONFIDENCE_THRESH
from core.features import FeatureExtractor

logger = logging.getLogger(__name__)

# Check PyTorch availability once at import time
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_AVAILABLE = True
    logger.debug("PyTorch available — GAN generator enabled.")
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning(
        "PyTorch not installed. Adversarial training will use the "
        "gradient-free perturbation fallback. "
        "For full GAN support: pip install torch"
    )


@dataclass
class AdversarialConfig:
    """
    Configuration for the adversarial training loop.
    """
    n_rounds:             int   = 5
    samples_per_round:    int   = 200
    generator_epochs:     int   = 50
    generator_lr:         float = 1e-3
    noise_dim:            int   = 32
    perturbation_scale:   float = 0.15
    evasion_target:       float = 0.40
    retrain_full_dataset: bool  = True
    random_seed:          int   = 42


@dataclass
class AdversarialResult:
    """
    Summary of one adversarial training run.
    """
    rounds_completed:           int         = 0
    total_adversarial_samples:  int         = 0
    evasion_rate_before:        float       = 0.0
    evasion_rate_after:         float       = 0.0
    hardening_gain:             float       = 0.0
    per_round_stats:            List[Dict]  = field(default_factory=list)
    duration_s:                 float       = 0.0
    backend:                    str         = "unknown"

    def summary(self) -> str:
        return (
            f"AdversarialResult("
            f"rounds={self.rounds_completed}, "
            f"samples={self.total_adversarial_samples}, "
            f"evasion_before={self.evasion_rate_before:.3f}, "
            f"evasion_after={self.evasion_rate_after:.3f}, "
            f"gain={self.hardening_gain:.3f}, "
            f"backend={self.backend})"
        )


class AdversarialTrainer:
    """
    Hardens a Random Forest classifier against adversarial evasion using a
    GAN-based adversarial training loop.
    """

    def __init__(self, config: Optional[AdversarialConfig] = None) -> None:
        self.config = config or AdversarialConfig()
        random.seed(self.config.random_seed)

    # PRIMARY INTERFACE
    def train_adversarial_loop(
        self,
        classifier: RandomForestClassifier,
        X_human: List[List[float]],
        y_human: Optional[List[int]] = None,
        X_bot: Optional[List[List[float]]] = None,
        y_bot: Optional[List[int]] = None,
    ) -> Tuple[RandomForestClassifier, AdversarialResult]:
        """
        Run the full adversarial training loop.
        """
        t_start  = time.time()
        cfg      = self.config
        backend  = "gan_pytorch" if _TORCH_AVAILABLE else "perturbation_fallback"

        logger.info(
            "Adversarial training started: rounds=%d samples/round=%d backend=%s",
            cfg.n_rounds, cfg.samples_per_round, backend
        )

        y_human = y_human or [0] * len(X_human)
        X_bot   = X_bot   or []
        y_bot   = y_bot   or [1] * len(X_bot)

        # Seed for adversarial generation — use known bot samples if available,
        # otherwise perturb human samples (they are the evasion target)
        seed_X = X_bot if X_bot else X_human

        # Measure initial evasion rate
        initial_evasion = self._measure_evasion_rate(
            classifier, self._generate_adversarial_batch(
                classifier, seed_X, cfg.samples_per_round, round_num=0
            )
        )

        all_adversarial_X: List[List[float]] = []
        all_adversarial_y: List[int]         = []
        per_round_stats:   List[Dict]         = []

        for round_num in range(1, cfg.n_rounds + 1):
            logger.info("Adversarial round %d/%d", round_num, cfg.n_rounds)

            # Step 1: Generate adversarial samples
            adv_X = self._generate_adversarial_batch(
                classifier, seed_X, cfg.samples_per_round, round_num
            )
            adv_y = [1] * len(adv_X)   # label all as bot

            # Measure evasion before this round's retraining
            evasion_before = self._measure_evasion_rate(classifier, adv_X)

            # Step 2: Accumulate adversarial data
            all_adversarial_X.extend(adv_X)
            all_adversarial_y.extend(adv_y)

            # Step 3: Retrain on combined dataset 
            if cfg.retrain_full_dataset:
                X_train = X_human + list(X_bot) + all_adversarial_X
                y_train = y_human + list(y_bot) + all_adversarial_y
            else:
                X_train = all_adversarial_X
                y_train = all_adversarial_y

            # Balance classes
            X_train, y_train = self._balance(X_train, y_train)

            try:
                classifier.fit(X_train, y_train)
            except Exception as e:
                logger.warning("Retraining failed on round %d: %s", round_num, e)
                continue

            # Step 4: Measure evasion after retraining
            evasion_after = self._measure_evasion_rate(classifier, adv_X)
            improvement   = evasion_before - evasion_after

            round_stat = {
                "round":           round_num,
                "adv_samples":     len(adv_X),
                "evasion_before":  round(evasion_before, 4),
                "evasion_after":   round(evasion_after, 4),
                "improvement":     round(improvement, 4),
            }
            per_round_stats.append(round_stat)

            logger.info(
                "Round %d: evasion %.3f→%.3f (improvement %.3f)",
                round_num, evasion_before, evasion_after, improvement
            )

        # Final evasion measurement
        final_adv = self._generate_adversarial_batch(
            classifier, seed_X, cfg.samples_per_round, round_num=cfg.n_rounds
        )
        final_evasion = self._measure_evasion_rate(classifier, final_adv)

        result = AdversarialResult(
            rounds_completed          = cfg.n_rounds,
            total_adversarial_samples = len(all_adversarial_X),
            evasion_rate_before       = round(initial_evasion, 4),
            evasion_rate_after        = round(final_evasion, 4),
            hardening_gain            = round(initial_evasion - final_evasion, 4),
            per_round_stats           = per_round_stats,
            duration_s                = round(time.time() - t_start, 2),
            backend                   = backend,
        )

        logger.info(result.summary())
        return classifier, result

    def generate_evasion_samples(
        self,
        classifier: RandomForestClassifier,
        n: int = 100,
        seed_X: Optional[List[List[float]]] = None,
    ) -> List[List[float]]:
        """
        Generate n adversarial feature vectors that attempt to evade the classifier.
        """
        seed = seed_X or [[random.random() for _ in range(ML_FEATURE_DIM)]
                          for _ in range(max(10, n))]
        return self._generate_adversarial_batch(classifier, seed, n, round_num=1)

    # ADVERSARIAL GENERATION BACKENDS
    def _generate_adversarial_batch(
        self,
        classifier: RandomForestClassifier,
        seed_X: List[List[float]],
        n: int,
        round_num: int,
    ) -> List[List[float]]:
        """
        Generate adversarial samples using the best available backend.
        """
        if _TORCH_AVAILABLE and classifier.is_trained:
            return self._gan_generate(classifier, n, round_num)
        else:
            return self._perturbation_generate(classifier, seed_X, n, round_num)

    def _gan_generate(
        self,
        classifier: RandomForestClassifier,
        n: int,
        round_num: int,
    ) -> List[List[float]]:
        """
        Train a generator network and sample adversarial vectors from it.
        """
        cfg    = self.config
        device = torch.device("cpu")

        # Define generator network
        generator = nn.Sequential(
            nn.Linear(cfg.noise_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, ML_FEATURE_DIM),
            nn.Sigmoid(),   # outputs in [0, 1] — matches normalised feature range
        ).to(device)

        optimiser = optim.Adam(generator.parameters(), lr=cfg.generator_lr)

        # Training loop
        generator.train()
        for epoch in range(cfg.generator_epochs):
            optimiser.zero_grad()

            # Sample noise
            noise = torch.randn(min(n, 64), cfg.noise_dim, device=device)
            fake_features = generator(noise)   # shape: (batch, 20)

            # Convert to numpy for RF inference
            fake_np = fake_features.detach().numpy().tolist()

            # Get RF's P(bot) for each generated sample
            bot_probs = classifier.predict_batch(fake_np)

            # Loss: we want P(bot) to be low (successful evasion)  Use mean P(bot) as a scalar loss signal
            mean_bot_prob = sum(bot_probs) / len(bot_probs)

            # Since RF is non-differentiable, use the mean P(bot) as a surrogate loss on the generator's output magnitude
            # (higher norm outputs → more human-like)
            feature_norm = fake_features.norm(dim=1).mean()
            loss = torch.tensor(mean_bot_prob, dtype=torch.float32,
                                requires_grad=False) - 0.01 * feature_norm

            # Backprop through the surrogate
            surrogate_loss = (fake_features ** 2).mean() * mean_bot_prob
            surrogate_loss.backward()
            optimiser.step()

            if epoch % 10 == 0:
                logger.debug(
                    "GAN round=%d epoch=%d mean_bot_prob=%.3f",
                    round_num, epoch, mean_bot_prob
                )

        # Sample n adversarial vectors from the trained generator
        generator.eval()
        with torch.no_grad():
            noise = torch.randn(n, cfg.noise_dim, device=device)
            generated = generator(noise).numpy().tolist()

        # Filter to only return vectors that actually achieve evasion
        evasion_vecs = [
            v for v in generated
            if classifier.predict_proba(v) < cfg.evasion_target
        ]

        # If too few evade, supplement with perturbation fallback
        if len(evasion_vecs) < n // 2:
            logger.debug(
                "GAN produced only %d evasion vectors — supplementing with perturbation.",
                len(evasion_vecs)
            )
            extra = self._perturbation_generate(
                classifier, generated, n - len(evasion_vecs), round_num
            )
            evasion_vecs.extend(extra)

        return evasion_vecs[:n]

    def _perturbation_generate(
        self,
        classifier: RandomForestClassifier,
        seed_X: List[List[float]],
        n: int,
        round_num: int,
    ) -> List[List[float]]:
        """
        Generate adversarial vectors by perturbing known bot samples.
        list of list of float
        """
        cfg      = self.config
        rng      = random.Random(cfg.random_seed + round_num)
        # Cooling schedule: perturbation scale decreases with rounds
        scale    = cfg.perturbation_scale * (0.85 ** (round_num - 1))
        results  = []
        attempts = 0
        max_attempts = n * 20   # hard limit on iterations

        while len(results) < n and attempts < max_attempts:
            attempts += 1

            # Pick a random seed vector
            seed = rng.choice(seed_X)
            candidate = list(seed)

            # Random perturbation of all features
            for i in range(ML_FEATURE_DIM):
                delta = rng.gauss(0, scale)
                candidate[i] = max(0.0, min(1.0, candidate[i] + delta))

            # Accept if it achieves evasion
            if not classifier.is_trained:
                results.append(candidate)  # untrained: accept all
            elif classifier.predict_proba(candidate) < cfg.evasion_target:
                results.append(candidate)

        # If insufficient evasion vectors found, fill with perturbed samples that minimise P(bot) even if above evasion_target
        if len(results) < n:
            remaining = n - len(results)
            for _ in range(remaining):
                seed      = rng.choice(seed_X)
                candidate = [
                    max(0.0, min(1.0, v + rng.gauss(0, scale * 2)))
                    for v in seed
                ]
                results.append(candidate)

        logger.debug(
            "Perturbation round=%d: generated %d/%d evasion samples (scale=%.3f)",
            round_num, len(results), n, scale
        )
        return results[:n]

    # HELPERS
    def _measure_evasion_rate(
        self,
        classifier: RandomForestClassifier,
        adv_X: List[List[float]],
    ) -> float:
        """
        Measure what fraction of adversarial samples evade the classifier.
        """
        if not adv_X or not classifier.is_trained:
            return 1.0   # untrained classifier evades nothing (worst case)

        bot_probs = classifier.predict_batch(adv_X)
        evaded    = sum(1 for p in bot_probs if p < self.config.evasion_target)
        return evaded / len(adv_X)

    @staticmethod
    def _balance(
        X: List[List[float]],
        y: List[int],
    ) -> Tuple[List[List[float]], List[int]]:
        """Downsample majority class to balance training data."""
        idx_0 = [i for i, lbl in enumerate(y) if lbl == 0]
        idx_1 = [i for i, lbl in enumerate(y) if lbl == 1]
        min_n = min(len(idx_0), len(idx_1))
        if min_n == 0:
            return X, y
        s0 = random.sample(idx_0, min_n)
        s1 = random.sample(idx_1, min_n)
        all_idx = s0 + s1
        random.shuffle(all_idx)
        return [X[i] for i in all_idx], [y[i] for i in all_idx]

    def __repr__(self) -> str:
        return (
            f"AdversarialTrainer("
            f"rounds={self.config.n_rounds}, "
            f"samples/round={self.config.samples_per_round}, "
            f"backend={'gan' if _TORCH_AVAILABLE else 'perturbation'})"
        )
