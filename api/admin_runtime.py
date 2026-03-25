import logging
from typing import Any, Dict

from fastapi import FastAPI

logger = logging.getLogger(__name__)


def default_runtime_config() -> Dict[str, Any]:
    # Import lazily so we always read the latest module-level values.
    import core.constants as c

    return {
        "trust_initial": float(c.TRUST_INITIAL),
        "trust_decay_mild": float(c.TRUST_DECAY_MILD),
        "trust_decay_moderate": float(c.TRUST_DECAY_MODERATE),
        "trust_decay_severe": float(c.TRUST_DECAY_SEVERE),
        "trust_decay_ai_bot": float(c.TRUST_DECAY_AI_BOT),
        "trust_recovery_rate": float(c.TRUST_RECOVERY_RATE),
        "trust_challenge_threshold": float(c.TRUST_CHALLENGE_THRESHOLD),
        "trust_block_threshold": float(c.TRUST_BLOCK_THRESHOLD),
        "ml_bot_confidence_thresh": float(c.ML_BOT_CONFIDENCE_THRESH),
        "ml_retrain_interval": int(c.ML_RETRAIN_INTERVAL),
        "cold_start_ramp_actions": int(c.COLD_START_RAMP_ACTIONS),
        "cold_start_ml_thresh": float(c.COLD_START_ML_THRESH),
        "cold_start_temporal_limit": float(c.COLD_START_TEMPORAL_LIMIT),
        "cold_start_recovery_rate": float(c.COLD_START_RECOVERY_RATE),
        "ewma_decay_factor": float(c.EWMA_DECAY_FACTOR),
        "canary_traffic_percent": int(c.CANARY_TRAFFIC_PERCENT),
    }


def get_runtime_config(app: FastAPI) -> Dict[str, Any]:
    cfg = getattr(app.state, "runtime_config", None)
    if cfg is None:
        cfg = default_runtime_config()
        app.state.runtime_config = cfg
    return cfg


def apply_runtime_patch(app: FastAPI, patch: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(get_runtime_config(app))
    cfg.update(patch)
    _apply_runtime_config_to_modules(app, cfg)
    app.state.runtime_config = cfg
    return cfg


def _apply_runtime_config_to_modules(app: FastAPI, cfg: Dict[str, Any]) -> None:
    import core.constants as constants_mod
    import core.engine as engine_mod
    import core.profile as profile_mod
    import coldstart.ramp_manager as ramp_mod
    import api.routers.sessions as sessions_mod
    import api.dependencies as deps_mod

    module_updates = {
        "TRUST_INITIAL": cfg["trust_initial"],
        "TRUST_DECAY_MILD": cfg["trust_decay_mild"],
        "TRUST_DECAY_MODERATE": cfg["trust_decay_moderate"],
        "TRUST_DECAY_SEVERE": cfg["trust_decay_severe"],
        "TRUST_DECAY_AI_BOT": cfg["trust_decay_ai_bot"],
        "TRUST_RECOVERY_RATE": cfg["trust_recovery_rate"],
        "TRUST_CHALLENGE_THRESHOLD": cfg["trust_challenge_threshold"],
        "TRUST_BLOCK_THRESHOLD": cfg["trust_block_threshold"],
        "ML_BOT_CONFIDENCE_THRESH": cfg["ml_bot_confidence_thresh"],
        "ML_RETRAIN_INTERVAL": cfg["ml_retrain_interval"],
        "COLD_START_RAMP_ACTIONS": cfg["cold_start_ramp_actions"],
        "COLD_START_ML_THRESH": cfg["cold_start_ml_thresh"],
        "COLD_START_TEMPORAL_LIMIT": cfg["cold_start_temporal_limit"],
        "COLD_START_RECOVERY_RATE": cfg["cold_start_recovery_rate"],
        "EWMA_DECAY_FACTOR": cfg["ewma_decay_factor"],
        "CANARY_TRAFFIC_PERCENT": cfg["canary_traffic_percent"],
    }

    for mod in (
        constants_mod,
        engine_mod,
        ramp_mod,
        profile_mod,
        sessions_mod,
    ):
        for key, value in module_updates.items():
            if hasattr(mod, key):
                setattr(mod, key, value)

    registry = getattr(app.state, "model_registry", None)
    if registry is not None:
        registry.canary_traffic_percent = int(cfg["canary_traffic_percent"])
        if hasattr(registry, "_save_registry"):
            try:
                registry._save_registry()
            except Exception as e:
                logger.warning("Could not persist registry canary traffic update: %s", e)

    app.state.shadow_mode = bool(
        cfg.get("shadow_mode", getattr(app.state, "shadow_mode", False))
    )

    # Update active sessions immediately.
    for engine, _ in list(getattr(deps_mod, "_session_cache", {}).values()):
        try:
            engine.shadow_mode = bool(app.state.shadow_mode)
            engine.trust_score = min(
                100.0,
                max(0.0, float(cfg.get("trust_initial", engine.trust_score))),
            ) if not engine.result_history else engine.trust_score
            if hasattr(engine, "_ramp"):
                ramp_actions = max(1, int(cfg["cold_start_ramp_actions"]))
                engine._ramp.ramp_actions = ramp_actions
                engine._ramp._interp_start = int(
                    ramp_actions * (1 - engine._ramp.interpolation_fraction)
                )
        except Exception as e:
            logger.warning("Failed to apply runtime config to active engine: %s", e)
