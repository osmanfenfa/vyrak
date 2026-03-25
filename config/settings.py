"""
Central application settings loaded from environment variables.
"""

import os
from functools import lru_cache


class Settings:
    """Flat settings object — no Pydantic required at import time."""

    database_url: str   = os.environ.get("DATABASE_URL", "")
    db_pool_size: int   = int(os.environ.get("DB_POOL_SIZE", "10"))
    db_max_overflow: int = int(os.environ.get("DB_MAX_OVERFLOW", "20"))
    db_echo: bool       = os.environ.get("DB_ECHO", "false").lower() == "true"

    redis_url: str          = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    redis_max_connections: int = int(os.environ.get("REDIS_MAX_CONNECTIONS", "50"))
    profile_ttl_seconds: int   = int(os.environ.get("PROFILE_TTL_SECONDS", str(90 * 86400)))

    jwt_secret_key: str     = os.environ.get("JWT_SECRET_KEY", "zelvryn-dev-secret-CHANGE-IN-PROD")
    jwt_algorithm: str      = os.environ.get("JWT_ALGORITHM", "HS256")
    jwt_expiry_minutes: int = int(os.environ.get("JWT_EXPIRY_MINUTES", "60"))

    model_dir: str          = os.environ.get("MODEL_PATH", "models")
    population_prior_path: str = os.environ.get(
        "POPULATION_PRIOR_PATH", "data/processed/population_prior.json"
    )

    shadow_mode_enabled: bool = (
        os.environ.get("SHADOW_MODE_ENABLED", "false").lower() == "true"
    )
    adversarial_hardening: bool = (
        os.environ.get("ADVERSARIAL_HARDENING", "false").lower() == "true"
    )
    canary_traffic_percent: int = int(os.environ.get("CANARY_TRAFFIC_PERCENT", "5"))

    ml_bot_confidence_thresh: float = float(
        os.environ.get("ML_BOT_CONFIDENCE_THRESH", "0.78")
    )
    ml_retrain_interval: int = int(os.environ.get("ML_RETRAIN_INTERVAL", "50"))

    api_host: str       = os.environ.get("API_HOST", "0.0.0.0")
    api_port: int       = int(os.environ.get("API_PORT", "8000"))
    api_workers: int    = int(os.environ.get("API_WORKERS", "1"))
    cors_origins: str   = os.environ.get(
        "CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
    )
    zelvryn_env: str    = os.environ.get("ZELVRYN_ENV", "development")
    
    log_level: str      = os.environ.get("LOG_LEVEL", "INFO").upper()
    audit_log_dir: str  = os.environ.get("AUDIT_LOG_DIR", "logs")

    shadow_export_dir: str = os.environ.get("SHADOW_EXPORT_DIR", "data/shadow_exports")

    @property
    def is_production(self) -> bool:
        return self.zelvryn_env == "production"

    @property
    def is_development(self) -> bool:
        return self.zelvryn_env == "development"

    @property
    def cors_origins_list(self) -> list:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def __repr__(self) -> str:
        return (
            f"Settings(env={self.zelvryn_env}, "
            f"shadow={self.shadow_mode_enabled}, "
            f"model_dir={self.model_dir!r})"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

settings = get_settings()