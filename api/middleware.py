"""
zelvryn/api/middleware.py
==========================
ASGI middleware stack for the Zelvryn FastAPI application.

Middleware components (applied outer → inner):
  1. RequestIDMiddleware   — injects X-Request-ID header into every request
                             and response for log correlation
  2. TimingMiddleware      — measures and attaches X-Process-Time-Ms header
  3. RequestLoggingMiddleware — structured access logging for every request
  4. AuditMiddleware       — writes CHALLENGE and BLOCK decisions to the
                             append-only security audit log
  5. SecurityHeadersMiddleware — attaches security-relevant HTTP response headers

CORS is configured in main.py via FastAPI's CORSMiddleware (not here) to keep
CORS origin configuration in one place with the application factory.

Author : Vyrak
System : Zelvryn — Adaptive Behavioral Signature Authentication
Version: 2.0
"""

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

logger          = logging.getLogger(__name__)
audit_logger    = logging.getLogger("zelvryn.audit")

# Path for the append-only audit log file
AUDIT_LOG_DIR:  str = os.environ.get("AUDIT_LOG_DIR",  "logs")
AUDIT_LOG_FILE: str = os.path.join(AUDIT_LOG_DIR, "audit.jsonl")


# ─────────────────────────────────────────────────────────────────────────────
# 1. REQUEST ID MIDDLEWARE
# ─────────────────────────────────────────────────────────────────────────────

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Ensures every request has a unique X-Request-ID header.

    If the client sends an X-Request-ID header, it is used (allowing clients
    to correlate their own logs with server logs). Otherwise a new UUID v4 is
    generated.

    The request ID is stored in request.state.request_id and echoed in the
    response headers.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = (
            request.headers.get("X-Request-ID")
            or str(uuid.uuid4())
        )
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# ─────────────────────────────────────────────────────────────────────────────
# 2. TIMING MIDDLEWARE
# ─────────────────────────────────────────────────────────────────────────────

class TimingMiddleware(BaseHTTPMiddleware):
    """
    Measures total request processing time and attaches it to the response.

    Adds the X-Process-Time-Ms header to every response. This is surfaced in
    the benchmark scripts and the analyst dashboard for p95 latency monitoring.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        t_start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        request.state.process_time_ms = elapsed_ms

        # Warn on slow responses (p99 target is < 100ms)
        if elapsed_ms > 100:
            logger.warning(
                "Slow response: %.1f ms [%s %s] request_id=%s",
                elapsed_ms,
                request.method,
                request.url.path,
                getattr(request.state, "request_id", ""),
            )

        return response


# ─────────────────────────────────────────────────────────────────────────────
# 3. REQUEST LOGGING MIDDLEWARE
# ─────────────────────────────────────────────────────────────────────────────

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Structured JSON access logging for every request.

    Logs:
      - Method, path, query string
      - Response status code
      - Processing time
      - Client IP
      - Request ID
      - User ID (if authenticated — set by route handlers in request.state)

    Does NOT log request or response bodies (PII risk).
    Skips logging for health check endpoints to reduce noise.
    """

    _SKIP_PATHS = {"/health", "/health/", "/favicon.ico"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip health checks
        if request.url.path in self._SKIP_PATHS:
            return await call_next(request)

        response = await call_next(request)

        # Build structured log record
        log_record = {
            "event":        "http_request",
            "method":       request.method,
            "path":         request.url.path,
            "query":        str(request.url.query) or None,
            "status_code":  response.status_code,
            "process_ms":   round(
                getattr(request.state, "process_time_ms", 0), 2
            ),
            "client_ip":    self._client_ip(request),
            "request_id":   getattr(request.state, "request_id", ""),
            "user_id":      getattr(request.state, "user_id", None),
            "timestamp":    time.time(),
        }

        log_level = logging.WARNING if response.status_code >= 500 else logging.INFO
        logger.log(log_level, json.dumps(log_record))

        return response

    @staticmethod
    def _client_ip(request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# 4. AUDIT MIDDLEWARE
# ─────────────────────────────────────────────────────────────────────────────

class AuditMiddleware(BaseHTTPMiddleware):
    """
    Writes security-significant events to an append-only audit log.

    The audit log captures:
      - All CHALLENGE and BLOCK decisions from action evaluation endpoints
      - All profile deletions (GDPR erasure events)
      - All model promotion events (forwarded via request.state.audit_event)

    The audit log is a JSONL file at logs/audit.jsonl. It must be:
      - Append-only (never truncated or modified)
      - Stored separately from application logs
      - Retained for the duration required by compliance requirements

    This middleware reads the audit_event set on request.state by route
    handlers. Route handlers that produce auditable events must set:
      request.state.audit_event = {
          "event_type": "...",
          "user_id": "...",
          ...
      }
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        Path(AUDIT_LOG_DIR).mkdir(parents=True, exist_ok=True)
        logger.debug("AuditMiddleware: logging to %s", AUDIT_LOG_FILE)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Check if a route handler set an audit event
        audit_event = getattr(request.state, "audit_event", None)
        if audit_event:
            self._write_audit_entry(
                event       = audit_event,
                request_id  = getattr(request.state, "request_id", ""),
                client_ip   = self._client_ip(request),
                status_code = response.status_code,
            )

        # Auto-audit CHALLENGE/BLOCK decisions from action endpoints
        decision = getattr(request.state, "zelvryn_decision", None)
        if decision in ("CHALLENGE", "BLOCK"):
            self._write_audit_entry(
                event = {
                    "event_type":  f"session_{decision.lower()}",
                    "user_id":     getattr(request.state, "user_id", None),
                    "session_id":  getattr(request.state, "session_id", None),
                    "trust_score": getattr(request.state, "trust_score", None),
                    "bot_prob":    getattr(request.state, "bot_prob", None),
                    "anomalies":   getattr(request.state, "anomalies", []),
                    "decision":    decision,
                },
                request_id  = getattr(request.state, "request_id", ""),
                client_ip   = self._client_ip(request),
                status_code = response.status_code,
            )

        return response

    def _write_audit_entry(
        self,
        event:       dict,
        request_id:  str,
        client_ip:   str,
        status_code: int,
    ) -> None:
        """Write a single audit entry to the JSONL audit log."""
        entry = {
            **event,
            "request_id":  request_id,
            "client_ip":   client_ip,
            "status_code": status_code,
            "logged_at":   time.time(),
        }
        try:
            with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as e:
            logger.error("Audit log write failed: %s", e)

        # Also emit to structured audit logger (for SIEM integration)
        audit_logger.info(json.dumps(entry))

    @staticmethod
    def _client_ip(request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# 5. SECURITY HEADERS MIDDLEWARE
# ─────────────────────────────────────────────────────────────────────────────

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Attaches security-relevant HTTP response headers to all API responses.

    These headers harden the API against common web vulnerabilities:
      - X-Content-Type-Options: prevents MIME sniffing
      - X-Frame-Options: prevents clickjacking
      - Strict-Transport-Security: enforces HTTPS (in production)
      - Cache-Control: prevents sensitive API responses from being cached
      - Referrer-Policy: limits referrer information leakage
      - X-Zelvryn-Version: service identification header
    """

    def __init__(self, app: ASGIApp, api_version: str = "2.0.0") -> None:
        super().__init__(app)
        self._version = api_version

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        response.headers["X-Content-Type-Options"]    = "nosniff"
        response.headers["X-Frame-Options"]            = "DENY"
        response.headers["Referrer-Policy"]            = "strict-origin-when-cross-origin"
        response.headers["Cache-Control"]              = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"]                     = "no-cache"
        response.headers["X-Zelvryn-Version"]          = self._version

        # HSTS — only enable in production (not in development where HTTP is used)
        if os.environ.get("ZELVRYN_ENV", "development") == "production":
            response.headers["Strict-Transport-Security"] = (
                "max-age=63072000; includeSubDomains; preload"
            )

        return response


# ─────────────────────────────────────────────────────────────────────────────
# MIDDLEWARE REGISTRATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def register_middleware(app) -> None:
    """
    Register all middleware on the FastAPI application.

    Middleware is applied in reverse registration order by Starlette:
    the last-registered middleware is the outermost wrapper.

    We register them here so main.py stays clean and so the middleware
    stack can be easily modified or extended.

    Parameters
    ----------
    app : FastAPI
        The application instance to attach middleware to.
    """
    # Applied last = outermost wrapper (processes requests first)
    app.add_middleware(SecurityHeadersMiddleware, api_version="2.0.0")
    app.add_middleware(AuditMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestIDMiddleware)
    # Applied first = innermost wrapper (processes requests last)

    logger.info("Middleware stack registered: 5 layers")
