"""Bearer authentication, bounded rate limiting, and safe request tracing."""

import contextvars
import hashlib
import json
import logging
import math
import secrets
import time
import uuid
from time import monotonic

from fastapi.responses import JSONResponse

request_id = contextvars.ContextVar("decider_request_id", default="-")
logger = logging.getLogger(__name__)
PUBLIC_PATHS = {"/", "/health", "/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"}


class APIKeyMiddleware:
    def __init__(self, app, *, api_key: str, requests_per_minute: int, burst: int):
        if len(api_key) < 32 or any(character.isspace() for character in api_key):
            raise ValueError("API key must have at least 32 characters and no whitespace")
        if requests_per_minute < 1 or burst < 1:
            raise ValueError("Rate limits must be positive")
        self.app = app
        self.digest = hashlib.sha256(api_key.encode()).digest()
        self.rate = requests_per_minute / 60
        self.capacity = burst
        self.tokens = float(burst)
        self.updated = monotonic()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or (
            scope["method"] in ("GET", "HEAD") and scope["path"] in PUBLIC_PATHS
        ):
            return await self.app(scope, receive, send)
        headers = [value for key, value in scope["headers"] if key.lower() == b"authorization"]
        valid = False
        if len(headers) == 1:
            scheme, separator, token = headers[0].partition(b" ")
            valid = (
                bool(separator)
                and scheme.lower() == b"bearer"
                and secrets.compare_digest(hashlib.sha256(token).digest(), self.digest)
            )
        if not valid:
            return await JSONResponse(
                {"detail": "A valid Bearer API key is required"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )(scope, receive, send)
        # One Uvicorn worker: no await between token accounting operations.
        now = monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now
        if self.tokens < 1:
            return await JSONResponse(
                {"detail": "API rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": str(max(1, math.ceil((1 - self.tokens) / self.rate)))},
            )(scope, receive, send)
        self.tokens -= 1
        return await self.app(scope, receive, send)


class RequestTraceMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        identifier = uuid.uuid4().hex
        marker = request_id.set(identifier)
        started, status = time.perf_counter(), 500

        async def traced_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message = dict(message)
                headers = [
                    (key, value)
                    for key, value in message.get("headers", [])
                    if key.lower() not in (b"x-request-id", b"cache-control")
                ]
                message["headers"] = [
                    *headers,
                    (b"x-request-id", identifier.encode()),
                    (b"cache-control", b"no-store"),
                ]
            await send(message)

        try:
            return await self.app(scope, receive, traced_send)
        finally:
            logger.info(
                json.dumps(
                    {
                        "event": "http_completed",
                        "request_id": identifier,
                        "method": scope["method"],
                        "path": scope["path"][:120],
                        "status": status,
                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                    }
                )
            )
            request_id.reset(marker)
