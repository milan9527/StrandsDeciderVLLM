"""Bounded HTTP admission, cancellation and the existing System One wire format."""

import asyncio
import logging
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from pydantic import Field
from strands_decider.schema import SystemOneRequest, SystemOneResponse

from .security import APIKeyMiddleware, RequestTraceMiddleware

logger = logging.getLogger(__name__)


class DecisionResponse(SystemOneResponse):
    latency_ms: float = Field(ge=0, description="Server-side elapsed time, including queuing")


class BodyLimit:
    def __init__(self, app, max_bytes: int):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        messages, total = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            total += len(message.get("body", b""))
            if total > self.max_bytes:
                return await JSONResponse(
                    {"detail": "Request body exceeds configured limit"}, status_code=413
                )(scope, receive, send)
            messages.append(message)
            if not message.get("more_body"):
                break
        index = 0

        async def replay():
            nonlocal index
            if index < len(messages):
                message = messages[index]
                index += 1
                return message
            return await receive()

        return await self.app(scope, replay, send)


async def run_until_disconnected(request: Request, evaluation):
    async def watch():
        # FastAPI has consumed the complete body before entering this handler.
        # Wait directly for disconnect instead of polling is_disconnected(),
        # whose AnyIO CancelScope can swallow cancellation of this watcher.
        while True:
            message = await request.receive()
            if message["type"] == "http.disconnect":
                return

    work, disconnect = asyncio.create_task(evaluation), asyncio.create_task(watch())
    try:
        done, _ = await asyncio.wait((work, disconnect), return_when=asyncio.FIRST_COMPLETED)
        if work in done:
            return work.result()
        raise HTTPException(499, "Client disconnected")
    finally:
        for task in (work, disconnect):
            if not task.done():
                task.cancel()
        await asyncio.gather(work, disconnect, return_exceptions=True)


def create_app(
    backend_factory,
    *,
    max_pending_requests: int = 32,
    timeout_seconds: float = 60,
    max_body_bytes: int = 1024 * 1024,
    warmup: bool = True,
    api_key: str | None = None,
    requests_per_minute: int = 1200,
    rate_limit_burst: int = 16,
) -> FastAPI:
    if min(max_pending_requests, timeout_seconds, max_body_bytes) <= 0:
        raise ValueError("HTTP limits must be positive")

    @asynccontextmanager
    async def lifespan(app):
        backend = backend_factory()
        app.state.backend = backend
        app.state.pending = 0
        try:
            if warmup:
                await backend.evaluate(
                    SystemOneRequest(
                        state="A confirmed payment matches its order.",
                        questions={"matches": {"type": "noul", "instructions": "Does it match?"}},
                    )
                )
            yield
        finally:
            backend.shutdown()

    app = FastAPI(
        title="Strands Decider · vLLM",
        version="0.1.0",
        description=(
            "Candidate-based decisions using the Decider v19 Pointer Head on vLLM. "
            "Supports choice, noul, and score. Supply your API key using Bearer authentication."
        ),
        lifespan=lifespan,
        swagger_ui_parameters={"persistAuthorization": False},
    )
    app.add_middleware(BodyLimit, max_bytes=max_body_bytes)
    security = []
    if api_key is not None:
        # Added outside BodyLimit so unauthorized callers cannot buffer request bodies.
        app.add_middleware(
            APIKeyMiddleware,
            api_key=api_key,
            requests_per_minute=requests_per_minute,
            burst=rate_limit_burst,
        )
        security = [Depends(HTTPBearer(auto_error=False, scheme_name="BearerAPIKey"))]
    app.add_middleware(RequestTraceMiddleware)

    @app.get("/")
    async def index():
        return {
            "service": "Strands Decider vLLM API",
            "inference": "/v1/systemone",
            "models": "/v1/models",
            "docs": "/docs",
            "authentication": "Bearer API key" if api_key is not None else "disabled",
        }

    @app.get("/v1/models", dependencies=security)
    async def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": app.state.backend.settings.model_name,
                    "object": "model",
                    "owned_by": "strands-decider",
                    "capabilities": ["choice", "noul", "score"],
                    "max_context_tokens": 4096,
                    "endpoint": "/v1/systemone",
                }
            ],
        }

    @app.get("/health")
    async def health():
        try:
            await app.state.backend.check_health()
        except Exception as exc:
            logger.exception("Engine health failed")
            raise HTTPException(503, "Inference engine unavailable") from exc
        return {
            "status": "ok",
            "engine": "vllm",
            "model": app.state.backend.settings.model_name,
            "pending_requests": app.state.pending,
        }

    @app.post(
        "/v1/systemone",
        dependencies=security,
        response_model=DecisionResponse,
        responses={
            401: {"description": "Missing or invalid API key"},
            413: {"description": "Request body exceeds the configured limit"},
            429: {"description": "Rate or concurrent request limit exceeded"},
            503: {"description": "Inference engine unavailable"},
            504: {"description": "Inference timed out and was cancelled"},
        },
    )
    async def systemone(payload: SystemOneRequest, request: Request):
        # These operations contain no await and are atomic within one event loop.
        if app.state.pending >= max_pending_requests:
            raise HTTPException(429, "Decision queue is full", headers={"Retry-After": "1"})
        app.state.pending += 1
        started = perf_counter()
        try:
            async with asyncio.timeout(timeout_seconds):
                response = await run_until_disconnected(
                    request, app.state.backend.evaluate(payload)
                )
            body = response.model_dump()
            body["latency_ms"] = round((perf_counter() - started) * 1000, 2)
            return JSONResponse(body)
        except TimeoutError as exc:
            raise HTTPException(504, "Decision deadline exceeded; requests cancelled") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Decision evaluation failed")
            raise HTTPException(503, "Inference engine failed; no decision returned") from exc
        finally:
            app.state.pending -= 1

    return app
