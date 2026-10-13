import asyncio
from types import SimpleNamespace

import httpx
from strands_decider.schema import SystemOneResponse, Usage

from decider_vllm.server import create_app


class Backend:
    settings = SimpleNamespace(model_name="test-only")

    def __init__(self):
        self.started = asyncio.Event()
        self.finish = asyncio.Event()
        self.cancelled = False

    async def evaluate(self, request):
        self.started.set()
        try:
            await self.finish.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return SystemOneResponse(
            model="test-only", answers={}, usage=Usage(input_tokens=1, output_tokens=0)
        )

    async def check_health(self):
        pass

    def shutdown(self):
        pass


async def test_queue_limit_rejects_excess_then_releases(payment):
    backend = Backend()
    app = create_app(lambda: backend, max_pending_requests=1, warmup=False)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        first = asyncio.create_task(client.post("/v1/systemone", json=payment.model_dump()))
        await backend.started.wait()
        second = await client.post("/v1/systemone", json=payment.model_dump())
        assert second.status_code == 429
        backend.finish.set()
        assert (await first).status_code == 200
        assert (await client.get("/health")).json()["pending_requests"] == 0


async def test_deadline_cancels_inference(payment):
    backend = Backend()
    app = create_app(lambda: backend, timeout_seconds=0.03, warmup=False)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.post("/v1/systemone", json=payment.model_dump())
        assert response.status_code == 504
        assert backend.cancelled
        assert app.state.pending == 0


async def test_streamed_body_limit_without_content_length():
    app = create_app(Backend, max_body_bytes=20, warmup=False)

    async def body():
        yield b"x" * 12
        yield b"x" * 12

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.post("/v1/systemone", content=body())
        assert response.status_code == 413


async def test_client_disconnect_cancels_backend(payment):
    backend = Backend()
    app = create_app(lambda: backend, warmup=False)
    body = payment.model_dump_json().encode()
    first = True
    sent = []

    async def receive():
        nonlocal first
        if first:
            first = False
            return {"type": "http.request", "body": body, "more_body": False}
        await backend.started.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    async with app.router.lifespan_context(app):
        await app(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/v1/systemone",
                "raw_path": b"/v1/systemone",
                "query_string": b"",
                "headers": [(b"content-type", b"application/json")],
                "client": ("127.0.0.1", 1),
                "server": ("127.0.0.1", 8100),
            },
            receive,
            send,
        )
    assert backend.cancelled
    assert app.state.pending == 0
