import asyncio
import logging

import httpx
import pytest
from test_server import Backend

from decider_vllm.server import create_app

KEY = "test-key-" + "x" * 40


async def test_authentication_precedes_body_reads_and_does_not_start_inference():
    backend = Backend()
    app = create_app(lambda: backend, api_key=KEY, warmup=False, max_body_bytes=20)
    messages = []

    async def receive():
        raise AssertionError("Unauthorized request body must not be read")

    async def send(message):
        messages.append(message)

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
                "headers": [],
                "client": ("127.0.0.1", 1),
                "server": ("127.0.0.1", 8100),
            },
            receive,
            send,
        )
    assert messages[0]["status"] == 401
    assert not backend.started.is_set()
    assert dict(messages[0]["headers"])[b"cache-control"] == b"no-store"


@pytest.mark.parametrize("authorization", ["", "Bearer wrong", f"Basic {KEY}"])
async def test_invalid_keys_are_rejected(authorization, payment):
    app = create_app(Backend, api_key=KEY, warmup=False)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.post(
            "/v1/systemone", json=payment.model_dump(), headers={"Authorization": authorization}
        )
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"


async def test_valid_key_models_docs_and_request_ids(payment, caplog):
    caplog.set_level(logging.INFO)
    backend = Backend()
    backend.finish.set()
    app = create_app(lambda: backend, api_key=KEY, warmup=False)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        assert (await client.get("/health")).status_code == 200
        schema = (await client.get("/openapi.json")).json()
        assert schema["paths"]["/v1/systemone"]["post"]["security"] == [{"BearerAPIKey": []}]
        assert "latency_ms" in schema["components"]["schemas"]["DecisionResponse"]["properties"]
        assert (await client.get("/v1/models")).status_code == 401
        headers = {"Authorization": f"Bearer {KEY}"}
        models = await client.get("/v1/models", headers=headers)
        assert models.json()["data"][0]["id"] == "test-only"
        response = await client.post("/v1/systemone", json=payment.model_dump(), headers=headers)
        assert response.status_code == 200
        assert len(response.headers["x-request-id"]) == 32
        assert response.headers["x-request-id"] != models.headers["x-request-id"]
        assert response.headers["cache-control"] == "no-store"
    assert KEY not in caplog.text
    assert "http_completed" in caplog.text


async def test_rate_limit_rejects_before_backend_and_allows_health(monkeypatch, payment):
    from decider_vllm import security

    moment = [100.0]
    monkeypatch.setattr(security, "monotonic", lambda: moment[0])
    backend = Backend()
    backend.finish.set()
    app = create_app(
        lambda: backend, api_key=KEY, warmup=False, requests_per_minute=60, rate_limit_burst=1
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        headers = {"Authorization": f"Bearer {KEY}"}
        first = await client.get("/v1/models", headers=headers)
        assert first.status_code == 200
        limited = await client.post("/v1/systemone", json=payment.model_dump(), headers=headers)
        assert limited.status_code == 429
        assert limited.headers["retry-after"] == "1"
        assert not backend.started.is_set()
        assert (await client.get("/health")).status_code == 200
        moment[0] += 1
        assert (await client.get("/v1/models", headers=headers)).status_code == 200


async def test_authenticated_admission_limit_releases_after_completion(payment):
    backend = Backend()
    app = create_app(lambda: backend, api_key=KEY, max_pending_requests=1, warmup=False)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": f"Bearer {KEY}"},
        ) as client,
    ):
        first = asyncio.create_task(client.post("/v1/systemone", json=payment.model_dump()))
        await backend.started.wait()
        assert (await client.post("/v1/systemone", json=payment.model_dump())).status_code == 429
        backend.finish.set()
        assert (await first).status_code == 200
        assert (await client.post("/v1/systemone", json=payment.model_dump())).status_code == 200
