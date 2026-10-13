"""Verify the public HTTPS API without writing credentials into test evidence."""

import argparse
import asyncio
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import httpx

from decider_vllm.evaluation import compare_responses, percentile, validate_response


async def verify(args):
    credentials = json.loads(args.credentials.read_text())
    url = credentials["endpoint"].rstrip("/")
    if not url.startswith("https://"):
        raise ValueError("A verified HTTPS endpoint is required")
    key = credentials["api_key"]
    headers = {"Authorization": f"Bearer {key}"}
    captures = json.loads(args.reference.read_text())["results"]
    measured = []
    checks = {}
    args.output.mkdir(parents=True, exist_ok=False)

    async with httpx.AsyncClient(
        base_url=url,
        timeout=40,
        trust_env=False,
        limits=httpx.Limits(
            max_connections=max(64, args.overload_requests),
            max_keepalive_connections=max(64, args.overload_requests),
        ),
    ) as client:
        health = await client.get("/health")
        checks["health"] = health.status_code == 200 and health.json().get("status") == "ok"
        docs = await client.get("/docs")
        checks["interactive_docs"] = docs.status_code == 200 and "SwaggerUIBundle" in docs.text
        schema = (await client.get("/openapi.json")).json()
        checks["documented_bearer_auth"] = schema["paths"]["/v1/systemone"]["post"]["security"] == [
            {"BearerAPIKey": []}
        ]
        checks["documented_response"] = (
            "latency_ms" in schema["components"]["schemas"]["DecisionResponse"]["properties"]
        )
        no_key = await client.post("/v1/systemone", json=captures[0]["request"])
        bad_key = await client.post(
            "/v1/systemone",
            json=captures[0]["request"],
            headers={"Authorization": "Bearer invalid-test-key"},
        )
        checks["no_key_rejected"] = no_key.status_code == 401
        checks["bad_key_rejected"] = bad_key.status_code == 401
        checks["authentication_before_validation"] = (
            await client.post("/v1/systemone", content=b"{")
        ).status_code == 401
        checks["models_require_auth"] = (await client.get("/v1/models")).status_code == 401
        models = await client.get("/v1/models", headers=headers)
        checks["model_listing"] = models.status_code == 200 and models.json()["data"][0][
            "capabilities"
        ] == ["choice", "noul", "score"]
        checks["invalid_input_rejected"] = (
            await client.post("/v1/systemone", json={}, headers=headers)
        ).status_code == 422
        checks["oversized_input_rejected"] = (
            await client.post("/v1/systemone", content=b"x" * (1024 * 1024 + 1), headers=headers)
        ).status_code == 413
        insecure = await client.get(url.replace("https://", "http://", 1) + "/health")
        checks["https_required"] = insecure.status_code == 403

        async def infer(case, phase, index):
            started = perf_counter()
            row = {"phase": phase, "index": index, "case": case["id"]}
            try:
                response = await client.post("/v1/systemone", json=case["request"], headers=headers)
                row.update(
                    status=response.status_code,
                    client_ms=(perf_counter() - started) * 1000,
                    request_id=response.headers.get("x-request-id"),
                    cache_control=response.headers.get("cache-control"),
                    retry_after=response.headers.get("retry-after"),
                )
                if response.status_code == 200:
                    payload = response.json()
                    validate_response(case["request"], payload)
                    row.update(
                        response=payload,
                        comparison=compare_responses(case["response"], payload, tolerance=0.02),
                    )
                else:
                    row["error"] = response.json().get("detail", "HTTP error")
            except Exception as error:
                row.update(status=0, error_type=type(error).__name__)
            measured.append(row)
            return row

        await asyncio.sleep(1)
        for index, case in enumerate(captures):
            await infer(case, "parity", index)
            await asyncio.sleep(0.06)
        await asyncio.sleep(1)
        # Burst intentionally exceeds admission/rate limits; 429 is the expected response.
        await asyncio.gather(
            *(
                infer(captures[index % 20], "overload", index)
                for index in range(args.overload_requests)
            )
        )
        await asyncio.sleep(1.2)
        # Eight simultaneous requests per batch, paced below the service's aggregate rate.
        for batch in range(20):
            await asyncio.gather(
                *(
                    infer(captures[(batch * 8 + index) % 20], "concurrency_8", batch * 8 + index)
                    for index in range(8)
                )
            )
            await asyncio.sleep(0.6)
        checks["unauthenticated_response_not_cached"] = (
            await client.post("/v1/systemone", json=captures[0]["request"])
        ).status_code == 401
        final_health = (await client.get("/health")).json()
        checks["capacity_recovered"] = (
            final_health.get("status") == "ok" and final_health.get("pending_requests") == 0
        )

    parity = [r for r in measured if r["phase"] == "parity"]
    steady = [r for r in measured if r["phase"] == "concurrency_8"]
    overload = [r for r in measured if r["phase"] == "overload"]
    successful = [r for r in measured if r["status"] == 200]
    checks.update(
        all_primitives_passed=len(parity) == 21
        and all(r["status"] == 200 and r["comparison"]["passed"] for r in parity),
        concurrency_8_passed=len(steady) == 160
        and all(r["status"] == 200 and r["comparison"]["passed"] for r in steady),
        overload_rejected=any(r["status"] == 429 for r in overload),
        overload_no_server_errors=all(r["status"] in (200, 429) for r in overload),
        overload_retry_after=all(r.get("retry_after") for r in overload if r["status"] == 429),
        successful_choices_match=all(r["comparison"]["passed"] for r in successful),
        responses_not_cached=all(r["cache_control"] == "no-store" for r in successful),
        unique_request_ids=(
            all(r["request_id"] for r in successful)
            and len({r["request_id"] for r in successful}) == len(successful)
        ),
    )
    phases = {}
    for phase in ("parity", "overload", "concurrency_8"):
        rows = [r for r in measured if r["phase"] == phase]
        values = [r["client_ms"] for r in rows if r["status"] == 200]
        phases[phase] = {
            "requests": len(rows),
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "mean_client_ms": sum(values) / len(values) if values else None,
            "p50_client_ms": percentile(values, 0.5),
            "p95_client_ms": percentile(values, 0.95),
        }
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "endpoint": url,
        "passed": all(checks.values()),
        "checks": checks,
        "phases": phases,
        "overload_requests": args.overload_requests,
        "method": "Public HTTPS; 21 original-reference cases, overload burst, paced batches at concurrency 8",
    }
    encoded = json.dumps(report, indent=2)
    raw = "\n".join(json.dumps(row, ensure_ascii=False) for row in measured) + "\n"
    assert key not in encoded and key not in raw
    (args.output / "verification.json").write_text(encoded + "\n")
    (args.output / "requests.jsonl").write_text(raw)
    print(encoded)
    if not report["passed"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("docs/results/gpu-concurrency-2026-10-13/reference-capture.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--overload-requests",
        type=int,
        default=64,
        help="Burst above the configured admission limit; use 512 for the expanded deployment",
    )
    args = parser.parse_args()
    if args.overload_requests < 1:
        parser.error("--overload-requests must be positive")
    asyncio.run(verify(args))


if __name__ == "__main__":
    main()
