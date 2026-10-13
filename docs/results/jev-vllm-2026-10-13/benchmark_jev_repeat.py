"""Compare public Decider and direct Jev APIs with identical business inputs."""

import argparse
import asyncio
import hashlib
import json
import platform
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import httpx

from decider_vllm.evaluation import (
    compare_responses,
    load_cases,
    percentile,
    validate_response,
)

JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"


def now():
    return datetime.now(UTC).isoformat()


def distribution(values):
    return (
        {
            "n": len(values),
            "mean": sum(values) / len(values),
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
            "min": min(values),
            "max": max(values),
        }
        if values
        else None
    )


async def run(args):
    credentials = json.loads(args.credentials.read_text())
    jev_key = args.jev_key_file.expanduser().read_text().strip()
    secrets = [credentials["api_key"], jev_key]
    if not all(key and not any(c.isspace() for c in key) for key in secrets):
        raise ValueError("Invalid credential format")
    decider_endpoint = credentials["endpoint"].rstrip("/") + "/v1/systemone"
    if not decider_endpoint.startswith("https://"):
        raise ValueError("Use HTTPS to protect credentials")
    cases = load_cases(args.corpus)
    references = {
        row["id"]: row["response"]
        for row in json.loads(args.reference.read_text())["results"]
        if row["id"] in {case["id"] for case in cases}
    }
    reference_rows = json.loads(args.reference.read_text())["results"]
    for case in cases:
        reference = next(row for row in reference_rows if row["id"] == case["id"])
        if reference["request"] != case["request"]:
            raise ValueError("Decider reference must use identical inputs")
    if (
        args.requests % len(cases)
        or not args.levels
        or args.requests < max(args.levels)
        or min(args.levels) < 1
        or len(set(args.levels)) != len(args.levels)
        or len(set(args.providers)) != len(args.providers)
    ):
        raise ValueError("Use distinct positive levels and a corpus-multiple request count")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "corpus.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")
    manifest = {
        "started_at": now(),
        "endpoints": {"decider": decider_endpoint, "jev": JEV_ENDPOINT},
        "jev_requested_model": "jev-latest",
        "decider_expected_config": {
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 128,
            "max_inflight_questions": 256,
            "max_pending_requests": 256,
            "requests_per_minute": 120000,
            "rate_limit_burst": 512,
            "server_timeout_seconds": 25,
        },
        "design": {
            "providers": args.providers,
            "levels": args.levels,
            "requests_per_cell": args.requests,
            "cases": len(cases),
            "repeats_per_case": args.requests // len(cases),
            "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
            "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
            "method": "Fixed workers; closed loop; immediate replacement; no client retries",
            "provider_order": "Sequential providers; alternate first provider at each level",
            "dispatch_order": "Round-robin corpus index; identical at each provider and level",
            "warmup": "max(corpus size, 2 * concurrency) before each cell; excluded",
            "transport": "Same client host; HTTPS/HTTP1.1; persistent pools; no redirects",
            "timeout_seconds": 40,
            "connect_timeout_seconds": 8,
            "percentiles": "Nearest rank; success-only; failures also retained separately",
            "stop_policy": (
                "Stop new submissions and higher levels for a provider on any HTTP "
                "400/401/403/413/422/429, or five other failures in one phase. "
                "Existing in-flight calls finish; no retries or quota bypass."
            ),
            "comparison_scope": (
                "Public API performance, not pure model compute. Jev bypasses the demo ECS "
                "and Lambda bridge. Provider infrastructure and network paths differ."
            ),
        },
        "client": {"python": platform.python_version(), "architecture": platform.machine()},
        "cells": [],
        "skipped": [],
    }
    own_baselines = {"decider": {}, "jev": {}}
    stopped = {}

    def save():
        manifest["provider_stop_reasons"] = stopped
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )

    save()
    limits = httpx.Limits(
        max_connections=max(args.levels),
        max_keepalive_connections=max(args.levels),
        keepalive_expiry=60,
    )
    timeout = httpx.Timeout(40, connect=8)
    async with (
        httpx.AsyncClient(
            headers={"Authorization": "Bearer " + credentials["api_key"]},
            limits=limits,
            timeout=timeout,
            trust_env=False,
            follow_redirects=False,
        ) as decider,
        httpx.AsyncClient(
            headers={"Authorization": "Bearer " + jev_key},
            limits=limits,
            timeout=timeout,
            trust_env=False,
            follow_redirects=False,
        ) as jev,
    ):
        health_url = credentials["endpoint"].rstrip("/") + "/health"
        health = await decider.get(health_url)
        health.raise_for_status()
        (args.output / "initial-health.json").write_text(health.text + "\n")
        if health.json().get("pending_requests") != 0:
            raise RuntimeError("Decider is already processing requests")
        with (
            (args.output / "requests.jsonl").open("w") as measured,
            (args.output / "warmups.jsonl").open("w") as warmups,
        ):

            async def phase(provider, concurrency, count, sink, kind):
                rows = []
                next_index = 0
                failures = 0
                stop_reason = None
                client = decider if provider == "decider" else jev
                endpoint = manifest["endpoints"][provider]

                async def worker():
                    nonlocal next_index, failures, stop_reason
                    while next_index < count and stop_reason is None:
                        index = next_index
                        next_index += 1
                        case = cases[index % len(cases)]
                        payload = case["request"]
                        if provider == "jev":
                            payload = {**payload, "model": "jev-latest"}
                        row = {
                            "provider": provider,
                            "phase": kind,
                            "concurrency": concurrency,
                            "index": index,
                            "case": case["id"],
                            "started_at": now(),
                        }
                        started = perf_counter()
                        try:
                            response = await client.post(endpoint, json=payload)
                            row.update(
                                client_ms=(perf_counter() - started) * 1000,
                                status=response.status_code,
                                request_id=response.headers.get("x-request-id")
                                or response.headers.get("request-id"),
                                headers={
                                    k: v
                                    for k, v in response.headers.items()
                                    if k.lower()
                                    in ("retry-after", "server-timing", "cache-control")
                                    or "ratelimit" in k.lower()
                                },
                            )
                            response.raise_for_status()
                            data = response.json()
                            validate_response(case["request"], data)
                            row.update(success=True, response=data)
                        except Exception as exc:
                            row.update(success=False, error_type=type(exc).__name__)
                            row.setdefault("status", 0)
                            row.setdefault("client_ms", (perf_counter() - started) * 1000)
                            # No exception messages or error bodies can echo credentials.
                            failures += 1
                            if row["status"] in (400, 401, 403, 413, 422, 429):
                                stop_reason = f"HTTP {row['status']}"
                            elif failures >= 5:
                                stop_reason = "Five failures in the same phase"
                        row["finished_at"] = now()
                        rows.append(row)
                        sink.write(json.dumps(row, ensure_ascii=False) + "\n")

                started_at = now()
                started = perf_counter()
                await asyncio.gather(*(worker() for _ in range(concurrency)))
                elapsed = perf_counter() - started
                finished_at = now()
                sink.flush()
                return rows, elapsed, started_at, finished_at, stop_reason

            for level_index, concurrency in enumerate(args.levels):
                order = ("decider", "jev") if level_index % 2 == 0 else ("jev", "decider")
                order = tuple(provider for provider in order if provider in args.providers)
                for provider in order:
                    if provider in stopped:
                        manifest["skipped"].append(
                            {"provider": provider, "concurrency": concurrency}
                        )
                        save()
                        continue
                    print(
                        json.dumps(
                            {
                                "event": "cell_started",
                                "provider": provider,
                                "concurrency": concurrency,
                            }
                        ),
                        flush=True,
                    )
                    warm_count = max(len(cases), 2 * concurrency)
                    warm_rows, _, _, _, warm_stop = await phase(
                        provider, concurrency, warm_count, warmups, "warmup"
                    )
                    for row in warm_rows:
                        if row["success"]:
                            own_baselines[provider].setdefault(row["case"], row["response"])
                    if warm_stop or any(not row["success"] for row in warm_rows):
                        stopped[provider] = {
                            "reason": warm_stop or "Warmup failure",
                            "phase": "warmup",
                            "concurrency": concurrency,
                        }
                        save()
                        continue
                    rows, elapsed, begin, end, reason = await phase(
                        provider, concurrency, args.requests, measured, "measured"
                    )
                    successes = [row for row in rows if row["success"]]
                    choices = defaultdict(Counter)
                    audits = []
                    for row in successes:
                        choices[row["case"]][row["response"]["answers"]["action"]["choice"]] += 1
                        if provider == "decider":
                            audits.append(
                                {
                                    "index": row["index"],
                                    "case": row["case"],
                                    **compare_responses(
                                        references[row["case"]], row["response"], tolerance=0.02
                                    ),
                                }
                            )
                    if audits:
                        (args.output / f"decider-parity-c{concurrency}.json").write_text(
                            json.dumps(audits, ensure_ascii=False, indent=2) + "\n"
                        )
                    cell = {
                        "provider": provider,
                        "concurrency": concurrency,
                        "started_at": begin,
                        "finished_at": end,
                        "warmups": len(warm_rows),
                        "warmup_failures": sum(not row["success"] for row in warm_rows),
                        "planned_requests": args.requests,
                        "attempts": len(rows),
                        "successes": len(successes),
                        "failures": len(rows) - len(successes),
                        "complete": len(rows) == args.requests,
                        "wall_seconds": elapsed,
                        "successful_rps": len(successes) / elapsed,
                        "status_counts": dict(Counter(row["status"] for row in rows)),
                        "latency_ms": distribution([row["client_ms"] for row in successes]),
                        "failure_latency_ms": distribution(
                            [row["client_ms"] for row in rows if not row["success"]]
                        ),
                        "all_latency_ms": distribution([row["client_ms"] for row in rows]),
                        "server_latency_ms": distribution(
                            [
                                row["response"]["latency_ms"]
                                for row in successes
                                if isinstance(row["response"].get("latency_ms"), (int, float))
                            ]
                        ),
                        "models": dict(Counter(row["response"]["model"] for row in successes)),
                        "input_tokens": distribution(
                            [
                                row["response"]["usage"]["input_tokens"]
                                for row in successes
                                if isinstance(
                                    (row["response"].get("usage") or {}).get("input_tokens"),
                                    (int, float),
                                )
                            ]
                        ),
                        "output_tokens": distribution(
                            [
                                row["response"]["usage"]["output_tokens"]
                                for row in successes
                                if isinstance(
                                    (row["response"].get("usage") or {}).get("output_tokens"),
                                    (int, float),
                                )
                            ]
                        ),
                        "choices": dict(choices),
                        "choices_matching_own_initial_warmup": sum(
                            row["response"]["answers"]["action"]["choice"]
                            == own_baselines[provider][row["case"]]["answers"]["action"]["choice"]
                            for row in successes
                        ),
                        "decider_numeric_parity_passed": (
                            sum(audit["passed"] for audit in audits) if audits else None
                        ),
                    }
                    manifest["cells"].append(cell)
                    if reason:
                        stopped[provider] = {
                            "reason": reason,
                            "phase": "measured",
                            "concurrency": concurrency,
                        }
                    save()
                    print(
                        json.dumps(
                            {
                                k: cell[k]
                                for k in (
                                    "provider",
                                    "concurrency",
                                    "attempts",
                                    "successes",
                                    "failures",
                                    "successful_rps",
                                    "latency_ms",
                                    "status_counts",
                                )
                            }
                        ),
                        flush=True,
                    )
            health = await decider.get(health_url)
            health.raise_for_status()
            (args.output / "final-health.json").write_text(health.text + "\n")
            if health.json().get("pending_requests") != 0:
                raise RuntimeError("Pending Decider requests remain after test")
    manifest["finished_at"] = now()
    save()
    for path in args.output.iterdir():
        if path.is_file():
            content = path.read_text()
            assert not any(key in content for key in secrets), "Credential appeared in evidence"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--jev-key-file", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("examples/payment-corpus.json"))
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("docs/results/gpu-concurrency-2026-10-13/reference-capture.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=1280)
    parser.add_argument(
        "--providers", nargs="+", choices=["decider", "jev"], default=["decider", "jev"]
    )
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
