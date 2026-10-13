"""Authenticated public-API load test with separate warmups and behavior checks."""

import argparse
import asyncio
import json
from collections import Counter
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


async def run(args):
    credentials = json.loads(args.credentials.read_text())
    url, key = credentials["endpoint"].rstrip("/"), credentials["api_key"]
    if not url.startswith("https://"):
        raise ValueError("Use HTTPS to protect the API key")
    cases = load_cases(args.corpus)
    references = {row["id"]: row for row in json.loads(args.reference.read_text())["results"]}
    for case in cases:
        if references[case["id"]]["request"] != case["request"]:
            raise ValueError("Reference and test inputs must match")
    if (
        args.requests % len(cases)
        or args.requests < max(args.levels)
        or min(args.levels) < 1
        or len(set(args.levels)) != len(args.levels)
    ):
        raise ValueError("Use distinct positive levels and a corpus-multiple request count")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "endpoint": url,
        "label": args.label,
        "engine_max_num_seqs": args.engine_seqs,
        "max_num_batched_tokens": args.batch_tokens,
        "http_limit": args.http_limit,
        "question_limit": args.question_limit,
        "requests_per_minute": args.requests_per_minute,
        "burst": args.burst,
        "timeout_seconds": 25,
        "client_timeout_seconds": 40,
        "method": "Fixed workers, closed-loop public HTTPS; no retries; warmups excluded",
        "warmup_policy": "max(corpus size, 2 * concurrency) before every level",
        "percentiles": "Nearest-rank; successful responses only; failures included in elapsed time",
        "tolerance": 0.02,
        "cells": [],
    }
    async with httpx.AsyncClient(
        base_url=url,
        headers={"Authorization": "Bearer " + key},
        timeout=40,
        trust_env=False,
        limits=httpx.Limits(
            max_connections=max(args.levels), max_keepalive_connections=max(args.levels)
        ),
    ) as client:
        health = await client.get("/health")
        health.raise_for_status()
        (args.output / "initial-health.json").write_text(health.text + "\n")
        with (
            (args.output / "requests.jsonl").open("w") as raw,
            (args.output / "warmups.jsonl").open("w") as warm,
        ):

            async def phase(concurrency, count, sink):
                rows = []
                next_index = 0

                async def worker():
                    nonlocal next_index
                    while next_index < count:
                        index = next_index
                        next_index += 1
                        case = cases[index % len(cases)]
                        row = {
                            "concurrency": concurrency,
                            "index": index,
                            "case": case["id"],
                            "started_at": datetime.now(UTC).isoformat(),
                        }
                        started = perf_counter()
                        try:
                            response = await client.post("/v1/systemone", json=case["request"])
                            row.update(
                                client_ms=(perf_counter() - started) * 1000,
                                status=response.status_code,
                                request_id=response.headers.get("x-request-id"),
                            )
                            response.raise_for_status()
                            payload = response.json()
                            validate_response(case["request"], payload)
                            row.update(success=True, response=payload)
                        except Exception as exc:
                            row.update(
                                success=False,
                                error_type=type(exc).__name__,
                                error=str(exc).replace(key, "[REDACTED]"),
                            )
                            row.setdefault("client_ms", (perf_counter() - started) * 1000)
                            row.setdefault("status", 0)
                        rows.append(row)
                        sink.write(json.dumps(row, ensure_ascii=False) + "\n")

                started_at, started = datetime.now(UTC).isoformat(), perf_counter()
                await asyncio.gather(*(worker() for _ in range(concurrency)))
                elapsed = perf_counter() - started
                sink.flush()
                return rows, elapsed, started_at

            for concurrency in args.levels:
                print(
                    json.dumps({"event": "level_started", "concurrency": concurrency}), flush=True
                )
                warm_count = max(len(cases), 2 * concurrency)
                warm_rows, _, _ = await phase(concurrency, warm_count, warm)
                warm_failures = sum(not row["success"] for row in warm_rows)
                if warm_failures:
                    report.setdefault("warmup_failures", []).append(
                        {"concurrency": concurrency, "failures": warm_failures}
                    )
                    (args.output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
                    if not args.continue_after_warmup_failure:
                        raise RuntimeError("Warmup failed; inspect warmups.jsonl before continuing")
                    health = await client.get("/health")
                    if health.status_code != 200 or health.json().get("pending_requests") != 0:
                        raise RuntimeError("Service has not recovered after warmup failures")
                rows, elapsed, started_at = await phase(concurrency, args.requests, raw)
                success = [row for row in rows if row["success"]]
                comparisons = [
                    {
                        "index": row["index"],
                        "case": row["case"],
                        **compare_responses(
                            references[row["case"]]["response"],
                            row["response"],
                            tolerance=0.02,
                        ),
                    }
                    for row in success
                ]
                (args.output / f"behavior-c{concurrency}.json").write_text(
                    json.dumps(comparisons, ensure_ascii=False, indent=2) + "\n"
                )
                latencies = [row["client_ms"] for row in success]
                server = [row["response"]["latency_ms"] for row in success]
                cell = {
                    "concurrency": concurrency,
                    "started_at": started_at,
                    "finished_at": datetime.now(UTC).isoformat(),
                    "warmups_excluded": warm_count,
                    "warmup_failures": warm_failures,
                    "requests": len(rows),
                    "successes": len(success),
                    "failures": len(rows) - len(success),
                    "status_counts": dict(Counter(row["status"] for row in rows)),
                    "wall_seconds": elapsed,
                    "successful_rps": len(success) / elapsed,
                    "mean_ms": sum(latencies) / len(latencies) if latencies else None,
                    "p50_ms": percentile(latencies, 0.5),
                    "p95_ms": percentile(latencies, 0.95),
                    "p99_ms": percentile(latencies, 0.99),
                    "max_ms": max(latencies, default=None),
                    "server_mean_ms": sum(server) / len(server) if server else None,
                    "server_p95_ms": percentile(server, 0.95),
                    "behavior_passed": sum(row["passed"] for row in comparisons),
                    "changed_choices": sum(
                        not question.get("same_choice", True)
                        for row in comparisons
                        for question in row.get("questions", {}).values()
                    ),
                }
                report["cells"].append(cell)
                (args.output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(cell), flush=True)
                if cell["failures"]:
                    health = await client.get("/health")
                    if health.status_code != 200 or health.json().get("pending_requests") != 0:
                        raise RuntimeError("Service has not recovered after failed requests")
        health = await client.get("/health")
        (args.output / "final-health.json").write_text(health.text + "\n")
        health.raise_for_status()
        if health.json().get("pending_requests") != 0:
            raise RuntimeError("Pending requests remain after the benchmark")
    for path in args.output.iterdir():
        assert key not in path.read_text(), "Credential must not appear in test evidence"
    if report.get("warmup_failures") or any(
        cell["failures"] or cell["behavior_passed"] != cell["successes"] for cell in report["cells"]
    ):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("examples/payment-corpus.json"))
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("docs/results/gpu-concurrency-2026-10-13/reference-capture.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--engine-seqs", type=int, required=True)
    parser.add_argument(
        "--batch-tokens",
        type=int,
        default=8192,
        help="Expected server batch-token budget, recorded only; does not configure the server",
    )
    parser.add_argument("--http-limit", type=int)
    parser.add_argument("--question-limit", type=int)
    parser.add_argument("--requests-per-minute", type=int, default=12000)
    parser.add_argument("--burst", type=int, default=64)
    parser.add_argument("--continue-after-warmup-failure", action="store_true")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument("--requests", type=int, default=400)
    args = parser.parse_args()
    if args.http_limit is None:
        args.http_limit = args.engine_seqs
    if args.question_limit is None:
        args.question_limit = args.engine_seqs
    if (
        min(
            args.engine_seqs,
            args.batch_tokens,
            args.http_limit,
            args.question_limit,
            args.requests_per_minute,
            args.burst,
        )
        < 1
    ):
        parser.error("Expected server configuration values must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
