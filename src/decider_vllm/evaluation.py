"""Compare original/vLLM endpoints and measure closed-loop HTTP concurrency."""

import asyncio
import json
import math
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import httpx
from strands_decider.schema import SystemOneRequest, SystemOneResponse


def load_cases(path: Path) -> list[dict]:
    cases = json.loads(path.read_text())
    if not cases or not isinstance(cases, list):
        raise ValueError("Corpus must be a nonempty JSON list")
    for case in cases:
        SystemOneRequest.model_validate(case["request"])
    return cases


def compare_responses(reference: dict, candidate: dict, *, tolerance: float) -> dict:
    def within(difference):
        # Wire values are rounded decimals. Avoid rejecting exactly 0.02 because
        # 0.6117 - 0.5917 becomes 0.020000000000000018 in binary floating point.
        return difference <= tolerance or math.isclose(
            difference, tolerance, rel_tol=0, abs_tol=1e-12
        )

    left, right = (
        SystemOneResponse.model_validate(value).answers for value in (reference, candidate)
    )
    if left.keys() != right.keys():
        return {"passed": False, "error": "Different question names"}
    rows = {}
    for key in left:
        a, b = left[key].model_dump(), right[key].model_dump()
        row = {"passed": True, "reference": a, "candidate": b}
        if a["type"] != b["type"]:
            row.update(passed=False, error="Different answer types")
        elif a["type"] == "noul":
            difference = abs(a["noul"] - b["noul"])
            row.update(max_probability_difference=difference, passed=within(difference))
        else:
            if a["probabilities"].keys() != b["probabilities"].keys():
                row.update(passed=False, error="Different candidate labels")
            else:
                difference = max(
                    abs(a["probabilities"][k] - b["probabilities"][k]) for k in a["probabilities"]
                )
                confidence_difference = abs(a["confidence"] - b["confidence"])
                row.update(
                    max_probability_difference=difference,
                    confidence_difference=confidence_difference,
                    passed=within(difference) and within(confidence_difference),
                )
                if a["type"] == "choice":
                    row["same_choice"] = a["choice"] == b["choice"]
                    row["passed"] &= row["same_choice"]
                else:
                    row["score_difference"] = abs(a["score"] - b["score"])
                    row["passed"] &= within(row["score_difference"]) and a["legend"] == b["legend"]
        rows[key] = row
    return {"passed": all(r["passed"] for r in rows.values()), "questions": rows}


async def post(client, url, case):
    started = perf_counter()
    response = await client.post(url.rstrip("/") + "/v1/systemone", json=case["request"])
    response.raise_for_status()
    payload = response.json()
    validate_response(case["request"], payload)
    return payload, (perf_counter() - started) * 1000


def validate_response(request: dict, payload: dict) -> None:
    """A 200 response alone is not a successful decision."""
    query = SystemOneRequest.model_validate(request)
    response = SystemOneResponse.model_validate(payload)
    if response.answers.keys() != query.questions.keys():
        raise ValueError("Response is missing questions or contains unexpected questions")
    for name, question in query.questions.items():
        answer = response.answers[name]
        if answer.type != question.type:
            raise ValueError(f"Wrong answer type for {name}")
        if answer.type == "noul":
            values = [answer.noul]
        else:
            values = [*answer.probabilities.values(), answer.confidence]
            if not answer.probabilities:
                raise ValueError(f"Missing probabilities for {name}")
            # Upstream rounds each probability to four decimal places.
            tolerance = len(answer.probabilities) * 0.00005 + 1e-6
            if abs(sum(answer.probabilities.values()) - 1) > tolerance:
                raise ValueError(f"Probabilities do not sum to one for {name}")
            if answer.type == "choice" and (
                answer.choice not in question.criteria
                or answer.probabilities.keys() != question.criteria.keys()
            ):
                raise ValueError(f"Invalid candidate labels for {name}")
            if answer.type == "score" and not math.isfinite(answer.score):
                raise ValueError(f"Invalid score for {name}")
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError(f"Invalid probability/confidence for {name}")


async def validate_endpoints(
    reference_url: str,
    candidate_url: str,
    corpus: Path,
    output: Path,
    tolerance: float,
    concurrency: int,
) -> dict:
    if not 0 <= tolerance < 1 or concurrency < 1:
        raise ValueError("Invalid tolerance or concurrency")
    cases = load_cases(corpus)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    slots = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
        # Obtain baseline sequentially; it may serialize inference internally.
        references = [(await post(client, reference_url, case))[0] for case in cases]

        async def check(index, case):
            async with slots:
                try:
                    candidate, elapsed = await post(client, candidate_url, case)
                    result = compare_responses(references[index], candidate, tolerance=tolerance)
                    return {
                        "id": case["id"],
                        **result,
                        "candidate_client_ms": elapsed,
                        "reference_response": references[index],
                        "candidate_response": candidate,
                    }
                except Exception as exc:
                    return {"id": case["id"], "passed": False, "error": str(exc)}

        rows = await asyncio.gather(*(check(i, c) for i, c in enumerate(cases)))
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "reference_url": reference_url,
        "candidate_url": candidate_url,
        "tolerance": tolerance,
        "candidate_concurrency": concurrency,
        "cases": len(rows),
        "passed": all(r["passed"] for r in rows),
        "passed_cases": sum(r["passed"] for r in rows),
        "results": rows,
    }
    (output / "validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def percentile(values, quantile):
    return sorted(values)[max(0, math.ceil(quantile * len(values)) - 1)] if values else None


async def benchmark(
    url: str,
    corpus: Path,
    output: Path,
    levels: list[int],
    requests: int,
) -> dict:
    cases = load_cases(corpus)
    if (
        not levels
        or any(c < 1 or c > requests for c in levels)
        or len(set(levels)) != len(levels)
        or requests % len(cases)
    ):
        raise ValueError(
            "Use distinct positive levels <= requests; requests must be a corpus multiple"
        )
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "url": url,
        "method": "fixed workers, closed-loop HTTP load, nearest-rank success-only percentiles",
        "warmups_excluded": 0,
        "warmup_policy": "Before each level: max(corpus size, 2 * concurrency) requests at that concurrency",
        "cells": [],
    }
    async with AsyncExitStack() as stack:
        client = await stack.enter_async_context(
            httpx.AsyncClient(
                timeout=60,
                trust_env=False,
                limits=httpx.Limits(
                    max_connections=max(levels), max_keepalive_connections=max(levels)
                ),
            )
        )
        raw = stack.enter_context((output / "requests.jsonl").open("w"))
        warmups = stack.enter_context((output / "warmups.jsonl").open("w"))
        for concurrency in levels:
            warm_slots = asyncio.Semaphore(concurrency)

            async def warm(index, slots=warm_slots, level=concurrency):
                async with slots:
                    case = cases[index % len(cases)]
                    payload, elapsed = await post(client, url, case)
                    warmups.write(
                        json.dumps(
                            {
                                "concurrency": level,
                                "index": index,
                                "case": case["id"],
                                "client_ms": elapsed,
                                "response": payload,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            warm_count = max(len(cases), 2 * concurrency)
            await asyncio.gather(*(warm(i) for i in range(warm_count)))
            warmups.flush()
            report["warmups_excluded"] += warm_count
            next_index, rows = 0, []

            async def worker(concurrency=concurrency, rows=rows):
                nonlocal next_index
                while next_index < requests:
                    index = next_index
                    next_index += 1
                    case = cases[index % len(cases)]
                    started = perf_counter()
                    row = {
                        "concurrency": concurrency,
                        "index": index,
                        "case": case["id"],
                        "started_at": datetime.now(UTC).isoformat(),
                    }
                    try:
                        payload, elapsed = await post(client, url, case)
                        row.update(success=True, client_ms=elapsed, response=payload)
                    except Exception as exc:
                        row.update(
                            success=False,
                            client_ms=(perf_counter() - started) * 1000,
                            error=str(exc) or repr(exc),
                            error_type=type(exc).__name__,
                        )
                    rows.append(row)
                    raw.write(json.dumps(row, ensure_ascii=False) + "\n")

            started_at = datetime.now(UTC).isoformat()
            started = perf_counter()
            await asyncio.gather(*(worker() for _ in range(concurrency)))
            elapsed = perf_counter() - started
            latencies = [r["client_ms"] for r in rows if r["success"]]
            report["cells"].append(
                {
                    "concurrency": concurrency,
                    "started_at": started_at,
                    "finished_at": datetime.now(UTC).isoformat(),
                    "warmups_excluded": warm_count,
                    "requests": requests,
                    "successes": len(latencies),
                    "failures": requests - len(latencies),
                    "wall_seconds": elapsed,
                    "successful_rps": len(latencies) / elapsed,
                    "mean_ms": sum(latencies) / len(latencies) if latencies else None,
                    "p50_ms": percentile(latencies, 0.5),
                    "p95_ms": percentile(latencies, 0.95),
                }
            )
            raw.flush()
            (output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
