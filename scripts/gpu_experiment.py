"""Capture actual GPU endpoint responses, compare them, and audit benchmark choices."""

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

from decider_vllm.evaluation import compare_responses, load_cases, post, validate_response


async def capture(args):
    cases = load_cases(args.corpus)
    cases.append({"id": "mixed-primitives", "request": json.loads(args.mixed.read_text())})
    if args.output.exists():
        raise FileExistsError(args.output)
    results = []
    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
        for case in cases:
            await post(client, args.url, case)
        for case in cases:
            response, latency = await post(client, args.url, case)
            results.append(
                {
                    "id": case["id"],
                    "request": case["request"],
                    "response": response,
                    "client_ms": latency,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "created_at": datetime.now(UTC).isoformat(),
                "url": args.url,
                "warmup_requests_excluded": len(cases),
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(f"Captured {len(results)} real endpoint responses")


def compare(args):
    reference = json.loads(args.reference.read_text())
    candidate = json.loads(args.candidate.read_text())
    references = {row["id"]: row for row in reference["results"]}
    assert set(references) == {r["id"] for r in candidate["results"]}
    rows = []
    for row in candidate["results"]:
        expected = references[row["id"]]
        assert expected["request"] == row["request"]
        rows.append(
            {
                "id": row["id"],
                **compare_responses(
                    expected["response"], row["response"], tolerance=args.tolerance
                ),
            }
        )
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "tolerance": args.tolerance,
        "passed": all(row["passed"] for row in rows),
        "passed_cases": sum(row["passed"] for row in rows),
        "cases": len(rows),
        "results": rows,
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}))
    if not report["passed"]:
        raise SystemExit(1)


def audit(args):
    reference = {row["id"]: row for row in json.loads(args.reference.read_text())["results"]}
    rows = []
    for line in args.requests.read_text().splitlines():
        row = json.loads(line)
        if not row["success"]:
            rows.append(
                {
                    "index": row["index"],
                    "concurrency": row["concurrency"],
                    "case": row["case"],
                    "passed": False,
                    "error": row["error"],
                }
            )
            continue
        baseline = reference[row["case"]]
        validate_response(baseline["request"], row["response"])
        rows.append(
            {
                "index": row["index"],
                "concurrency": row["concurrency"],
                "case": row["case"],
                **compare_responses(
                    baseline["response"], row["response"], tolerance=args.tolerance
                ),
            }
        )
    report = {
        "tolerance": args.tolerance,
        "requests": len(rows),
        "passed": all(row["passed"] for row in rows),
        "passed_requests": sum(row["passed"] for row in rows),
        "results": rows,
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}))
    if not report["passed"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--url", required=True)
    c.add_argument("--corpus", type=Path, required=True)
    c.add_argument("--mixed", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("compare")
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tolerance", type=float, default=0.02)
    a = sub.add_parser("audit")
    a.add_argument("--reference", type=Path, required=True)
    a.add_argument("--requests", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--tolerance", type=float, default=0.02)
    args = parser.parse_args()
    if args.action == "capture":
        asyncio.run(capture(args))
    else:
        globals()[args.action](args)


if __name__ == "__main__":
    main()
