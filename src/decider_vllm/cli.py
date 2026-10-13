import argparse
import asyncio
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Strands Decider v19 / vLLM adapter")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Merge local LoRA and package Pointer Head")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--base", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify-artifacts")
    verify.add_argument("--model", type=Path, required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--model", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8100)
    serve.add_argument("--max-num-seqs", type=int, default=16)
    serve.add_argument("--max-num-batched-tokens", type=int, default=8192)
    serve.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    serve.add_argument("--max-inflight-questions", type=int, default=64)
    serve.add_argument("--max-pending-requests", type=int, default=32)
    serve.add_argument("--timeout-seconds", type=float, default=60)
    serve.add_argument("--allow-truncation", action="store_true")
    validation = commands.add_parser("validate", help="Compare real original and vLLM endpoints")
    validation.add_argument("--reference-url", required=True)
    validation.add_argument("--candidate-url", required=True)
    validation.add_argument("--corpus", type=Path, required=True)
    validation.add_argument("--output", type=Path, required=True)
    validation.add_argument("--tolerance", type=float, default=0.02)
    validation.add_argument("--concurrency", type=int, default=8)
    bench = commands.add_parser("benchmark")
    bench.add_argument("--url", required=True)
    bench.add_argument("--corpus", type=Path, required=True)
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    bench.add_argument("--requests", type=int, default=80)
    args = parser.parse_args()
    if args.command == "export":
        from .export import export_model

        report = export_model(args.checkpoint, args.base, args.output)
        print(json.dumps({"output": str(args.output), "export": report}, indent=2))
    elif args.command == "verify-artifacts":
        from .artifacts import verify_export

        print(json.dumps(verify_export(args.model)))
    elif args.command == "serve":
        import uvicorn

        from .backend import RuntimeSettings, create_backend
        from .server import create_app

        settings = RuntimeSettings(
            model=args.model,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_inflight_questions=args.max_inflight_questions,
            strict_window=not args.allow_truncation,
        )
        app = create_app(
            lambda: create_backend(settings),
            max_pending_requests=args.max_pending_requests,
            timeout_seconds=args.timeout_seconds,
        )
        uvicorn.run(app, host=args.host, port=args.port, workers=1, access_log=False)
    elif args.command == "validate":
        from .evaluation import validate_endpoints

        report = asyncio.run(
            validate_endpoints(
                args.reference_url,
                args.candidate_url,
                args.corpus,
                args.output,
                args.tolerance,
                args.concurrency,
            )
        )
        print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
        if not report["passed"]:
            raise SystemExit(1)
    else:
        from .evaluation import benchmark

        report = asyncio.run(
            benchmark(
                args.url,
                args.corpus,
                args.output,
                args.levels,
                args.requests,
            )
        )
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
