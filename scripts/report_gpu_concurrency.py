"""Verify raw measurements and write a compact, reproducible GPU comparison report."""

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path


def verify_run(path):
    report = json.loads((path / "benchmark.json").read_text())
    rows = [json.loads(line) for line in (path / "requests.jsonl").read_text().splitlines()]
    warmups = [json.loads(line) for line in (path / "warmups.jsonl").read_text().splitlines()]
    assert len(rows) == sum(cell["requests"] for cell in report["cells"])
    for cell in report["cells"]:
        selected = [r for r in rows if r["concurrency"] == cell["concurrency"]]
        assert len(selected) == cell["requests"]
        assert len({r["index"] for r in selected}) == cell["requests"]
        assert sum(not r["success"] for r in selected) == cell["failures"]
        assert (
            sum(r["concurrency"] == cell["concurrency"] for r in warmups)
            == cell["warmups_excluded"]
        )
        assert len(Counter(r["case"] for r in selected)) == 20
        assert set(Counter(r["case"] for r in selected).values()) == {cell["requests"] // 20}
        values = sorted(r["client_ms"] for r in selected if r["success"])
        assert len(values) == cell["successes"]
        if values:
            assert math.isclose(sum(values) / len(values), cell["mean_ms"], abs_tol=1e-8)
            for q, key in [(0.5, "p50_ms"), (0.95, "p95_ms")]:
                assert values[math.ceil(q * len(values)) - 1] == cell[key]
        assert math.isclose(
            len(values) / cell["wall_seconds"], cell["successful_rps"], abs_tol=1e-8
        )
    assert len(warmups) == report["warmups_excluded"]
    audit = json.loads((path / "behavior-audit.json").read_text())
    assert audit["requests"] == len(rows)
    assert audit["passed_requests"] == sum(row["passed"] for row in audit["results"])
    return report, audit


def behavior_summary(audit):
    questions = [
        question for row in audit["results"] for question in row.get("questions", {}).values()
    ]
    return {
        "passed": audit["passed"],
        "passed_requests": audit["passed_requests"],
        "same_choices": sum(q.get("same_choice", False) for q in questions),
        "choice_judgments": sum("same_choice" in q for q in questions),
        "max_probability_difference": max(
            (q.get("max_probability_difference", 0) for q in questions), default=None
        ),
        "max_confidence_difference": max(
            (q.get("confidence_difference", 0) for q in questions), default=None
        ),
    }


def write_bilingual_table(path, table, chinese_header):
    chinese_path = path.with_name(path.stem + ".zh-CN.md")
    path.write_text(f"English | [简体中文]({chinese_path.name})\n\n" + "\n".join(table) + "\n")
    chinese_path.write_text(
        f"[English]({path.name}) | 简体中文\n\n" + "\n".join([chinese_header, *table[1:]]) + "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline, baseline_audit = verify_run(args.results / "reference-concurrency")
    candidate, candidate_audit = verify_run(args.results / "vllm-concurrency")
    extras = {
        name: verify_run(args.results / name)
        for name in ("vllm-recheck", "vllm-extended")
        if (args.results / name).exists()
    }
    parity = json.loads((args.results / "gpu-parity.json").read_text())
    assert [c["concurrency"] for c in baseline["cells"]] == [
        c["concurrency"] for c in candidate["cells"]
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    for directory in ("reference-concurrency", "vllm-concurrency"):
        shutil.copytree(args.results / directory, args.output / directory)
    for directory in extras:
        shutil.copytree(args.results / directory, args.output / directory)
    for name in (
        "reference-capture.json",
        "vllm-capture.json",
        "gpu-parity.json",
        "gpu-hardware.csv",
        "cpu-hardware.json",
        "reference-health.json",
        "vllm-health.json",
        "reference-gpu.csv",
        "vllm-gpu.csv",
        "runtime-metadata.json",
        "reference-runtime.json",
        "installed-code-sha256.json",
        "experiment-manifest.json",
        "vllm-final-health.json",
    ):
        if (args.results / name).exists():
            shutil.copyfile(args.results / name, args.output / name)
    table = [
        "| Concurrency | Original RPS | vLLM RPS | Original P50 / P95 ms | vLLM P50 / P95 ms | Failures: original / vLLM |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for original, vllm in zip(baseline["cells"], candidate["cells"], strict=True):
        table.append(
            f"| {original['concurrency']} | {original['successful_rps']:.2f} | "
            f"{vllm['successful_rps']:.2f} | {original['p50_ms']:.1f} / "
            f"{original['p95_ms']:.1f} | {vllm['p50_ms']:.1f} / {vllm['p95_ms']:.1f} | "
            f"{original['failures']} / {vllm['failures']} |"
        )
    metadata = {
        "reference_requests": baseline_audit["requests"],
        "vllm_requests": candidate_audit["requests"],
        "reference_behavior_passed": baseline_audit["passed_requests"],
        "vllm_behavior_passed": candidate_audit["passed_requests"],
        "reference_behavior": behavior_summary(baseline_audit),
        "vllm_behavior": behavior_summary(candidate_audit),
        "additional_runs": {name: behavior_summary(audit) for name, (_, audit) in extras.items()},
        "gpu_parity": {
            "passed": parity["passed"],
            "passed_cases": parity["passed_cases"],
            "cases": parity["cases"],
        },
        "sha256": {},
    }
    for path in sorted(args.output.rglob("*")):
        if path.is_file():
            metadata["sha256"][str(path.relative_to(args.output))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    (args.output / "verification.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_bilingual_table(
        args.output / "table.md",
        table,
        "| 并发 | 原版 RPS | vLLM RPS | 原版 P50 / P95 ms | vLLM P50 / P95 ms | 失败：原版 / vLLM |",
    )
    print("\n".join(table))
    for name, (report, _) in extras.items():
        extra_table = [
            "| Concurrency | Requests | Successful RPS | Mean ms | P50 ms | P95 ms | Failures |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for cell in report["cells"]:
            extra_table.append(
                f"| {cell['concurrency']} | {cell['requests']} | "
                f"{cell['successful_rps']:.2f} | {cell['mean_ms']:.1f} | "
                f"{cell['p50_ms']:.1f} | {cell['p95_ms']:.1f} | {cell['failures']} |"
            )
        write_bilingual_table(
            args.output / f"table-{name}.md",
            extra_table,
            "| 并发 | 请求数 | 成功 RPS | 平均 ms | P50 ms | P95 ms | 失败 |",
        )
        print(name + "\n" + "\n".join(extra_table))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def plot(candidate_report, filename, suffix):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        plot_lines(axes, candidate_report, baseline)
        axes[0].legend(frameon=False)
        fig.suptitle(f"Strands Decider v19 · L4 · 4 vCPUs per service · {suffix}")
        for extension in ("png", "svg"):
            fig.savefig(args.output / f"{filename}.{extension}", dpi=170)
        plt.close(fig)

    failures = sum(cell["failures"] for cell in candidate["cells"])
    plot(candidate, "concurrency-initial", f"initial run ({failures} failures)")
    if "vllm-recheck" in extras:
        plot(extras["vllm-recheck"][0], "concurrency", "repeat, same engine settings")


def plot_lines(axes, candidate, baseline):
    for report, label, color in [
        (baseline, "Original (serial)", "#475569"),
        (candidate, "vLLM adapter", "#0077b6"),
    ]:
        x = [cell["concurrency"] for cell in report["cells"]]
        for axis, metric, title, unit in zip(
            axes,
            ("successful_rps", "p50_ms", "p95_ms"),
            ("Successful throughput", "Median latency", "P95 latency"),
            ("requests / second", "milliseconds", "milliseconds"),
            strict=True,
        ):
            axis.plot(x, [cell[metric] for cell in report["cells"]], "o-", label=label, color=color)
            axis.set_xscale("log", base=2)
            axis.set_xticks(x, labels=[str(value) for value in x])
            axis.set_xlabel("Concurrent requests")
            axis.set_ylabel(unit)
            axis.set_title(title)
            axis.grid(alpha=0.2)


if __name__ == "__main__":
    main()
