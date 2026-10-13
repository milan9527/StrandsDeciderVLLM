import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import boto3

ROOT = Path("/home/ec2-user/ai/StrandsDeciderVLLM")
PRIVATE = ROOT / ".deployment/batchtokens-20261013"
OUTPUT = PRIVATE / "results"
state = json.loads((ROOT / ".deployment/state.json").read_text())
ssm = boto3.client("ssm", region_name="us-east-1")
OUTPUT.mkdir(exist_ok=False)
operations = []


def command(script, label, timeout=900):
    result = ssm.send_command(
        InstanceIds=[state["instance"]],
        DocumentName="AWS-RunShellScript",
        Parameters={
            "commands": [script.replace("__BUCKET__", state["bucket"])],
            "executionTimeout": [str(timeout)],
        },
        OutputS3BucketName=state["bucket"],
        OutputS3KeyPrefix="output/ssm",
        Comment="Decider batch token budget experiment: " + label,
    )
    identifier = result["Command"]["CommandId"]
    operations.append({"id": identifier, "label": label, "started": datetime.now(UTC).isoformat()})
    print("SSM", label, identifier, flush=True)
    while True:
        time.sleep(5)
        try:
            result = ssm.get_command_invocation(CommandId=identifier, InstanceId=state["instance"])
        except ssm.exceptions.InvocationDoesNotExist:
            continue
        status = result["Status"]
        if status in ("Pending", "InProgress", "Delayed"):
            continue
        operations[-1]["status"] = status
        (PRIVATE / "operations.json").write_text(json.dumps(operations, indent=2) + "\n")
        print(status, result.get("StandardOutputContent", "")[-1200:],
              result.get("StandardErrorContent", "")[-800:], flush=True)
        return status == "Success"


def benchmark(tokens, levels, label=None):
    name = label or f"batch-{tokens}"
    print("Starting", name, levels, flush=True)
    result = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"), "scripts/benchmark_service.py",
            "--credentials", ".deployment/api-credentials.json",
            "--output", str(OUTPUT / name), "--label", name,
            "--engine-seqs", "128", "--batch-tokens", str(tokens),
            "--http-limit", "256", "--question-limit", "256",
            "--requests-per-minute", "120000", "--burst", "512",
            "--requests", "1280", "--levels", *map(str, levels),
        ],
        cwd=ROOT, check=False,
    )
    report_file = OUTPUT / name / "benchmark.json"
    report = json.loads(report_file.read_text()) if report_file.exists() else {}
    complete = len(report.get("cells", [])) == len(levels)
    errors = sum(c["failures"] for c in report.get("cells", []))
    choices = sum(c["changed_choices"] for c in report.get("cells", []))
    print("Finished", name, "exit", result.returncode, "complete", complete,
          "HTTP failures", errors, "changed choices", choices, flush=True)
    # Preserve numeric deviations separately; stop on capacity/choice failures.
    if not complete or errors or choices or report.get("warmup_failures"):
        raise RuntimeError("Unhealthy or incomplete test: " + name)


armed = False
try:
    if not command((PRIVATE / "setup.sh").read_text(), "setup"):
        raise RuntimeError("Setup failed")
    armed = True
    for tokens in (8192, 16384, 32768):
        if tokens != 8192:
            script = (PRIVATE / "configure.sh").read_text().replace("__TOKENS__", str(tokens))
            if not command(script, f"configure-{tokens}"):
                (OUTPUT / f"startup-failed-{tokens}.json").write_text(json.dumps({
                    "budget": tokens, "startup_failed": True,
                    "at": datetime.now(UTC).isoformat(),
                }, indent=2) + "\n")
                break
        benchmark(tokens, [8, 16, 32, 64, 128])
    else:
        # Verify that any high-load benefit repeats on the largest tested budget.
        benchmark(32768, [32, 64, 128], "batch-32768-repeat")
finally:
    if armed:
        restored = command(
            "#!/bin/bash\nset -euo pipefail\n/opt/decider-batchtokens-20261013/restore.sh\n",
            "restore-8192",
        )
        if not restored:
            raise RuntimeError("Restore did not complete; inspect immediately")
    s3 = boto3.client("s3", region_name="us-east-1")
    destination = OUTPUT / "host"
    destination.mkdir(exist_ok=True)
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=state["bucket"], Prefix="output/batchtokens-20261013/"
    ):
        for item in page.get("Contents", []):
            name = item["Key"].removeprefix("output/batchtokens-20261013/")
            if "/" not in name:
                s3.download_file(state["bucket"], item["Key"], str(destination / name))
    print("Finalization complete; host evidence downloaded.", flush=True)
