# GPU concurrency benchmark

English | [简体中文](GPU_CONCURRENCY.zh-CN.md)

This report preserves the original benchmark. A later [public API release](DEPLOYED.md) adds authentication, conservative admission limits, request tracing, and an HTTP disconnect-watcher fix, with separate HTTPS verification. Its measurements must not be combined with this benchmark.

On 2026-10-13, we compared the original deployed Decider image with this project's vLLM adapter. Both used the same v19 model and inputs, with exclusive use of the same NVIDIA L4 GPU during their respective runs.

**vLLM delivers measured concurrency gains. Start with concurrency 4–8 under the current settings.** At concurrency 8, throughput was about 2.36 times the original service's, and P95 latency fell by about 56%. Increasing concurrency to 16 or 32 did not improve throughput further. Five requests timed out after approximately 60 seconds across the higher-concurrency tests; their root cause remains unresolved.

## Comparison results

The table compares the second vLLM run, using unchanged inference settings, with the original baseline. Each level contains 80 requests, excluding startup and warmup. This run completed without failures. Failures from the initial and extended runs are reported below and remain in the raw evidence.

| Concurrency | Original RPS | vLLM RPS | Original P50 / P95 ms | vLLM P50 / P95 ms |
|---:|---:|---:|---:|---:|
| 1 | 14.63 | 22.55 | 68.4 / 69.1 | 44.3 / 44.8 |
| 2 | 14.82 | 24.29 | 134.9 / 135.5 | 82.2 / 82.8 |
| 4 | 14.81 | 33.31 | 269.9 / 271.0 | 118.8 / 127.5 |
| 8 | 14.76 | 34.91 | 541.5 / 542.8 | 228.5 / 237.6 |
| 16 | 14.75 | 30.15 | 1082.0 / 1084.4 | 526.7 / 549.9 |
| 32 | 14.73 | 30.64 | 2160.0 / 2168.4 | 1013.8 / 1094.6 |

![Repeat run with unchanged settings: throughput and latency](results/gpu-concurrency-2026-10-13/concurrency.png)

The original service sustains about 14.7 RPS; additional concurrency mainly increases serial queuing. vLLM reduces single-request latency from about 68 ms to 44 ms and uses batching to reduce queuing for concurrent requests. These gains include LoRA merging, different PyTorch / CUDA kernels, CPU thread settings, and scheduling, so they cannot be attributed entirely to batching.

`max_num_seqs=32` is a limit, not a guarantee that any 32 inputs can be processed together. The 8192-token batch budget also constrains scheduling. With these settings, throughput plateaus around concurrency 4–8, and additional requests continue to queue. RPS here means individual decision requests per second, not complete payment workflows per second.

## Timeouts and the extended run

In the initial vLLM run, two of 80 requests at concurrency 16 timed out. That level took 61.08 seconds overall, reducing successful throughput to **1.28 RPS**. All other levels completed successfully. A repeat run with the same settings completed all 480 requests, followed by an extended run with 400 requests at each of concurrency 4, 8, 16, and 32:

| Concurrency | Successful / total requests | Successful RPS | Mean ms | P50 ms | P95 ms | Timeouts |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 400 / 400 | 33.31 | 119.9 | 119.9 | 126.7 | 0 |
| 8 | 400 / 400 | 34.93 | 228.5 | 230.1 | 233.2 | 0 |
| 16 | 400 / 400 | 30.21 | 528.3 | 528.8 | 542.9 | 0 |
| 32 | 397 / 400 | 6.44 | 943.7 | 940.7 | 1040.3 | 3 |

**Latency percentiles include successful requests only. Timeouts still count as failures and contribute to the level's total elapsed time.** At concurrency 32, successful requests therefore have a P95 of about one second, while three roughly 60-second timeouts reduce successful throughput to 6.44 RPS. P95 alone would hide the problem at that level.

The three vLLM runs contain **2,560 measured requests: 2,555 successes and five timeouts**. Concurrency levels 4 and 8 each completed 560 requests without failure. Inference-engine settings remained unchanged across the three runs. The unresolved intermittent timeouts cannot be described as a fixed cold-start issue, and these measurements do not establish a production SLA.

For behavior, **all 2,555 successful responses selected the same candidates as the original service**, with probability and confidence differences within the 0.02 tolerance. The initial audit incorrectly rejected four confidence differences of exactly 0.02 because their floating-point representation was `0.020000000000000018`. The comparison was corrected and rerun; the original audit remains in `vllm-concurrency/behavior-audit-before-float-fix.json`. This correction did not change model answers, relax the tolerance, or turn timeouts into successes.

These results support further integration testing at concurrency 4–8. Before switching production traffic, investigate the timeouts and test longer inputs, sustained load, requests containing multiple questions, and the existing ECS backend's concurrency limits.

Evidence: [full initial comparison](results/gpu-concurrency-2026-10-13/table.md) · [repeat with unchanged settings](results/gpu-concurrency-2026-10-13/table-vllm-recheck.md) · [extended run](results/gpu-concurrency-2026-10-13/table-vllm-extended.md) · [verification manifest and per-request evidence](results/gpu-concurrency-2026-10-13/verification.json).

## Test conditions

| Item | Configuration |
|---|---|
| AWS | us-east-1; temporary On-Demand `g6.4xlarge` |
| GPU | NVIDIA L4, 24 GB class VRAM, driver 595.91.07 |
| Inference resources | Each service limited to CPU affinity `0–3` and 14 GiB RAM |
| Load generator | Separate container on the same host; CPUs `4–5`, 4 GiB RAM |
| Original service | Original production image pinned by digest; official Decider + PyTorch + FLA; serial execution |
| vLLM | 0.31.0, merged LoRA, BF16 backbone, original FP32 Pointer Head |
| vLLM scheduling | `max_num_seqs=32`, `max_num_batched_tokens=8192`, GPU memory utilization 0.7 |
| CPU threads | Original retains `OMP_NUM_THREADS=4`; vLLM uses 1, as recommended at startup |
| Other settings | Single GPU; CUDA Graphs, prefix caching, and chunked prefill disabled |
| Request path | `POST /v1/systemone` over loopback HTTP; includes input processing, queuing, inference, and response delivery |
| Concurrency levels | 1, 2, 4, 8, 16, 32 |
| Inputs | Four payment scenarios × five steps; 20 distinct single-question requests |
| Input length | 304–445 tokens; mean 373.9 tokens |
| Initial / repeat run | 80 requests per level, each input repeated four times; 480 measured requests per run |
| Extended run | Concurrency 4, 8, 16, 32; 400 requests per level, 1,600 total |
| Warmup | Before each level, `max(20, 2 × concurrency)` requests at that concurrency; 176 excluded for six-level runs and 136 for the four-level extended run |

`g6.xlarge` and `g6.2xlarge` lacked capacity in the availability zones attempted. We therefore used a `g6.4xlarge`, which also has a single L4, and restricted inference-container resources. The host was still a g6.4xlarge, so these are not direct g6.xlarge measurements. The client used additional CPUs; when moving to a host with only four vCPUs, generate load from another machine and validate again.

The two implementations were measured sequentially to avoid GPU contention. Model startup, image construction, and warmups are excluded. Requests did not pass through CloudFront, the ECS API, DynamoDB, or the public internet, so these results are not directly comparable with end-to-end latency in the original demo website.

## Statistics and behavior checks

A fixed number of client workers continuously submit requests, each sending its next request after the previous one finishes. This is the load-generation method, separate from the payment demo's autonomous business workflow. RPS is successful requests divided by the total elapsed time for that level. P50 and P95 use nearest-rank percentiles of successful-request latency. P95 is the latency at or below which 95% of successful requests completed.

First, we captured original-service responses for the 20 scenario requests and one mixed request containing `choice`, `noul`, and `score`: 21 requests and 23 judgments. These were compared with actual vLLM responses. Every measured benchmark response was then matched by case ID against the captured original response:

- `choice` must select the same candidate ID.
- Absolute differences in probabilities, confidence, `noul`, and `score` must not exceed 0.02.
- An HTTP 200 must also pass checks for complete questions, valid candidate IDs, and valid probabilities.

Repeated requests test for cross-request mix-ups or numerical changes under concurrency; they do not add 480 independent business examples. Matching the original service does not establish business correctness. The engine adaptation did not train new weights or correct the original model's errors.

## Reproduction

See the [README](../README.md) for model export and image construction. The measured vLLM configuration was:

```bash
docker run --rm --gpus all --shm-size=2g \
  --cpuset-cpus=0-3 --memory=14g \
  -p 127.0.0.1:8100:8100 \
  -v "$PWD/models/decider-vllm:/models/decider-vllm:ro" \
  strands-decider-vllm:0.1.0 serve \
  --model /models/decider-vllm --host 0.0.0.0 --port 8100 \
  --gpu-memory-utilization 0.7 --max-num-seqs 32 \
  --max-num-batched-tokens 8192
```

Use the pinned production model image for the original service. `scripts/reference_server.py` is a behavior reference only; its different kernel stack cannot replace this performance baseline. Start, capture, and stop each service separately to avoid simultaneous GPU use:

```bash
python scripts/gpu_experiment.py capture \
  --url http://127.0.0.1:8099 \
  --corpus examples/payment-corpus.json --mixed examples/payment.json \
  --output artifacts/reference-capture.json

decider-vllm benchmark --url http://127.0.0.1:8099 \
  --corpus examples/payment-corpus.json --levels 1 2 4 8 16 32 \
  --requests 80 --output artifacts/reference-concurrency

python scripts/gpu_experiment.py audit \
  --reference artifacts/reference-capture.json \
  --requests artifacts/reference-concurrency/requests.jsonl \
  --output artifacts/reference-concurrency/behavior-audit.json

# Stop the original service, start vLLM, and use new output paths.
python scripts/gpu_experiment.py capture \
  --url http://127.0.0.1:8100 \
  --corpus examples/payment-corpus.json --mixed examples/payment.json \
  --output artifacts/vllm-capture.json

python scripts/gpu_experiment.py compare \
  --reference artifacts/reference-capture.json \
  --candidate artifacts/vllm-capture.json --output artifacts/gpu-parity.json

decider-vllm benchmark --url http://127.0.0.1:8100 \
  --corpus examples/payment-corpus.json --levels 1 2 4 8 16 32 \
  --requests 80 --output artifacts/vllm-concurrency

python scripts/gpu_experiment.py audit \
  --reference artifacts/reference-capture.json \
  --requests artifacts/vllm-concurrency/requests.jsonl \
  --output artifacts/vllm-concurrency/behavior-audit.json
```

The capture script separately records 21 additional warmup requests. Performance statistics use only `benchmark.json` and the measured `requests.jsonl`. `scripts/report_gpu_concurrency.py` recomputes statistics from per-request evidence and generates charts and SHA-256 manifests.

## Resource cleanup

At 08:27 UTC on 2026-10-13, the temporary EC2 instance was confirmed terminated. Its EBS volume, temporary S3 bucket, IAM role / instance profile, and security group had all been deleted. Results were downloaded to this project. At the end of that test batch, the original production g6.xlarge remained running, the demo website returned HTTP 200, and no production traffic switch or GitHub push had been performed as part of the benchmark.

See [cleanup.json](results/gpu-concurrency-2026-10-13/cleanup.json) for cleanup verification and [local-checks.json](results/gpu-concurrency-2026-10-13/local-checks.json) for the 26 passing local tests, code checks, and source hashes.
