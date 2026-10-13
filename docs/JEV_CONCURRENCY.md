# Decider vLLM versus Jev: concurrency performance

English | [简体中文](JEV_CONCURRENCY.zh-CN.md)

2026-10-13. Keep Decider's batch token budget at **8192** and compare concurrency 1–128 against the direct Jev API, using the same client host and the same 20 business inputs.

**The Decider column uses the vLLM adapter, not the original serial implementation.** Its single L4 GPU runs on an On-Demand EC2 `g6.xlarge`. This report measures the independent public inference service; the companion payment demo has not switched to it. Earlier Decider-only experiments are summarized separately in [Decider vLLM performance](VLLM_PERFORMANCE.md).

Decider: **10,239/10,240** measured requests succeeded. Jev: **10,240/10,240**. Excluded warmups: 560 / 560, with 0 / 0 failures. No stop conditions were triggered; all planned levels completed.

## Comparison

At concurrency 1–4, Decider has lower mean and P95 latency. At 8–32, Jev has higher throughput and lower latency. Jev degrades at 64 and 128; at 64 its mean latency remains below Decider's but its P95 is higher. Evaluate throughput and tail latency together instead of selecting concurrency from peak throughput alone.

| Concurrency | Decider RPS | Mean ms | P95 ms | Jev RPS | Mean ms | P95 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 18.01 | 55.4 | 59.4 | 7.90 | 126.5 | 160.2 |
| 2 | 22.53 | 88.6 | 91.0 | 15.39 | 129.8 | 171.2 |
| 4 | 32.76 | 121.8 | 134.6 | 30.27 | 131.7 | 166.7 |
| 8 | 35.74 | 223.4 | 237.0 | 61.83 | 128.9 | 166.9 |
| 16 | 33.61 | 474.7 | 539.4 | 127.45 | 123.5 | 156.9 |
| 32 | 30.61 | 1039.2 | 1073.1 | 240.94 | 130.0 | 172.6 |
| 64 | 30.19 | 2081.0 | 2128.5 | 68.15 | 921.2 | 4059.0 |
| 128 | 30.59 | 3981.9 | 11236.8 | 52.56 | 1743.5 | 5502.5 |

RPS counts successfully completed decision requests; each request contains one choice question. Means and percentiles use valid successful responses only; total cell wall time includes failures. Each cell plans 1,280 requests; actual counts are below.

## Complete latency and success counts

| Model | Concurrency | Success/attempts | RPS | Mean ms | P50 ms | P95 ms | P99 ms | Max ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decider | 1 | 1280/1280 | 18.01 | 55.4 | 54.7 | 59.4 | 64.3 | 258.6 |
| decider | 2 | 1280/1280 | 22.53 | 88.6 | 88.5 | 91.0 | 92.6 | 111.0 |
| decider | 4 | 1280/1280 | 32.76 | 121.8 | 119.8 | 134.6 | 136.6 | 299.0 |
| decider | 8 | 1280/1280 | 35.74 | 223.4 | 222.5 | 237.0 | 240.6 | 261.9 |
| decider | 16 | 1280/1280 | 33.61 | 474.7 | 469.6 | 539.4 | 556.4 | 564.9 |
| decider | 32 | 1280/1280 | 30.61 | 1039.2 | 1041.0 | 1073.1 | 1085.4 | 1132.3 |
| decider | 64 | 1279/1280 | 30.19 | 2081.0 | 2106.3 | 2128.5 | 2785.0 | 3048.1 |
| decider | 128 | 1280/1280 | 30.59 | 3981.9 | 2816.2 | 11236.8 | 16226.0 | 24435.3 |
| jev | 1 | 1280/1280 | 7.90 | 126.5 | 120.7 | 160.2 | 227.7 | 433.8 |
| jev | 2 | 1280/1280 | 15.39 | 129.8 | 124.2 | 171.2 | 223.1 | 395.8 |
| jev | 4 | 1280/1280 | 30.27 | 131.7 | 126.5 | 166.7 | 246.7 | 491.0 |
| jev | 8 | 1280/1280 | 61.83 | 128.9 | 123.6 | 166.9 | 230.4 | 443.4 |
| jev | 16 | 1280/1280 | 127.45 | 123.5 | 117.6 | 156.9 | 189.9 | 283.4 |
| jev | 32 | 1280/1280 | 240.94 | 130.0 | 124.4 | 172.6 | 212.8 | 345.7 |
| jev | 64 | 1280/1280 | 68.15 | 921.2 | 209.7 | 4059.0 | 6696.4 | 10942.5 |
| jev | 128 | 1280/1280 | 52.56 | 1743.5 | 1165.1 | 5502.5 | 7845.4 | 20466.5 |

## Longer Jev repeat

Plan 6,400 requests each at concurrency 32 and 128, repeating each input 320 times. These results remain separate and do not replace the sweep. See the repeat [manifest](results/jev-vllm-2026-10-13/jev-repeat-32-128/manifest.json).

| Jev concurrency | Success/attempts | Measured seconds | RPS | Mean ms | P95 ms | P99 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 6400/6400 | 26.10 | 245.21 | 129.9 | 174.8 | 233.8 |
| 128 | 6400/6400 | 121.89 | 52.51 | 2412.9 | 7307.3 | 11280.4 |

## Failures and timing boundaries

The Decider sweep had one HTTP 502 at concurrency 64 (`refund/verify`, index 1271, client latency 48.5 ms), with no backend request ID. Its exact source was not isolated, and no retry replaced this record.

At concurrency 128, ten Decider `underpaid/approve` responses changed confidence from the original reference's 0.5917 to 0.6119: a 0.0202 difference exceeds 0.02. The choice remains `approved`; maximum candidate probability difference is 0.0151. Numeric failures are retained without relaxing tolerance. These HTTP successes are counted separately from the HTTP 502.

At concurrency 128, Decider client mean / P95 are 3981.9 / 11236.8 ms; server-reported mean / P95 are 108.5 / 187.1 ms. Server timing starts after request parsing and includes preprocessing, engine queueing, and computation. Much of the client delay is outside that timer, but this run did not isolate client processing, connection pools, transport, CloudFront, or request parsing as its cause.

## Inputs and outputs

| Model | Input token range | Mean input | Output token range | Mean output |
|---|---:|---:|---:|---:|
| Decider | 304–445 | 373.9 | 1–1 | 1.0 |
| Jev | 575–744 | 657.0 | 33–79 | 55.0 |

Four payment scenarios × five independent steps form the 20 inputs. Each input repeats 64 times per cell in the same round-robin order. Business facts, questions, and candidate criteria are identical; Jev adds only `model=jev-latest`. Returned Jev version: `jev-1.13.0`.

Token usage is reported by each service; tokenizers, internal templates, and accounting need not match. Decider's `output_tokens=1` counts one decision question, not the textual size of the response JSON. Jev usage is retained as reported and does not establish its internal generation method.

Decider passed numeric comparison with the fixed original capture in **10,229/10,239** successful requests at the unchanged 0.02 tolerance. Among 10,239 same-input pairs where both services succeeded, 9,215 selected the same candidate. This is action agreement, not business accuracy. Jev and Decider probability calibrations are not treated as interchangeable.

## Paths and method

- Decider: client → CloudFront → private VPC origin → EC2 `g6.xlarge` / NVIDIA L4 → vLLM 0.31.0. Same v19 weights; batch budget 8192, engine sequence limit 128, HTTP/question admission 256, GPU memory target 0.7.
- Jev: same client → `https://api.typesafe.ai/v1/systemone`, bypassing the payment demo's two-slot ECS gate and Lambda bridge. The original demo configuration is unchanged.
- Test providers separately and alternate which runs first at each concurrency. Fixed workers immediately replace completed requests. Both clients reuse HTTPS/HTTP1.1 connections and support the highest concurrency; no retries.
- Each cell has `max(20, 2 × concurrency)` excluded warmups. Client timing runs from request submission to complete response receipt, including transport, queueing, and processing. Percentiles use nearest rank.
- HTTP timeout setting 40 seconds; connection timeout 8 seconds. Decider's server inference deadline is 25 seconds; Jev's internal deadline is unknown. Missing Jev `latency_ms` is not replaced with an invented compute time.
- Any HTTP 400/401/403/413/422/429, or five other failures in one phase, stops new submissions and higher levels for that provider while in-flight requests finish. Warmup failure also stops that provider. No key rotation, retries, or quota bypass.
- Fixed business state; no payment execution, propagation of prior answers, or model fine-tuning.

This measures **API service performance across two deployment paths**, not model kernels on equal hardware. Jev hardware, replica count, scheduling, and repeated-input caching policy are unknown. Short fixed-count closed-loop runs repeating these 20 inputs do not establish provider maximum capacity or a production SLA. Older Jev results through ECS/Lambda remain separate.

## Service state and reproduction

Decider runtime arguments match before and after the test, the GPU remains running, and final authenticated public inference passes. No production service code or concurrency settings were changed.

This command calls real services. Configure Decider to match the report first; the script records and tests requests without changing server settings.

```bash
uv run --no-sync python scripts/benchmark_jev.py \
  --credentials .deployment/api-credentials.json \
  --jev-key-file ~/ai/jev/jevkey.txt \
  --requests 1280 --levels 1 2 4 8 16 32 64 128 \
  --output artifacts/jev-comparison-new
```

For the longer repeat, use `--providers jev --levels 32 128 --requests 6400` with a new output directory.

[Summary](results/jev-vllm-2026-10-13/summary.json) · [Manifest](results/jev-vllm-2026-10-13/manifest.json) · [Requests](results/jev-vllm-2026-10-13/requests.jsonl) · [Warmups](results/jev-vllm-2026-10-13/warmups.jsonl) · [Inputs](results/jev-vllm-2026-10-13/corpus.json) · [Benchmark client](results/jev-vllm-2026-10-13/benchmark_jev.py) · [Runtime arguments](results/jev-vllm-2026-10-13/host/initial-command.json) · [Final API check](results/jev-vllm-2026-10-13/final-public-check.json) · [Checksums](results/jev-vllm-2026-10-13/sha256.json)
