# Concurrency 1–128: public GPU API capacity test

English | [简体中文](CONCURRENCY_128.zh-CN.md)

On **2026-10-13**, the service's original eight-request admission limit and 20 requests/second rate limit were raised for a real **1, 2, 4, 8, 16, 32, 64, 128** concurrency sweep.

**12,800/12,800 measured requests succeeded**, with no HTTP errors, rate-limit rejections, or inference timeouts. All responses preserved the original candidate choices. Numeric checks passed for **12,799/12,800** responses at the unchanged 0.02 tolerance. Another **944 warmup requests** succeeded and were excluded from the measurements.

Follow-up: the [8192 / 16384 / 32768 batch token experiment](BATCH_TOKENS.md) uses a fresh baseline and retains this earlier data separately.

## Full sweep

Every level used 1,280 requests, repeating each of the 20 payment inputs exactly 64 times. The hardware, model image, weights, and engine settings remained the same throughout the sweep.

| Concurrency | Success / requests | RPS | Mean ms | P50 ms | P95 ms | P99 ms | Request failures |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1280/1280 | 18.10 | 55.1 | 54.9 | 59.9 | 63.7 | 0 |
| 2 | 1280/1280 | 22.85 | 87.4 | 87.4 | 89.9 | 91.7 | 0 |
| 4 | 1280/1280 | 33.11 | 120.6 | 120.6 | 131.8 | 135.2 | 0 |
| 8 | 1280/1280 | 33.84 | 236.0 | 238.3 | 245.0 | 247.0 | 0 |
| 16 | 1280/1280 | 29.34 | 544.4 | 545.1 | 561.8 | 572.7 | 0 |
| 32 | 1280/1280 | 30.06 | 1057.9 | 1059.8 | 1089.9 | 1103.5 | 0 |
| 64 | 1280/1280 | 29.76 | 2112.7 | 2137.0 | 2178.7 | 2822.1 | 0 |
| 128 | 1280/1280 | 29.65 | 4111.9 | 4279.8 | 4986.1 | 7568.0 | 0 |

## High-concurrency repeat

Concurrency 64 and 128 were repeated with 1,280 requests each:

| Concurrency | Success / requests | RPS | Mean ms | P50 ms | P95 ms | P99 ms | Request failures |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1280/1280 | 29.65 | 2120.4 | 2142.8 | 2179.0 | 2838.1 | 0 |
| 128 | 1280/1280 | 29.65 | 4109.3 | 4297.5 | 5018.0 | 7469.6 | 0 |

Peak observed throughput in the sweep was **33.84 RPS at concurrency 8**. In the 128-concurrency repeat, P95 was **5.02 seconds**. Admitting more concurrent requests primarily increases waiting under this configuration; it does not provide proportional GPU throughput.

The GPU schedules requests in batches within an **8192-token batch budget**. These inputs have 304–445 tokens each, so 128 admitted requests are processed over multiple batches. Queuing is included in the reported end-to-end latency.

## Numeric comparison exception

One request in the initial 128-concurrency pass (`underpaid/approve`, index 27) exceeded the numeric tolerance. Both implementations selected `approved`. Confidence changed from 0.5917 to 0.6119, an absolute difference of **0.0202** against the **0.02** threshold; the largest candidate-probability difference was 0.0151. The extended repeat had no numeric failures.

This remains a failed numeric comparison in the raw evidence. The threshold was not relaxed and a later success did not replace the failure. This run did not isolate the cause of the numeric deviation. Applications that use confidence thresholds need to validate those separately from candidate-ID agreement.

## Configuration and method

| Setting | Value |
|---|---|
| Endpoint | https://d1e9udi8st2p3g.cloudfront.net |
| Hardware | On-Demand EC2 `g6.xlarge`, NVIDIA L4, 4 vCPUs; container memory limit 14 GiB |
| Model / runtime | Same Decider v19 image; vLLM `0.31.0`; no weight changes |
| HTTP admission / in-flight questions | 256 / 256 |
| vLLM `max_num_seqs` | 128 |
| Rate-limit threshold | 120,000/minute = 2,000/second; burst 512 |
| Context / batch token budget | 4096 per question / 8192 per batch |
| GPU memory fraction | 0.7 |
| Deadline / client timeout | 25 seconds / 40 seconds |
| Caching / execution | Prefix caching, chunked prefill, and CUDA Graphs disabled |

The rate threshold is a rejection threshold, not a claim that this GPU completes 2,000 requests/second. No admission or rate rejections occurred within the tested concurrency range.

The corpus contains four payment scenarios × five steps, with one choice question per request. The client runs outside the GPU host, uses HTTPS connection reuse, and maintains a fixed number of workers: each worker sends its next request after receiving the previous response. There are no retries. Before every level, `max(20, 2 × concurrency)` warmup requests run separately.

Throughput counts successful requests over the complete measured wall time; latency percentiles are nearest-rank values from successful responses. Client and server timings, HTTP statuses, unique request IDs, raw answers, and per-response comparisons are preserved. Maximum probability and confidence differences from the original were **0.0151** and **0.0202**.

This is a short-run capacity measurement on these inputs. It does not establish long-context performance, multi-question capacity, long-term availability, or business accuracy. The [earlier 1–32 retest](CONCURRENCY_RETEST.md) remains a separate dataset.

## Service after testing

The expanded settings were retained persistently: HTTP/question admission 256, engine limit 128, and the raised rate threshold. The original eight-request gate is removed. The automatic numeric check first triggered a rollback. The same tested configuration was then re-enabled and persisted, and authenticated public inference was verified afterward.

Authentication, request-body/context limits, and the inference deadline remain enabled. An automatic rollback timer protected the test and was stopped after successful finalization.

## Reproduce and inspect

The server must first be configured for the values above. CLI configuration arguments below record expected settings; they do not modify the server.

```bash
uv run --no-sync python scripts/benchmark_service.py \
  --credentials .deployment/api-credentials.json \
  --engine-seqs 128 --http-limit 256 --question-limit 256 \
  --requests-per-minute 120000 --burst 512 \
  --label sweep-1-128 --levels 1 2 4 8 16 32 64 128 \
  --requests 1280 --output artifacts/sweep-1-128-new
```

Evidence: [summary](results/concurrency-1-128-2026-10-13/summary.json), [sweep metrics](results/concurrency-1-128-2026-10-13/sweep-1-128/benchmark.json), [repeat metrics](results/concurrency-1-128-2026-10-13/repeat-64-128/benchmark.json), [raw sweep responses](results/concurrency-1-128-2026-10-13/sweep-1-128/requests.jsonl), [actual startup arguments](results/concurrency-1-128-2026-10-13/host/final-command.json), [final public checks](results/concurrency-1-128-2026-10-13/final-public-check.json), and [SHA-256 manifest](results/concurrency-1-128-2026-10-13/sha256.json).
