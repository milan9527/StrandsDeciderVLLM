# Concurrency retest after the HTTP cleanup fix

English | [简体中文](CONCURRENCY_RETEST.zh-CN.md)

This record describes the earlier run that restored eight-request admission at its end. A subsequent [1–128 test](CONCURRENCY_128.md) raised and retained the service limits, with one numeric-tolerance exception documented separately.

**2026-10-13: 5,200/5,200 measured requests succeeded, including concurrency 16 and 32.** No HTTP failures or inference timeouts occurred. Every successful response preserved the original candidate choice and passed the existing 0.02 numeric tolerance. This is a short-run stability result; higher concurrency still increased latency without improving throughput.

## Results

These calls used the actual [public HTTPS endpoint](https://d1e9udi8st2p3g.cloudfront.net) on the existing On-Demand `g6.xlarge` / NVIDIA L4. For the comparison below, engine, question, and HTTP limits were all temporarily set to 32. The rate limit was raised to 200 requests/second with burst 64, so the normal 20 requests/second limit would not determine the result.

First pass, 400 requests per concurrency level:

| Concurrency | Success / requests | RPS | Mean ms | P50 ms | P95 ms | P99 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 400/400 | 18.52 | 53.9 | 53.3 | 58.3 | 61.0 |
| 4 | 400/400 | 32.08 | 124.4 | 123.6 | 132.0 | 134.1 |
| 8 | 400/400 | 36.04 | 220.7 | 220.3 | 228.4 | 254.9 |
| 16 | 400/400 | 29.60 | 539.0 | 539.7 | 549.2 | 559.4 |
| 32 | 400/400 | 30.71 | 1015.9 | 1034.7 | 1050.3 | 1130.9 |

Extended repeat with the same settings, 800 requests per level:

| Concurrency | Success / requests | RPS | Mean ms | P50 ms | P95 ms | P99 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 800/800 | 33.32 | 239.7 | 241.0 | 245.9 | 254.8 |
| 16 | 800/800 | 30.95 | 515.4 | 515.6 | 538.2 | 571.4 |
| 32 | 800/800 | 30.47 | 1039.7 | 1047.6 | 1062.8 | 1131.5 |

A separate control retained the deployed engine/question/HTTP limit of eight and raised only the rate limit: 400/400 requests at concurrency eight succeeded, reaching **34.16 RPS and P95 239.7 ms**.

## Interpretation

- Concurrency 16 and 32 completed successfully in both test rounds. This run did not reproduce the earlier timeouts.
- In the extended repeat, concurrency eight reached **33.32 RPS** with **P95 245.9 ms**. Raising concurrency primarily increased waiting time under these settings.
- The eight-request service limit remains a latency/throughput choice, rather than evidence that vLLM cannot process more concurrent requests.
- The test does not prove the cause of all five timeouts in the [historical benchmark](GPU_CONCURRENCY.md), nor establish long-term reliability.

The service has been restored to **eight concurrent requests and the original 20 requests/second token bucket**, with burst 16. The benchmark's temporarily raised limits are no longer active. Its observed capacity must not be presented as the default public API's sustained rate limit.

## Method and scope

The same deployed image, v19 weights, GPU, batch token budget of 8192, 4096-token context window, 0.7 GPU-memory fraction, and 25-second inference deadline were retained. Prefix caching, chunked prefill, and CUDA Graphs remained disabled. No additional GPU instance was launched.

The corpus has four payment scenarios × five steps, totaling 20 inputs of 304–445 tokens. Each level repeats every input equally. The client runs outside the GPU instance and uses fixed concurrent workers, HTTPS connection reuse, a 40-second client timeout, and no retries. This test focuses on single-question decision requests.

Before each level, `max(20, 2 × concurrency)` requests warm the service. All **312 warmup requests** are recorded separately and excluded from reported latency and throughput. Throughput uses total measured wall time; latency percentiles use nearest-rank successful responses. Raw responses, status codes, request IDs, server/client latency, and per-response comparisons are retained.

The request count is 800 for the engine-eight control, 2,000 for the first engine-32 pass, and 2,400 for the extended repeat. Maximum absolute probability difference from the original was **0.0150**, and maximum confidence difference was **0.0200**. This measures adapter consistency, not business decision accuracy.

The earlier benchmark used loopback HTTP on a g6.4xlarge with CPU/memory limits. This retest uses public HTTPS on a g6.xlarge. Compare concurrency levels within this retest; do not merge the two datasets.

## Reproduction and evidence

Use [benchmark_service.py](../scripts/benchmark_service.py) only after deliberately configuring the service's test limits. Passing `--engine-seqs` records the expected setting; it does not reconfigure the server. Captured container arguments independently verify the settings used here.

```bash
uv run --no-sync python scripts/benchmark_service.py \
  --credentials .deployment/api-credentials.json \
  --engine-seqs 32 --label engine-32-repeat \
  --levels 8 16 32 --requests 800 \
  --output artifacts/concurrency-retest-new
```

The operator used a temporary systemd override and a 25-minute automatic rollback timer. After the run, the original startup arguments matched exactly, the override was removed, and readiness recovered. See [operations](../deployment/README.md) before changing a live service.

Evidence: [summary](results/public-concurrency-2026-10-13/summary.json), [first pass](results/public-concurrency-2026-10-13/engine-32/benchmark.json), [extended repeat](results/public-concurrency-2026-10-13/engine-32-repeat/benchmark.json), [raw repeated requests](results/public-concurrency-2026-10-13/engine-32-repeat/requests.jsonl), [restored arguments](results/public-concurrency-2026-10-13/host/restored-command.json), and [SHA-256 manifest](results/public-concurrency-2026-10-13/sha256.json).
