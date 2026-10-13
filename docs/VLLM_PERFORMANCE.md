# Decider vLLM: concurrency and batch-budget summary

English | [简体中文](VLLM_PERFORMANCE.zh-CN.md)

This document summarizes the Decider vLLM experiments collected on **2026-10-13**: the original serial-service comparison, concurrency retests, the 1–128 sweep, and larger batch token budgets. The separate [Decider vLLM versus Jev report](JEV_CONCURRENCY.md) covers the two public APIs.

**For these short decision inputs on a single NVIDIA L4, concurrency 4–8 is a useful starting range.** The standalone 1–128 sweep peaked at 33.84 decisions/second at concurrency 8. Raising concurrency mainly increased latency. Raising the batch token budget from 8192 to 16384 or 32768 did not improve observed throughput, so the deployed budget remains **8192**.

## What the numbers mean

| Term | Meaning in these tests |
|---|---|
| Request | One decision for one business step, not a complete payment workflow |
| Client concurrency | Workers waiting for responses; each immediately submits another request after completion |
| RPS | Successfully completed decision requests / measured wall time |
| Latency | Client HTTP round trip, including transport, queueing, and processing |
| P95 / P99 | Nearest-rank percentiles of successful responses; failures still affect throughput and are counted separately |
| `max_num_seqs` | Engine sequence limit, not a guarantee that all admitted requests run in the same batch |
| `max_num_batched_tokens` | Total scheduling token budget per batch, not a request count or individual input length |

The shared corpus contains four payment scenarios × five independent steps: **20 fixed inputs**, each containing one choice question. Inputs have **304–445 tokens, mean 373.9**, including business facts, instructions, and candidate criteria. Decider's `output_tokens=1` counts one decision; the response JSON is assembled by the adapter, not generated token by token. Business state does not advance between steps.

Warmups are excluded. Workers reuse HTTP connections and do not retry failed requests. These repeated inputs test performance and adaptation consistency; they do not constitute thousands of independent business-accuracy examples.

## Experiments and evidence

Each row is a separate experiment. Hardware paths, limits, and sample counts differ; results are not pooled.

| Experiment | Environment / purpose | Measured Decider vLLM requests | Main observation |
|---|---|---:|---|
| [Original versus vLLM](GPU_CONCURRENCY.md) | Same g6.4xlarge / L4 host; inference container limited to 4 vCPU / 14 GiB; loopback HTTP | 2555/2560 successful across three vLLM runs | At concurrency 8, the completed repeat reached 34.91 RPS versus the original's 14.76; five timeouts remain in the other runs |
| [HTTP cleanup retest](CONCURRENCY_RETEST.md) | g6.xlarge / L4; public HTTPS; engine limits 8 and 32 | 5200/5200 successful; all numeric comparisons passed | Concurrency 16/32 succeeded; the extended concurrency-8 repeat reached 33.32 RPS |
| [Concurrency 1–128](CONCURRENCY_128.md) | g6.xlarge / L4; public HTTPS; engine 128, HTTP/question admission 256, batch budget 8192 | 12800/12800 successful, including 64/128 repeats | Peak sweep 33.84 RPS at concurrency 8; one confidence difference exceeded tolerance |
| [Batch budgets 8192/16384/32768](BATCH_TOKENS.md) | Same public deployment; only the batch-budget argument changes | 23040/23040 successful; all numeric comparisons passed | No throughput gain; concurrency-128 P95 increased with the larger budgets |

The original comparison's gains include merged weights, different kernels, thread settings, and scheduling. They cannot be attributed entirely to batching. The HTTP cleanup fix was followed by successful retests, but those retests did not establish the cause of every historical timeout.

## Standalone 1–128 sweep

This table comes entirely from the [standalone sweep](results/concurrency-1-128-2026-10-13/sweep-1-128/benchmark.json), using **1280 measured requests per level**. Each of the 20 inputs repeats exactly 64 times. Every request in this table succeeded.

| Concurrency | RPS | Mean ms | P95 ms | P99 ms |
|---:|---:|---:|---:|---:|
| 1 | 18.10 | 55.1 | 59.9 | 63.7 |
| 2 | 22.85 | 87.4 | 89.9 | 91.7 |
| 4 | 33.11 | 120.6 | 131.8 | 135.2 |
| 8 | 33.84 | 236.0 | 245.0 | 247.0 |
| 16 | 29.34 | 544.4 | 561.8 | 572.7 |
| 32 | 30.06 | 1057.9 | 1089.9 | 1103.5 |
| 64 | 29.76 | 2112.7 | 2178.7 | 2822.1 |
| 128 | 29.65 | 4111.9 | 4986.1 | 7568.0 |

The additional concurrency-128 repeat reached **29.65 RPS, P95 5018.0 ms**. Allowing more concurrent requests did not provide proportional processing capacity.

One `underpaid/approve` response in the first 128-concurrency pass kept the choice `approved`, but confidence changed from 0.5917 to 0.6119. The **0.0202** difference exceeded the unchanged **0.02** tolerance. Numeric parity therefore passed **12799/12800**, independently of HTTP success.

## Larger batch token budgets

The following table uses the [batch-budget experiment's own fresh baseline](results/batch-tokens-2026-10-13/summary.json). It does not reuse the sweep above. All settings except `max_num_batched_tokens` remained the same; each cell contains 1280 successful requests.

Each cell shows **RPS / P95 milliseconds**.

| Concurrency | 8192 tokens | 16384 tokens | 32768 tokens |
|---:|---:|---:|---:|
| 8 | 37.01 / 231.0 | 33.67 / 242.7 | 34.27 / 242.2 |
| 16 | 30.55 / 546.1 | 30.50 / 547.7 | 30.71 / 535.0 |
| 32 | 30.34 / 1069.5 | 29.94 / 1081.9 | 29.78 / 1087.2 |
| 64 | 30.20 / 2137.2 | 29.75 / 2189.4 | 29.67 / 2200.0 |
| 128 | 29.92 / 4932.3 | 29.61 / 5166.7 | 29.44 / 5983.9 |

The 32768-budget repeat at concurrency 128 reached **29.23 RPS, P95 5903.6 ms**. The 23040-request total includes separate repeats at concurrency 32, 64, and 128; all passed numeric parity.

A 32768-token budget can accommodate 64 of the longest current inputs (`64 × 445 = 28480`), yet concurrency 64 remained around 30 RPS. A higher ceiling does not force larger actual batches or increase GPU compute capacity. This experiment did not isolate the limiting kernel, CPU stage, or network component. Small differences between individual runs should not be treated as proven tuning effects.

## Current configuration and limits of the conclusions

The public service keeps batch budget **8192**, engine sequence limit **128**, and HTTP/question admission **256**. These are admission and scheduling limits, not recommended simultaneous load or an RPS guarantee. The configured rate threshold of 2000 requests/second is also not measured GPU throughput. See [deployed configuration](DEPLOYED.md).

Use concurrency **4–8 as a starting point for this corpus**, then choose a limit against the required latency and observed failure rate. Keep probability/confidence checks separate from candidate-ID agreement. This remains a BF16 backbone with the original FP32 Pointer Head; the benchmark did not train or improve model weights.

The later [Jev comparison](JEV_CONCURRENCY.md) retained one Decider HTTP 502 and ten confidence differences of 0.0202. Its Decider concurrency-128 client P95 was 11236.8 ms, while server-reported P95 was 187.1 ms. Much of that run's delay was outside the server timer; its source remains unisolated. That measurement is kept separate from the standalone sweep and shows why neither high-concurrency tail latency nor hardware saturation should be inferred from one aggregate number.

Before further tuning, profile actual batch sizes and time spent in client transport, request handling, CPU preprocessing, scheduling, and GPU kernels. CUDA Graphs require compatibility work; prefix caching and chunked prefill cannot simply be enabled because the adapter requires complete prefill. Longer contexts, multiple questions per request, sustained arrivals, and multiple replicas need their own tests.

Raw requests, warmups, numerical audits, runtime arguments, and SHA-256 manifests are linked from each experiment above. The [validation record](VALIDATION.md) tracks checks and exceptions; the [Jev report](JEV_CONCURRENCY.md) covers the separate service comparison.
