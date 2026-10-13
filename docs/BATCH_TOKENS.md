# Batch token budget experiment

English | [简体中文](BATCH_TOKENS.zh-CN.md)

2026-10-13. On the same On-Demand EC2 `g6.xlarge` / NVIDIA L4, change only `max_num_batched_tokens`: 8192, 16384, and 32768. Engine sequence capacity stays at 128; HTTP and question admission remain 256.

**23,040/23,040** measured requests succeeded, with **0** HTTP failures and **0** changed choices. Numeric parity passed **23,040/23,040** requests under the unchanged 0.02 absolute probability/confidence tolerance. **1,948** warmup requests are excluded; warmup failures: **0**.

**Larger batch budgets did not improve observed throughput; retain 8192.** Concurrency 16–128 remained around 30 requests/second. At concurrency 128, P95 changed from 4.93 seconds with the 8192 baseline to 5.98 seconds with 32768; the 32768 repeat measured 5.90 seconds. The single concurrency-8 runs vary; their percentage differences do not establish a causal parameter effect.

## Same-run comparison

| Batch token budget | Concurrency | RPS | Mean ms | P95 ms | RPS vs fresh 8192 baseline | Peak VRAM MiB | Numeric parity |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 8 | 37.01 | 215.8 | 231.0 | +0.0% | 15792 | 1280/1280 |
| 8192 | 16 | 30.55 | 522.8 | 546.1 | +0.0% | 16250 | 1280/1280 |
| 8192 | 32 | 30.34 | 1047.7 | 1069.5 | +0.0% | 16702 | 1280/1280 |
| 8192 | 64 | 30.20 | 2081.6 | 2137.2 | +0.0% | 17272 | 1280/1280 |
| 8192 | 128 | 29.92 | 4069.6 | 4932.3 | +0.0% | 17272 | 1280/1280 |
| 16384 | 8 | 33.67 | 237.3 | 242.7 | -9.0% | 15382 | 1280/1280 |
| 16384 | 16 | 30.50 | 523.7 | 547.7 | -0.2% | 15840 | 1280/1280 |
| 16384 | 32 | 29.94 | 1061.6 | 1081.9 | -1.3% | 16292 | 1280/1280 |
| 16384 | 64 | 29.75 | 2119.0 | 2189.4 | -1.5% | 17186 | 1280/1280 |
| 16384 | 128 | 29.61 | 4142.2 | 5166.7 | -1.1% | 17570 | 1280/1280 |
| 32768 | 8 | 34.27 | 233.0 | 242.2 | -7.4% | 14484 | 1280/1280 |
| 32768 | 16 | 30.71 | 520.0 | 535.0 | +0.5% | 14942 | 1280/1280 |
| 32768 | 32 | 29.78 | 1067.2 | 1087.2 | -1.8% | 15594 | 1280/1280 |
| 32768 | 64 | 29.67 | 2123.8 | 2200.0 | -1.8% | 16704 | 1280/1280 |
| 32768 | 128 | 29.44 | 4201.1 | 5983.9 | -1.6% | 18518 | 1280/1280 |

VRAM values are peaks in one-second samples during measured phases, not memory requirements or guaranteed maxima. They include allocator caches; repeated runs retain prior allocations. RPS changes use the fresh 8192 baseline at matching concurrency; the earlier 1–128 experiment remains separate.

## 32768 repeat

| Concurrency | RPS | Mean ms | P95 ms | Numeric parity |
|---:|---:|---:|---:|---:|
| 32 | 29.67 | 1071.6 | 1105.1 | 1280/1280 |
| 64 | 29.22 | 2157.9 | 2233.4 | 1280/1280 |
| 128 | 29.23 | 4244.4 | 5903.6 | 1280/1280 |

## Interpretation

32768 can accommodate 64 of the longest current inputs: `64 × 445 = 28480` tokens. Concurrency 64 still stays around 30 requests/second, so the higher budget did not remove the observed throughput plateau. A larger budget permits the scheduler to schedule more tokens; it neither forces full batches nor increases GPU processing capacity.

No kernel or CPU profiling was performed, so this experiment does not isolate a particular kernel, memory bandwidth, or CPU stage as the bottleneck. Profile actual batch sizes and GPU/CPU time before evaluating adapter changes such as CUDA Graphs. Prefix caching and chunked prefill cannot simply be enabled because the current adapter requires complete prefill.

## Method

- Test concurrency 8, 16, 32, 64, and 128 for each budget; repeat 32, 64, and 128 at budget 32768.
- Each cell contains 1,280 measured requests: four scenarios × five independent choice questions, with each of the 20 inputs repeated exactly 64 times. Requests must exactly match the original reference capture.
- One question per request, 304–445 input tokens, mean 373.9. Each scenario retains fixed business facts: no payment execution or propagation of answers into subsequent inputs.
- Fixed asynchronous workers outside the GPU host call public HTTPS, reuse connections, and send the next request after receiving a complete response. No retries. Each cell has `max(20, 2 × concurrency)` separate warmups.
- Client latency includes transport, queueing, and inference; server latency is retained separately. Throughput is successes / measured phase wall time. Successful-response percentiles use nearest rank.
- Keep vLLM 0.31.0, model image and weights, BF16 backbone / FP32 Pointer Head, 4 vCPU / 14 GiB container, 4096 context, and 0.7 GPU memory target unchanged. Prefix caching, chunked prefill, and CUDA Graphs remain disabled. Rate threshold 120,000/minute, burst 512; server/client deadlines 25/40 seconds.
- Compare every response against the fixed original capture: candidate ID, probabilities, and confidence. Adapter consistency is not business accuracy. Numeric exceptions are retained without relaxing tolerance.

8192 / 16384 / 32768 are scheduling token budgets, not per-request input lengths. At this workload's mean length, they roughly accommodate 21 / 43 / 87 whole inputs; the scheduler determines actual batches. Results apply to these fixed short inputs, not arbitrary long contexts or long-term availability.

## Service state and evidence

Restored the **8192-token batch budget**, retaining engine capacity **128** and HTTP/question admission **256**. Restored runtime arguments match the pre-test configuration. The GPU remains running.


[Summary](results/batch-tokens-2026-10-13/summary.json) ·
[Frozen client](results/batch-tokens-2026-10-13/benchmark_service.py) ·
[Exact corpus](results/batch-tokens-2026-10-13/payment-corpus.json) ·
[GPU samples](results/batch-tokens-2026-10-13/host/gpu-monitor.csv) ·
[Original runtime arguments](results/batch-tokens-2026-10-13/host/original-command.json) ·
[Restored runtime arguments](results/batch-tokens-2026-10-13/host/final-command.json) ·
[Final authenticated API check](results/batch-tokens-2026-10-13/final-public-check.json) ·
[SHA-256](results/batch-tokens-2026-10-13/sha256.json)
