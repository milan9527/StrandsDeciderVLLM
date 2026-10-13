# Validation record

English | [简体中文](VALIDATION.zh-CN.md)

## Public API release

The subsequent authenticated public deployment passed **33 local tests** (one GPU pytest test skipped on the CPU host) and **22 public HTTPS acceptance checks**. Real inference passed 21 parity requests containing 23 judgments and all 160 requests in paced batches at concurrency eight. A deliberate 64-request burst produced nine successes and 55 expected 429 responses, with no 5xx and normal capacity recovery. See the [deployment record](DEPLOYED.md) for the actual endpoint, configuration, latency, and raw evidence.

The sections below preserve the earlier adapter and benchmark validation.

Date: 2026-10-13. Local development used an aarch64 CPU host. GPU validation ran separately on an AWS g6.4xlarge with an NVIDIA L4, with the inference container limited to 4 vCPUs and 14 GiB RAM.

| Check | Result |
|---|---|
| Local automated tests | 26 passed; one pytest GPU test skipped because the local host lacks CUDA |
| Ruff lint and formatting | Passed |
| Python source compilation | Passed |
| Wheel / source distribution build | Passed; plugin entry point registered; packages exclude models, credentials, and runtime caches |
| Real v19 LoRA merge | Completed; merged 16,819,200 adapter parameters |
| Export integrity | SHA-256 checks passed for nine files |
| CPU behavior comparison with real weights | All 21 requests and 23 judgments passed |
| Actual vLLM GPU loading and comparison | Completed; 21 real API requests and 23 judgments passed |
| Actual Docker image build | Completed with vLLM 0.31.0; startup with the real model passed |
| GPU throughput and latency | Measured at concurrency 1, 2, 4, 8, 16, and 32; see the [concurrency report](GPU_CONCURRENCY.md) |
| Switch of the original deployment | **Not performed** |

## CPU behavior comparison with real weights

Compared implementations:

- Reference: the pinned official Strands Decider implementation, with the original backbone, LoRA, and Pointer Head.
- Adapter: the real backbone with merged LoRA, this project's packed hidden-state processing, and the original Pointer Head.

Backbone weights were exported in BF16; both implementations ran on the CPU in FP32. This validates weight export, input positions, and the decision-head adapter. **It does not execute vLLM CUDA kernels and cannot replace GPU behavior validation.**

The test covers five steps in each of four payment scenarios, giving 20 requests, plus one mixed request containing `choice`, `noul`, and `score`.

| Metric | Result |
|---|---:|
| Requests passed | 21 / 21 |
| Judgments | 23 |
| Matching candidate choices | 21 / 21 |
| Maximum absolute probability difference | 0.0032 |
| Maximum absolute confidence difference | 0.0042 |
| Pass criteria | Same choices; probability and confidence differences ≤ 0.02 |

This measures behavioral consistency before and after adaptation. It does not measure business accuracy or show that the original model's decision errors have been corrected.

See the [per-case CPU results](results/local-real-weights.json) and the [historical check manifest](results/checks.json) for the initial CPU environment and code hashes. That manifest records the state before GPU testing; current GPU evidence is described below.

To reproduce:

```bash
uv run --no-sync python scripts/validate_local.py \
  --checkpoint ../StrandsDecider/.deployment/models/decider \
  --base ../StrandsDecider/.deployment/models/base \
  --model models/decider-vllm \
  --corpus examples/payment-corpus.json \
  --output artifacts/local-real-weights-rerun.json
```

## Automated test coverage

- Matches the tokens, candidate positions, and temperature values that official `evaluate` actually passes to the model.
- Covers multithreaded preprocessing, multiple question groups, context-window overflow, and official truncation rules.
- Checks packed batches with different lengths, candidate counts, and calibration temperatures against independent scoring.
- Keeps calibration in FP32 when the default floating-point dtype is BF16.
- Covers concurrent requests, unique request IDs, and correct association of questions and answers.
- Checks cleanup and capacity recovery after request cancellation, individual question failure, or cancellation API failure.
- Covers HTTP 429 and 504, disconnect cancellation, and body-size limits for chunked requests.
- Detects changed choices; the benchmark does not count an HTTP 200 with missing judgments as a success.
- Excludes each concurrency level's warmup requests from measured latency statistics.
- Accepts a decimal difference of exactly 0.02 and rejects 0.0201, avoiding false failures caused by floating-point representation.

The asynchronous scheduling and HTTP boundary tests use test doubles. The real-weight comparison uses the actual model, without fabricated candidate answers. `tests/test_gpu.py` was skipped on the local CPU host; the separate GPU API tests below actually executed vLLM and CUDA.

## Real GPU behavior and concurrency

The reference service used the original production image pinned by digest. vLLM used the same model with merged LoRA, and each service had exclusive use of the GPU during its run. The production-kernel baseline was not replaced with CPU simulation, random answers, or the helper reference server.

| GPU behavior metric | Result |
|---|---:|
| Requests / judgments | 21 / 23; all passed |
| Maximum absolute probability difference | 0.0112 |
| Maximum absolute confidence difference | 0.0150 |
| Maximum absolute score difference | 0.0003 |
| Pass criteria | Same choices; absolute numeric differences ≤ 0.02 |

All 480 original-service requests succeeded. In the first vLLM run, two of 480 requests timed out; all 478 successful responses selected the same candidates as the original. The initial comparison tool incorrectly rejected four confidence differences of exactly 0.02. The floating-point boundary check was corrected, and the original audit was retained; the tolerance was not relaxed. A repeat run with the same inference settings completed all 480 requests successfully and passed all behavior checks. Under concurrency, the maximum probability difference was 0.0150 and the maximum confidence difference was 0.0200.

The extended run submitted 400 requests at each of concurrency 4, 8, 16, and 32. The first three levels completed without failure; concurrency 32 had three timeouts. Across all three vLLM runs, 2,555 of 2,560 requests succeeded, and every successful response passed comparison against the original. Concurrency levels 4 and 8 each completed a cumulative 560 requests without failure.

See the [GPU concurrency report](GPU_CONCURRENCY.md) and the [raw responses, statistics, and verification manifest](results/gpu-concurrency-2026-10-13/verification.json). The live demo was not switched to vLLM. The cause of high-concurrency timeouts still needs investigation, and these short tests do not establish production reliability.
