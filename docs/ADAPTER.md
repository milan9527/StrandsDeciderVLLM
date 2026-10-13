# Adapter guide

English | [简体中文](ADAPTER.zh-CN.md)

## Mapping to official inference

| Official implementation | This adapter |
|---|---|
| Qwen3.5 text backbone + PEFT LoRA | `export.py` uses the official backbone loader and PEFT `merge_and_unload(safe_merge=True)` |
| `render_state` / `render_question` | Reuses the official implementations from the pinned commit |
| `_chunks` / `_fit` / `_option_token_index` | Creates a lightweight context for each request; `_last_offsets` is never shared |
| Final hidden states from the Transformer | Full prefill hidden states from the vLLM Qwen3.5 model |
| `PointerHead` | Uses the official class and original `head.safetensors`, retaining FP32 |
| `temperature_by_kind` | Passed to the pooler with each question |
| `_to_answer` | Reuses the official implementation, preserving all three types and ordinal smoothing for `score` |
| `/v1/systemone` | Preserves the request/response schema and `latency_ms` |

The plugin registers `StrandsDeciderForPooling` through `vllm.general_plugins`. Start the service with this project's `decider-vllm serve`. Standard chat/completions, embeddings, and bare `/pooling` requests lack the candidate-position metadata needed to provide an equivalent interface.

## Key files

| File | Responsibility |
|---|---|
| `src/decider_vllm/export.py` | Merge and export, with atomic writes and overwrite protection |
| `src/decider_vllm/artifacts.py` | Export manifest and integrity checks |
| `src/decider_vllm/preprocessing.py` | Official text processing and per-request positions |
| `src/decider_vllm/pointer.py` | Batched candidate scoring over packed token inputs |
| `src/decider_vllm/vllm_model.py` | Qwen3.5 pooling plugin and GPU weight loading |
| `src/decider_vllm/backend.py` | Asynchronous requests, scheduling metadata, and cancellation |
| `src/decider_vllm/server.py` | HTTP API, admission limits, and lifecycle management |
| `src/decider_vllm/evaluation.py` | Real-endpoint behavior comparison, benchmarking, and raw evidence |

## Initial version constraints

- Supports only the v19 Qwen3.5 Pointer Head checkpoint; checkpoints are not upgraded automatically.
- Single GPU, with tensor parallelism = pipeline parallelism = 1.
- BF16 backbone and FP32 head; quantization and dynamic LoRA are unsupported.
- Enforces a strict 4096-token window by default, returning 422 when exceeded. `--allow-truncation` uses the official truncation logic, but still rejects requests if candidate options would be truncated.
- Image inputs are rejected.
- Prefix caching and chunked prefill are disabled to preserve candidate hidden states. The pooler checks processed lengths and rejects incomplete data.
- CUDA Graphs are disabled; graph capture compatibility with this custom pooler has not been validated.
- Does not rely on vLLM's default LAST or MEAN pooling, or a standard fixed-label classification head.
- Each question is submitted with its complete state. Token usage and latency can differ from the original service's shared-context path.
- The pooler accepts synthetic profiling requests without business metadata during vLLM startup. The business API always supplies that metadata explicitly.
- The service does not implement the demo's GPU start/stop controls, DynamoDB exclusive lease, user login, or business execution logic.

## Validation goals

1. Use the same tokenizer, text templates, candidate order, and context-window rules for the original and exported models.
2. Preserve behavior for single questions, multiple questions, and batches with different input lengths and candidate counts.
3. Match the official reference's choices. The default absolute probability/confidence tolerance is 0.02 and can be set explicitly.
4. Keep inputs, candidate positions, and outputs associated with the correct requests under concurrency.
5. Cancel sibling questions and restore capacity after a timeout, disconnect, or individual question failure.
6. Complete loading and behavior checks on the target GPU before judging performance or suitability as a replacement.

Changing the inference engine does not correct the model's learned business judgments. Existing mistakes can remain; decision quality and performance need separate evaluation.

## GPU measurement status

Real model loading, behavior comparison for 21 requests / 23 judgments, and concurrency measurements from 1 to 32 have been completed on an L4. See the [GPU concurrency report](GPU_CONCURRENCY.md). With the current settings, throughput plateaus around concurrency 4–8. The initial run had two timeouts at concurrency 16, and the extended run had three more at concurrency 32. Their root cause remains unresolved.

The subsequent [public API deployment](API.md) adds correlation IDs and stage timings for HTTP requests, preprocessing, vLLM submission, and result delivery. An HTTP test reproduced a disconnect-watcher cancellation race that could stall response cleanup after inference completed; the watcher now awaits ASGI disconnect messages directly. This is a verified code fix, but it does not establish the cause of every historical GPU timeout. Original measurements remain unchanged; public-endpoint validation is recorded separately in the [deployment record](DEPLOYED.md).

Future performance work should evaluate larger token batches, CPU preprocessing overhead, synchronization points, and CUDA Graph compatibility individually, repeating behavior checks after each change. The measured gains include LoRA merging, different kernels, and batch scheduling; they cannot be attributed entirely to one vLLM setting.

## Dependency references

- [Pinned Strands Decider source](https://github.com/strands-labs/strands-decider/tree/63d24ae286e50105fba0bd1db15b0e243aea4650)
- [vLLM 0.31.0](https://github.com/vllm-project/vllm/tree/v0.31.0)
- [Qwen3.5 implementation](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/model_executor/models/qwen3_5.py)
- [Pooling metadata](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/pool/metadata.py)
- [Pooling model adapters](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/model_executor/models/adapters.py)
- [Pooling documentation](https://docs.vllm.ai/en/stable/models/pooling_models/)

Dependencies are provided under their respective licenses. This project imports the official model classes and input-processing code through dependencies and does not redistribute model weights.
