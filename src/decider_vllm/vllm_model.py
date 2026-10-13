"""vLLM 0.31 pooling plugin. Imported inside model inspection / GPU workers."""

import importlib.metadata
from pathlib import Path

from vllm.model_executor.layers.pooler.abstract import Pooler
from vllm.model_executor.models.adapters import as_embedding_model
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM

from . import VLLM_VERSION
from .artifacts import load_config
from .pointer import load_pointer, score_packed


class DeciderPooler(Pooler):
    def __init__(self, model_path: str, hidden_size: int):
        super().__init__()
        config = load_config(model_path)
        self.head = load_pointer(
            Path(model_path) / "decider/head.safetensors", hidden_size, config.pointer_dim
        )

    def get_supported_tasks(self):
        return {"classify"}

    def forward(self, hidden_states, pooling_metadata):
        cursor = pooling_metadata.get_pooling_cursor()
        if cursor.is_partial_prefill() or not all(cursor.get_finished_mask()):
            raise RuntimeError("Decider requires full prefill; disable chunking and prefix caching")
        lengths = cursor.prompt_lens_cpu.tolist()
        metadata = []
        for length, params in zip(lengths, pooling_metadata.pooling_params, strict=True):
            if params.extra_kwargs is None:
                # vLLM's startup profiler synthesizes PoolingParams without user data.
                # The HTTP adapter always supplies validated per-request metadata.
                info = {
                    "prompt_length": length,
                    "answer_position": length - 1,
                    "option_positions": [max(0, length - 2), length - 1],
                    "temperature": 1.0,
                }
            else:
                info = params.extra_kwargs["decider"]
            metadata.append(info)
        return score_packed(self.head, hidden_states, lengths, metadata)


_QwenForPooling = as_embedding_model(Qwen3_5ForCausalLM)


class StrandsDeciderForPooling(_QwenForPooling):
    def __init__(self, *, vllm_config, prefix=""):
        if importlib.metadata.version("vllm") != VLLM_VERSION:
            raise RuntimeError(f"This adapter is pinned to vLLM {VLLM_VERSION}")
        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError("Initial adapter supports a single GPU only")
        if (
            vllm_config.cache_config.enable_prefix_caching
            or vllm_config.scheduler_config.enable_chunked_prefill
        ):
            raise ValueError("Disable prefix caching and chunked prefill for Pointer Head")
        if vllm_config.model_config.quantization:
            raise ValueError("Quantized models have not been validated by this adapter")
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def _init_pooler(self, vllm_config, prefix=""):
        return DeciderPooler(
            vllm_config.model_config.model,
            vllm_config.model_config.hf_text_config.hidden_size,
        )

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        # These parameters were loaded from the separate decision-head artifact.
        return loaded | {f"pooler.head.{name}" for name, _ in self.pooler.head.named_parameters()}
