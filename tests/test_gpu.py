"""Opt-in real CUDA test. Skipped explicitly on development machines without GPU."""

import asyncio
import json
import os
from pathlib import Path

import pytest
import torch
from strands_decider.schema import SystemOneRequest

from decider_vllm.backend import RuntimeSettings, create_backend

MODEL = os.environ.get("DECIDER_VLLM_GPU_MODEL")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not MODEL or not torch.cuda.is_available(), reason="Requires an exported model and CUDA"
    ),
]


async def test_real_vllm_load_mixed_primitives_and_concurrent_isolation():
    """Actually creates the GPU engine; no engine, hidden-state, or output mocks."""
    corpus = json.loads((Path(__file__).parents[1] / "examples/payment-corpus.json").read_text())
    requests = [SystemOneRequest.model_validate(c["request"]) for c in corpus]
    mixed = SystemOneRequest.model_validate(
        json.loads((Path(__file__).parents[1] / "examples/payment.json").read_text())
    )
    requests.append(mixed)
    backend = create_backend(RuntimeSettings(model=MODEL, max_num_seqs=8))
    try:
        baseline = [await backend.evaluate(request) for request in requests]
        concurrent = await asyncio.gather(*(backend.evaluate(r) for r in requests))
        from decider_vllm.evaluation import compare_responses

        for expected, actual in zip(baseline, concurrent, strict=True):
            comparison = compare_responses(
                expected.model_dump(), actual.model_dump(), tolerance=0.02
            )
            assert comparison["passed"], comparison
        await backend.check_health()
    finally:
        backend.shutdown()
