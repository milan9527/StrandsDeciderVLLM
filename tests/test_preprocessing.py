from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
import torch
from strands_decider.infer import EngineConfig, SystemOneEngine
from strands_decider.schema import SystemOneRequest

from decider_vllm.preprocessing import Preprocessor


def official_inputs(tokenizer, config, request, strict=True):
    captured = []

    class SpyModel:
        def __call__(self, **kwargs):
            captured.append(kwargs)
            n = kwargs["n_slots"]
            width = int(n.max())
            logits = torch.zeros(len(n), width)
            logits[torch.arange(width)[None, :] >= n[:, None]] = -torch.inf
            return {"log_probs": logits.log_softmax(-1)}

    model = SpyModel()
    model.config = config
    engine = object.__new__(SystemOneEngine)
    engine.model, engine.tok, engine.device = model, tokenizer, "cpu"
    engine.cfg = EngineConfig(
        device="cpu", strict_window=strict, use_prefix_cache=False, max_batch=8
    )
    engine.evaluate(request)
    result = []
    for batch in captured:
        for i, mask in enumerate(batch["attention_mask"]):
            result.append(
                (
                    tuple(batch["input_ids"][i, mask.bool()].tolist()),
                    tuple(batch["opt_idx"][i, : batch["n_slots"][i]].tolist()),
                    float(batch["temperature"][i]),
                )
            )
    return result


def test_matches_official_forward_inputs_for_all_primitives(tokenizer, config, payment):
    prepared = Preprocessor(tokenizer, config).prepare(payment)
    reference = official_inputs(tokenizer, config, payment)
    for actual, (tokens, positions, temperature) in zip(prepared, reference, strict=True):
        assert actual.token_ids == tokens
        assert actual.option_positions == positions
        assert actual.temperature == pytest.approx(temperature)


def test_multiple_chunks_remain_independent_under_threads(tokenizer, config, payment):
    requests = [
        SystemOneRequest(
            state="state " * (i + 1),
            questions={
                f"question_{j}": q for j, q in enumerate(list(payment.questions.values()) * 4)
            },
        )
        for i in range(16)
    ]
    preprocessor = Preprocessor(tokenizer, config)
    with ThreadPoolExecutor(max_workers=8) as pool:
        actual = list(pool.map(preprocessor.prepare, requests))
    for request, prepared in zip(requests, actual, strict=True):
        reference = official_inputs(tokenizer, config, request)
        assert [(q.token_ids, q.option_positions) for q in prepared] == [
            (tokens, positions) for tokens, positions, _ in reference
        ]


def test_strict_window_and_original_truncation(tokenizer, config, payment):
    request = payment.model_copy(update={"state": "large state " * 300})
    short = replace(config, max_length=256)
    with pytest.raises(ValueError, match="context window"):
        Preprocessor(tokenizer, short).prepare(request)
    result = Preprocessor(tokenizer, short, strict_window=False).prepare(request)
    original = official_inputs(tokenizer, short, request, strict=False)
    assert [(q.token_ids, q.option_positions) for q in result] == [
        (tokens, positions) for tokens, positions, _ in original
    ]


def test_no_silent_image_or_excess_question_acceptance(tokenizer, config, payment):
    processor = Preprocessor(tokenizer, config, max_questions=2)
    with pytest.raises(ValueError, match="At most"):
        processor.prepare(payment)
    with pytest.raises(ValueError, match="images"):
        processor.prepare(payment.model_copy(update={"images": ["example"]}))


def test_response_readout_uses_original_semantics(tokenizer, config, payment):
    processor = Preprocessor(tokenizer, config)
    questions = processor.prepare(payment)
    assert processor.answer(questions[0], [0.1, 0.8, 0.1]).choice == "credit_payment"
    assert 0 <= processor.answer(questions[1], [0.4, 0.6]).noul <= 1
    assert processor.answer(questions[2], [0.1, 0.8, 0.1]).type == "score"
