import asyncio
import json
from copy import deepcopy

import pytest

from decider_vllm.evaluation import compare_responses, percentile, validate_response


def test_comparison_fails_on_changed_choice_even_inside_probability_tolerance():
    reference = {
        "model": "reference",
        "answers": {
            "action": {
                "type": "choice",
                "choice": "wait",
                "probabilities": {"wait": 0.51, "approve": 0.49},
                "confidence": 0.02,
            }
        },
        "usage": {"input_tokens": 10, "output_tokens": 1},
    }
    candidate = deepcopy(reference)
    candidate["answers"]["action"].update(
        choice="approve", probabilities={"wait": 0.49, "approve": 0.51}
    )
    assert not compare_responses(reference, candidate, tolerance=0.03)["passed"]


def test_percentiles_and_empty_failure_only_batch():
    assert percentile(list(range(1, 81)), 0.95) == 76
    assert percentile([], 0.95) is None


def test_comparison_accepts_decimal_tolerance_boundary_but_rejects_larger_difference():
    reference = {
        "model": "reference",
        "answers": {
            "action": {
                "type": "choice",
                "choice": "approved",
                "probabilities": {"approved": 0.7, "wait": 0.3},
                "confidence": 0.5917,
            }
        },
        "usage": {"input_tokens": 10, "output_tokens": 1},
    }
    candidate = deepcopy(reference)
    candidate["answers"]["action"]["confidence"] = 0.6117
    assert compare_responses(reference, candidate, tolerance=0.02)["passed"]
    candidate["answers"]["action"]["confidence"] = 0.6118
    assert not compare_responses(reference, candidate, tolerance=0.02)["passed"]


def test_http_200_without_decisions_is_not_counted_as_success(payment):
    with pytest.raises(ValueError, match="missing questions"):
        validate_response(
            payment.model_dump(),
            {"model": "broken", "answers": {}, "usage": {"input_tokens": 1, "output_tokens": 0}},
        )


async def test_concurrency_warmups_are_excluded_from_latency_statistics(tmp_path, monkeypatch):
    from decider_vllm import evaluation

    case = {
        "id": "simple",
        "request": {
            "state": "ready",
            "questions": {"ready": {"type": "noul", "instructions": "Is it ready?"}},
        },
    }
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps([case]))
    calls = 0

    async def fake_post(client, url, request):
        nonlocal calls
        calls += 1
        position = calls
        await asyncio.sleep(0)
        # C=1: 2 warm + 4 measured; C=2: 4 warm + 4 measured.
        warm = position <= 2 or 7 <= position <= 10
        return {"marker": position}, 10000 if warm else 10

    monkeypatch.setattr(evaluation, "post", fake_post)
    output = tmp_path / "run"
    result = await evaluation.benchmark("http://unused", corpus, output, [1, 2], 4)
    assert calls == 14
    assert result["warmups_excluded"] == 6
    assert len((output / "warmups.jsonl").read_text().splitlines()) == 6
    assert len((output / "requests.jsonl").read_text().splitlines()) == 8
    assert all(cell["p95_ms"] == 10 and cell["mean_ms"] == 10 for cell in result["cells"])
