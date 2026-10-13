import asyncio
from types import SimpleNamespace

import pytest
import torch

from decider_vllm.backend import DecisionBackend, RuntimeSettings
from decider_vllm.preprocessing import Preprocessor


class FakeScheduler:
    def __init__(self, delay=0.01, fail_on_count=None):
        self.delay = delay
        self.fail_on_count = fail_on_count
        self.active, self.peak = 0, 0
        self.ids, self.aborted, self.metadata = set(), set(), []

    async def encode(self, prompt, params, request_id):
        assert request_id not in self.ids
        self.ids.add(request_id)
        self.active += 1
        self.peak = max(self.peak, self.active)
        info = params.extra_kwargs["decider"]
        self.metadata.append(info)
        count = len(info["option_positions"])
        assert len(prompt["prompt_token_ids"]) == info["prompt_length"]
        try:
            await asyncio.sleep(self.delay * count)
            if count == self.fail_on_count:
                raise RuntimeError("Simulated engine failure")
            probabilities = torch.zeros(count)
            probabilities[count - 1] = 1
            yield SimpleNamespace(finished=True, outputs=SimpleNamespace(data=probabilities))
        finally:
            self.active -= 1

    async def abort(self, request_id):
        self.aborted.add(request_id)

    async def check_health(self):
        return None

    def shutdown(self):
        pass


def make_backend(tokenizer, config, engine):
    return DecisionBackend(
        engine,
        Preprocessor(tokenizer, config),
        SimpleNamespace,
        RuntimeSettings(model="unused", max_inflight_questions=4),
    )


async def test_requests_overlap_and_do_not_mix_answers(tokenizer, config, payment):
    engine = FakeScheduler()
    backend = make_backend(tokenizer, config, engine)
    responses = await asyncio.gather(*(backend.evaluate(payment) for _ in range(8)))
    assert engine.peak == 4
    assert engine.active == 0
    assert len(engine.ids) == 24
    for response in responses:
        assert list(response.answers) == list(payment.questions)
        assert response.answers["action"].choice == "manual_review"
        assert response.usage.output_tokens == 3


async def test_cancellation_aborts_active_and_cancels_waiting_questions(tokenizer, config, payment):
    engine = FakeScheduler(delay=1)
    backend = make_backend(tokenizer, config, engine)
    tasks = [asyncio.create_task(backend.evaluate(payment)) for _ in range(3)]
    while engine.peak < 4:
        await asyncio.sleep(0.001)
    for task in tasks:
        task.cancel()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert all(isinstance(r, asyncio.CancelledError) for r in results)
    assert engine.active == 0
    assert engine.ids == engine.aborted
    assert backend.slots._value == 4


async def test_failure_cancels_siblings_and_restores_slots(tokenizer, config, payment):
    engine = FakeScheduler(fail_on_count=2)
    backend = make_backend(tokenizer, config, engine)
    with pytest.raises(RuntimeError, match="Simulated"):
        await backend.evaluate(payment)
    assert engine.active == 0
    assert backend.slots._value == 4


async def test_broken_abort_does_not_hide_original_failure(tokenizer, config, payment):
    class BrokenAbort(FakeScheduler):
        async def abort(self, request_id):
            raise ConnectionError("Worker disappeared")

    backend = make_backend(tokenizer, config, BrokenAbort(fail_on_count=2))
    with pytest.raises(RuntimeError, match="Simulated engine failure"):
        await backend.evaluate(payment)
    assert backend.slots._value == 4
