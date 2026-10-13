"""Async vLLM adapter: one scheduler request per independent question."""

import asyncio
import importlib.metadata
import logging
import math
import uuid
from dataclasses import dataclass
from time import perf_counter

from strands_decider.schema import SystemOneResponse, Usage
from transformers import AutoTokenizer

from . import VLLM_VERSION, register
from .artifacts import load_config
from .preprocessing import Preprocessor
from .security import request_id as http_request_id

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeSettings:
    model: str
    model_name: str = "strands-decider-2B-hobson-v19-vllm"
    max_num_seqs: int = 16
    max_num_batched_tokens: int = 8192
    gpu_memory_utilization: float = 0.55
    max_inflight_questions: int = 64
    strict_window: bool = True


class DecisionBackend:
    def __init__(
        self, engine, preprocessor: Preprocessor, params_factory, settings: RuntimeSettings
    ):
        self.engine = engine
        self.preprocessor = preprocessor
        self.params_factory = params_factory
        self.settings = settings
        self.slots = asyncio.Semaphore(settings.max_inflight_questions)

    async def _question(self, question):
        async with self.slots:
            request_id = uuid.uuid4().hex
            started = perf_counter()
            logger.info(
                "engine_submitted request_id=%s engine_request_id=%s input_tokens=%d",
                http_request_id.get(),
                request_id,
                len(question.token_ids),
            )
            params = self.params_factory(
                task="classify",
                skip_reading_prefix_cache=True,
                extra_kwargs={"decider": question.metadata()},
            )
            output = None
            try:
                async for result in self.engine.encode(
                    {"prompt_token_ids": list(question.token_ids)}, params, request_id
                ):
                    if result.finished:
                        output = result.outputs.data
            except BaseException as error:
                logger.warning(
                    "engine_interrupted request_id=%s engine_request_id=%s error_type=%s elapsed_ms=%.2f",
                    http_request_id.get(),
                    request_id,
                    type(error).__name__,
                    (perf_counter() - started) * 1000,
                )
                # Also clean up if validation/transport fails after scheduling.
                try:
                    await asyncio.wait_for(self.engine.abort(request_id), timeout=2)
                except Exception:
                    logger.exception("Failed to abort vLLM request %s", request_id)
                raise
            if output is None:
                raise RuntimeError("vLLM returned no final decision")
            probabilities = output.float().cpu().tolist()
            if (
                len(probabilities) != len(question.option_positions)
                or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities)
                or abs(sum(probabilities) - 1) > 1e-4
            ):
                raise RuntimeError("vLLM returned an invalid candidate probability vector")
            answer = self.preprocessor.answer(question, probabilities)
            logger.info(
                "engine_completed request_id=%s engine_request_id=%s elapsed_ms=%.2f",
                http_request_id.get(),
                request_id,
                (perf_counter() - started) * 1000,
            )
            return answer

    async def evaluate(self, request):
        started = perf_counter()
        prepared = await asyncio.to_thread(self.preprocessor.prepare, request)
        logger.info(
            "preprocessing_completed request_id=%s questions=%d elapsed_ms=%.2f",
            http_request_id.get(),
            len(prepared),
            (perf_counter() - started) * 1000,
        )
        tasks = [asyncio.create_task(self._question(q)) for q in prepared]
        try:
            answers = await asyncio.gather(*tasks)
        finally:
            # Failure, timeout, or client disconnect cancels every sibling question.
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        return SystemOneResponse(
            model=self.settings.model_name,
            answers={q.name: a for q, a in zip(prepared, answers, strict=True)},
            usage=Usage(
                input_tokens=sum(len(q.token_ids) for q in prepared),
                output_tokens=len(prepared),
            ),
        )

    async def check_health(self):
        await self.engine.check_health()

    def shutdown(self):
        self.engine.shutdown()


def create_backend(settings: RuntimeSettings) -> DecisionBackend:
    if importlib.metadata.version("vllm") != VLLM_VERSION:
        raise RuntimeError(f"Install the gpu extra with vLLM=={VLLM_VERSION}")
    from vllm import PoolingParams
    from vllm.config import PoolerConfig
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    if settings.max_num_seqs < 1 or settings.max_inflight_questions < 1:
        raise ValueError("Concurrency limits must be positive")
    config = load_config(settings.model)
    if settings.max_num_batched_tokens < config.max_length:
        raise ValueError("max_num_batched_tokens must cover one complete context window")
    register()
    args = AsyncEngineArgs(
        model=settings.model,
        runner="pooling",
        pooler_config=PoolerConfig(task="classify"),
        dtype="bfloat16",
        max_model_len=config.max_length,
        max_num_seqs=settings.max_num_seqs,
        max_num_batched_tokens=settings.max_num_batched_tokens,
        gpu_memory_utilization=settings.gpu_memory_utilization,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        enforce_eager=True,
        trust_remote_code=False,
        enable_log_requests=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(settings.model, local_files_only=True)
    preprocessor = Preprocessor(tokenizer, config, strict_window=settings.strict_window)
    engine = AsyncLLM.from_engine_args(args)
    return DecisionBackend(engine, preprocessor, PoolingParams, settings)
