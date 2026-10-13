"""Reuse pinned upstream rendering/fitting with request-local mutable state."""

from dataclasses import dataclass
from types import SimpleNamespace

from strands_decider.infer import EngineConfig, SystemOneEngine, _option_token_index, _to_answer
from strands_decider.modeling import StrandsDeciderConfig
from strands_decider.prompting import RenderedQuestion, render_question, render_state
from strands_decider.schema import SystemOneRequest


@dataclass(frozen=True)
class PreparedQuestion:
    name: str
    token_ids: tuple[int, ...]
    option_positions: tuple[int, ...]
    rendered: RenderedQuestion
    temperature: float

    def metadata(self) -> dict:
        return {
            "option_positions": list(self.option_positions),
            "answer_position": len(self.token_ids) - 1,
            "temperature": self.temperature,
            "prompt_length": len(self.token_ids),
        }


class Preprocessor:
    def __init__(
        self,
        tokenizer,
        config: StrandsDeciderConfig,
        *,
        strict_window: bool = True,
        max_questions: int = 32,
        question_chunk_size: int = 8,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.strict_window = strict_window
        self.max_questions = max_questions
        self.question_chunk_size = question_chunk_size

    def prepare(self, request: SystemOneRequest) -> list[PreparedQuestion]:
        if request.images:
            raise ValueError("This adapter supports text only; images are not accepted")
        if len(request.questions) > self.max_questions:
            raise ValueError(f"At most {self.max_questions} questions per request")
        names = list(request.questions)
        rendered = [render_question(q) for q in request.questions.values()]
        # _chunks/_fit write _last_offsets. A new lightweight context per request
        # preserves upstream tokenization without sharing that field across requests.
        context = object.__new__(SystemOneEngine)
        context.model = SimpleNamespace(config=self.config)
        context.cfg = EngineConfig(
            strict_window=self.strict_window,
            max_batch=self.question_chunk_size,
            use_prefix_cache=False,
        )
        context.tok = self.tokenizer
        prepared = []
        for chunk, state_ids, question_ids, offsets in context._chunks(
            render_state(request.state), rendered
        ):
            rows = zip(names[chunk], rendered[chunk], question_ids, offsets, strict=True)
            for name, question, tokens, spans in rows:
                positions = _option_token_index(spans, question.option_spans, 0)
                prepared.append(
                    PreparedQuestion(
                        name=name,
                        token_ids=tuple(state_ids + tokens),
                        option_positions=tuple(len(state_ids) + p for p in positions),
                        rendered=question,
                        temperature=self.config.temperature_by_kind.get(
                            question.kind, self.config.temperature
                        ),
                    )
                )
        return prepared

    def answer(self, question: PreparedQuestion, probabilities: list[float]):
        return _to_answer(
            question.rendered, probabilities, ordinal_smoothing=self.config.ordinal_smoothing
        )
