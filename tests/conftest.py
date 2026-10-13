import json
from pathlib import Path

import pytest
from strands_decider.modeling import StrandsDeciderConfig
from strands_decider.schema import SystemOneRequest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


@pytest.fixture
def tokenizer():
    raw = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1}, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]", model_max_length=4096
    )


@pytest.fixture
def config():
    return StrandsDeciderConfig(
        base_model="unused",
        head_type="pointer",
        use_lora=False,
        max_length=4096,
        temperature=0.9,
        temperature_by_kind={"choice": 0.734189596436441, "score": 1.3},
        ordinal_smoothing=0.1,
    )


@pytest.fixture
def payment():
    path = Path(__file__).parents[1] / "examples/payment.json"
    return SystemOneRequest.model_validate(json.loads(path.read_text()))
