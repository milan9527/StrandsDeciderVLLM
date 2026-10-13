"""Packed prefill -> original FP32 Pointer Head -> calibrated probabilities.

Independent of vLLM imports so the numerical core can be checked on CPU.
"""

import math

import torch
from safetensors.torch import load_file
from strands_decider.modeling import PointerHead, apply_temperature, masked_log_softmax


def load_pointer(path, hidden_size: int, pointer_dim: int) -> PointerHead:
    head = PointerHead(hidden_size, pointer_dim).to(dtype=torch.float32).eval()
    head.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    return head


@torch.inference_mode()
def score_packed(
    head: PointerHead,
    hidden: torch.Tensor,
    lengths: list[int],
    metadata: list[dict],
) -> list[torch.Tensor]:
    if hidden.ndim != 2 or len(lengths) != len(metadata) or not lengths:
        raise ValueError("Invalid packed hidden states or request metadata")
    if any(length <= 0 for length in lengths) or sum(lengths) > hidden.shape[0]:
        raise ValueError("Invalid prompt lengths")
    option_indices, answer_indices, counts, temperatures = [], [], [], []
    start = 0
    for length, info in zip(lengths, metadata, strict=True):
        positions = info["option_positions"]
        answer = info["answer_position"]
        temperature = info["temperature"]
        if info["prompt_length"] != length:
            raise ValueError("Partial/cached prefills cannot be scored by this adapter")
        if (
            not 2 <= len(positions) <= 255
            or any(type(p) is not int or not 0 <= p < length for p in positions)
            or type(answer) is not int
            or answer != length - 1
        ):
            raise ValueError("Invalid candidate or answer positions")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Invalid decision temperature")
        option_indices.append([start + p for p in positions])
        answer_indices.append(start + answer)
        counts.append(len(positions))
        temperatures.append(temperature)
        start += length
    width = max(counts)
    indices = torch.tensor(
        [row + [0] * (width - len(row)) for row in option_indices],
        device=hidden.device,
        dtype=torch.long,
    )
    answers = torch.tensor(answer_indices, device=hidden.device, dtype=torch.long)
    logits = head(hidden[answers].float(), hidden[indices].float())
    logits = apply_temperature(
        logits, torch.tensor(temperatures, device=hidden.device, dtype=torch.float32)
    )
    probabilities = masked_log_softmax(
        logits, torch.tensor(counts, device=hidden.device, dtype=torch.long)
    ).exp()
    return [row[:count] for row, count in zip(probabilities, counts, strict=True)]
