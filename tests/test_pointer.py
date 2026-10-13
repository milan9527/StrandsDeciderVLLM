import pytest
import torch
from strands_decider.modeling import PointerHead, apply_temperature, masked_log_softmax

from decider_vllm.pointer import score_packed


def test_packed_matches_individual_official_heads_and_request_reordering():
    torch.manual_seed(37)
    head = PointerHead(16, 8).eval()
    lengths = [11, 7, 19]
    hidden = torch.randn(sum(lengths), 16)
    options = [[1, 5], [0, 3, 5], [2, 3, 7, 12, 16]]
    temperatures = [0.73, 0.91, 1.32]
    metadata = [
        dict(prompt_length=n, answer_position=n - 1, option_positions=p, temperature=t)
        for n, p, t in zip(lengths, options, temperatures, strict=True)
    ]
    packed = score_packed(head, hidden, lengths, metadata)
    sequences = hidden.split(lengths)
    for sequence, positions, temp, actual in zip(
        sequences, options, temperatures, packed, strict=True
    ):
        logits = head(sequence[-1:][None, 0], sequence[positions][None, :])
        expected = masked_log_softmax(
            apply_temperature(logits, temp), torch.tensor([len(positions)])
        ).exp()[0]
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    order = [2, 0, 1]
    reordered = score_packed(
        head,
        torch.cat([sequences[i] for i in order]),
        [lengths[i] for i in order],
        [metadata[i] for i in order],
    )
    for i, actual in zip(order, reordered, strict=True):
        torch.testing.assert_close(actual, packed[i])


@pytest.mark.parametrize(
    "change",
    [
        {"option_positions": [1, 99]},
        {"option_positions": [1]},
        {"answer_position": 3},
        {"prompt_length": 99},
        {"temperature": 0},
        {"temperature": float("nan")},
    ],
)
def test_reject_corrupt_or_partial_positions(change):
    info = dict(prompt_length=8, answer_position=7, option_positions=[1, 3], temperature=1)
    info.update(change)
    with pytest.raises(ValueError):
        score_packed(PointerHead(4, 2), torch.randn(8, 4), [8], [info])


def test_temperature_stays_fp32_under_bfloat16_default():
    head = PointerHead(8, 4).float().eval()
    hidden = torch.randn(10, 8, dtype=torch.float32)
    metadata = [
        {
            "prompt_length": 10,
            "answer_position": 9,
            "option_positions": [1, 3, 6],
            "temperature": 0.734189596436441,
        }
    ]
    expected = score_packed(head, hidden, [10], metadata)[0]
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        actual = score_packed(head, hidden, [10], metadata)[0]
    finally:
        torch.set_default_dtype(previous)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
