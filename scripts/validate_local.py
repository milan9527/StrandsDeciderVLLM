"""Real-weight CPU checks; this is not a vLLM GPU or performance benchmark."""

import argparse
import gc
import json
from datetime import UTC, datetime
from pathlib import Path

import torch
from peft import PeftModel
from strands_decider.infer import EngineConfig, SystemOneEngine
from strands_decider.modeling import (
    StrandsDeciderConfig,
    StrandsDeciderModel,
    config_path,
    load_head_state,
)
from strands_decider.schema import SystemOneRequest, SystemOneResponse, Usage
from transformers import AutoTokenizer

from decider_vllm.artifacts import load_config, verify_export
from decider_vllm.evaluation import compare_responses, load_cases
from decider_vllm.pointer import load_pointer, score_packed
from decider_vllm.preprocessing import Preprocessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=0.02)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    cases = load_cases(args.corpus)
    example = Path(__file__).parents[1] / "examples/payment.json"
    cases.append({"id": "mixed-primitives", "request": json.loads(example.read_text())})
    requests = [SystemOneRequest.model_validate(c["request"]) for c in cases]
    integrity = verify_export(args.model)
    config = StrandsDeciderConfig.from_json(config_path(str(args.checkpoint)))
    config.base_model, config.base_revision = str(args.base.resolve()), None
    torso = StrandsDeciderModel._load_torso(config, "cpu", "eager")
    if config.use_lora:
        torso = PeftModel.from_pretrained(torso, args.checkpoint / "lora", is_trainable=False)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    original = StrandsDeciderModel(config, torso, tokenizer)
    original.head.load_state_dict(load_head_state(str(args.checkpoint)))
    engine = SystemOneEngine(
        original, EngineConfig(device="cpu", strict_window=True, use_prefix_cache=False)
    )
    references = []
    for case, request in zip(cases, requests, strict=True):
        references.append(engine.evaluate(request).model_dump())
        print("reference", case["id"], flush=True)
    del original, torso, engine
    gc.collect()

    config = load_config(args.model)
    config.base_model = str(args.model.resolve())
    merged = StrandsDeciderModel._load_torso(config, "cpu", "eager").float().eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    processor = Preprocessor(tokenizer, config)
    head = load_pointer(
        args.model / "decider/head.safetensors", merged.config.hidden_size, config.pointer_dim
    )
    rows = []
    with torch.inference_mode():
        for case, request, reference in zip(cases, requests, references, strict=True):
            prepared = processor.prepare(request)
            # Real Transformer batched forward, then pack unpadded states exactly
            # as the vLLM pooler receives them. No synthetic logits or answers.
            lengths = [len(q.token_ids) for q in prepared]
            width = max(lengths)
            ids = torch.tensor(
                [
                    list(q.token_ids) + [tokenizer.pad_token_id] * (width - len(q.token_ids))
                    for q in prepared
                ]
            )
            mask = torch.arange(width)[None, :] < torch.tensor(lengths)[:, None]
            hidden = merged(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            packed = torch.cat([h[:n] for h, n in zip(hidden, lengths, strict=True)])
            probabilities = score_packed(head, packed, lengths, [q.metadata() for q in prepared])
            response = SystemOneResponse(
                model="merged-cpu-validation",
                answers={
                    q.name: processor.answer(q, row.tolist())
                    for q, row in zip(prepared, probabilities, strict=True)
                },
                usage=Usage(input_tokens=sum(lengths), output_tokens=len(prepared)),
            ).model_dump()
            comparison = compare_responses(reference, response, tolerance=args.tolerance)
            rows.append({"id": case["id"], **comparison})
            print("merged", case["id"], "passed:", comparison["passed"], flush=True)
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "Real v19 weights, official CPU reference vs merged CPU backbone and custom Pointer Head",
        "gpu_vllm_executed": False,
        "precision": "FP32 CPU execution; merged backbone weights exported in BF16",
        "tolerance": args.tolerance,
        "integrity": integrity,
        "passed": all(r["passed"] for r in rows),
        "passed_cases": sum(r["passed"] for r in rows),
        "cases": len(rows),
        "results": rows,
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
