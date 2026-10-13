"""Merge PEFT with the official loader; never modify source checkpoints."""

import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import save_file
from strands_decider.modeling import (
    StrandsDeciderConfig,
    StrandsDeciderModel,
    config_path,
    load_head_state,
)
from transformers import AutoTokenizer

from . import ARCHITECTURE, UPSTREAM_REVISION, VLLM_VERSION
from .artifacts import sha256, verify_export


def export_model(checkpoint: Path, base: Path, output: Path) -> dict:
    checkpoint, base, output = checkpoint.resolve(), base.resolve(), output.absolute()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    config = StrandsDeciderConfig.from_json(config_path(str(checkpoint)))
    if config.head_type != "pointer":
        raise ValueError("Only Pointer Head checkpoints are supported")
    base_config = json.loads((base / "config.json").read_text())
    if base_config.get("model_type") not in {"qwen3_5", "qwen3_5_text"}:
        raise ValueError("This adapter targets Qwen3.5 text weights")
    # Check the decision weights before allocating the backbone.
    head = {
        key: value.float().contiguous() for key, value in load_head_state(str(checkpoint)).items()
    }
    source_config = Path(config_path(str(checkpoint)))
    config.base_model = str(base)
    config.base_revision = None
    config.torch_dtype = "bfloat16"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        torso = StrandsDeciderModel._load_torso(config, "cpu", "eager")
        adapter_parameters = 0
        if config.use_lora:
            torso = PeftModel.from_pretrained(torso, str(checkpoint / "lora"), is_trainable=False)
            adapter_parameters = sum(
                p.numel() for name, p in torso.named_parameters() if "lora_" in name
            )
            if not adapter_parameters:
                raise ValueError("No LoRA parameters loaded")
            torso = torso.merge_and_unload(safe_merge=True)
        if any("lora_" in name for name, _ in torso.named_parameters()):
            raise ValueError("Unmerged adapter weights remain")
        torso.eval()
        torso.save_pretrained(temporary, safe_serialization=True, max_shard_size="2GB")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if config.force_bos:
            from strands_decider.modeling import ensure_bos

            ensure_bos(tokenizer)
        tokenizer.save_pretrained(temporary)

        # AutoModel saves bare decoder keys. vLLM's pooling adapter maps them to model.*.
        hf_path = temporary / "config.json"
        hf = json.loads(hf_path.read_text())
        hf.update(
            architectures=[ARCHITECTURE],
            model_type="qwen3_5_text",
            dtype="bfloat16",
            torch_dtype="bfloat16",
        )
        hf_path.write_text(json.dumps(hf, indent=2) + "\n")
        decision_dir = temporary / "decider"
        decision_dir.mkdir()
        save_file(head, decision_dir / "head.safetensors")
        config.use_lora = False
        config.base_model = "."
        (decision_dir / "config.json").write_text(config.to_json() + "\n")
        manifest = {
            "format_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "upstream_revision": UPSTREAM_REVISION,
            "vllm_version": VLLM_VERSION,
            "source_config_sha256": sha256(source_config),
            "source_base_config_sha256": sha256(base / "config.json"),
            "source_adapter_sha256": {
                p.name: sha256(p) for p in sorted((checkpoint / "lora").glob("*")) if p.is_file()
            },
            "merged_adapter_parameters": adapter_parameters,
            "torch_version": torch.__version__,
            "decider_sha256": {
                name: sha256(decision_dir / name) for name in ("head.safetensors", "config.json")
            },
            "files_sha256": {
                str(p.relative_to(temporary)): sha256(p)
                for p in sorted(temporary.rglob("*"))
                if p.is_file()
            },
        }
        (decision_dir / "export.json").write_text(json.dumps(manifest, indent=2) + "\n")
        verify_export(temporary)
        temporary.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
