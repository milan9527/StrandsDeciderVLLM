"""Validate the locally exported model bundle before starting an engine."""

import hashlib
import json
import math
from pathlib import Path

from strands_decider.modeling import StrandsDeciderConfig

from . import ARCHITECTURE, UPSTREAM_REVISION


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(model: str | Path) -> StrandsDeciderConfig:
    root = Path(model)
    config = StrandsDeciderConfig.from_json(str(root / "decider/config.json"))
    hf = json.loads((root / "config.json").read_text())
    manifest = json.loads((root / "decider/export.json").read_text())
    if hf.get("architectures") != [ARCHITECTURE] or hf.get("model_type") != "qwen3_5_text":
        raise ValueError("Expected an exported StrandsDeciderForPooling Qwen3.5 text model")
    if config.head_type != "pointer" or config.use_lora:
        raise ValueError("Only Pointer Head checkpoints with merged LoRA are supported")
    if manifest.get("upstream_revision") != UPSTREAM_REVISION:
        raise ValueError("This export uses a different Strands Decider source revision")
    if config.max_length < 1:
        raise ValueError("max_length must be positive")
    temperatures = [config.temperature, *config.temperature_by_kind.values()]
    if any(not math.isfinite(t) or t <= 0 for t in temperatures):
        raise ValueError("Temperatures must be finite and positive")
    for name in ("head.safetensors", "config.json"):
        expected = manifest["decider_sha256"][name]
        if sha256(root / "decider" / name) != expected:
            raise ValueError(f"Export integrity check failed: decider/{name}")
    return config


def verify_export(model: str | Path) -> dict:
    """Full, explicit weight verification; startup only hashes the small decision files."""
    root = Path(model)
    load_config(root)
    manifest = json.loads((root / "decider/export.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != expected:
            raise ValueError(f"Export integrity check failed: {name}")
    return {"status": "ok", "files_verified": len(manifest["files_sha256"])}
