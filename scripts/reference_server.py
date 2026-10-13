"""Serial official reference endpoint for behavior checks, not a tuned speed baseline."""

import argparse
import asyncio
from time import perf_counter

import uvicorn
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from strands_decider.infer import EngineConfig, SystemOneEngine
from strands_decider.modeling import (
    StrandsDeciderConfig,
    StrandsDeciderModel,
    config_path,
    load_head_state,
)
from strands_decider.schema import SystemOneRequest
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--port", type=int, default=8099)
    args = parser.parse_args()
    config = StrandsDeciderConfig.from_json(config_path(args.checkpoint))
    config.base_model, config.base_revision = args.base, None
    torso = StrandsDeciderModel._load_torso(config, "cpu", None)
    if config.use_lora:
        torso = PeftModel.from_pretrained(torso, args.checkpoint + "/lora", is_trainable=False)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = StrandsDeciderModel(config, torso, tokenizer)
    model.head.load_state_dict(load_head_state(args.checkpoint))
    engine = SystemOneEngine(
        model,
        EngineConfig(device=args.device, strict_window=True, use_prefix_cache=False, max_batch=8),
    )
    app = FastAPI()
    lock = asyncio.Lock()

    @app.get("/health")
    async def health():
        return {"status": "ok", "engine": "official", "device": args.device}

    @app.post("/v1/systemone")
    async def decision(request: SystemOneRequest):
        started = perf_counter()
        async with lock:
            try:
                # Keep ownership until the thread completes, including on disconnect.
                # This reference process is deliberately serial.
                task = asyncio.create_task(asyncio.to_thread(engine.evaluate, request))
                try:
                    response = await asyncio.shield(task)
                except asyncio.CancelledError:
                    await task
                    raise
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
        return {
            **response.model_dump(),
            "latency_ms": round((perf_counter() - started) * 1000, 2),
        }

    uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1, access_log=False)


if __name__ == "__main__":
    main()
