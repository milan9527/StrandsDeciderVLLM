"""Lazy plugin registration: importing this package does not initialize CUDA."""

__version__ = "0.1.0"
VLLM_VERSION = "0.31.0"
UPSTREAM_REVISION = "63d24ae286e50105fba0bd1db15b0e243aea4650"
ARCHITECTURE = "StrandsDeciderForPooling"


def register() -> None:
    from vllm import ModelRegistry

    ModelRegistry.register_model(ARCHITECTURE, "decider_vllm.vllm_model:StrandsDeciderForPooling")
