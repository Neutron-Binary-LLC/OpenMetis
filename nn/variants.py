from dataclasses import dataclass
from typing import Dict, Any
from .block import MathConfig

def get_tiny_config():
    return MathConfig(
        dim=128,
        n_heads=4,
        max_loop_iters=4,
        prelude_layers=1,
        coda_layers=1,
        n_experts=4,
        workspace_dim=64,
        lora_rank=8
    )

def get_small_config():
    return MathConfig(
        dim=512,
        n_heads=8,
        max_loop_iters=8,
        prelude_layers=2,
        coda_layers=2,
        n_experts=8,
        workspace_dim=128,
        lora_rank=16
    )

def get_medium_config():
    return MathConfig(
        dim=1024,
        n_heads=16,
        max_loop_iters=12,
        prelude_layers=4,
        coda_layers=4,
        n_experts=16,
        workspace_dim=256,
        lora_rank=32
    )

def get_large_config():
    return MathConfig(
        dim=2048,
        n_heads=32,
        max_loop_iters=16,
        prelude_layers=6,
        coda_layers=6,
        n_experts=32,
        workspace_dim=512,
        lora_rank=64
    )

VARIANTS = {
    "tiny": get_tiny_config(),
    "small": get_small_config(),
    "medium": get_medium_config(),
    "large": get_large_config(),
}
