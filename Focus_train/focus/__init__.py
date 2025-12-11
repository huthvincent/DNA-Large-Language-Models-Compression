"""Focus-token compression package for BioNeMo EVO-2."""

from .attention import FocusTokenAttention
from .token_injection import FocusTokenInserter, FOCUS_TOKEN
from .kv_pruner import prune_past_key_values, FocusTokenKVPruner
from .adapter import LLMAdapter
from .train import train_focus_tokens
from .infer import generate_with_focus_tokens
from .metrics import MetricsCollector

__all__ = [
    "FocusTokenAttention",
    "FocusTokenInserter",
    "FOCUS_TOKEN",
    "prune_past_key_values",
    "FocusTokenKVPruner",
    "LLMAdapter",
    "train_focus_tokens",
    "generate_with_focus_tokens",
    "MetricsCollector",
]
