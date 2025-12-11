from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

import torch
from torch import nn

if __package__ is None or __package__ == "":
    package_root = os.path.dirname(os.path.dirname(__file__))
    if package_root not in sys.path:
        sys.path.append(package_root)
    from focus.backends import BioNeMoAdapter, HFAdapter  # type: ignore
else:  # pragma: no cover
    from .backends import BioNeMoAdapter, HFAdapter


class LLMAdapter(nn.Module):
    """Facade that dispatches to backend-specific adapters."""

    _BACKENDS = {
        "hf": HFAdapter,
    }
    if BioNeMoAdapter is not None:  # pragma: no cover - optional backend
        _BACKENDS["bionemo"] = BioNeMoAdapter

    def __init__(self, backend: nn.Module) -> None:
        super().__init__()
        self.adapter = backend

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        config: Dict,
        device: Optional[str] = None,
    ) -> "LLMAdapter":
        backend_key = config.get("backend", "hf").lower()
        if backend_key not in cls._BACKENDS:
            raise ValueError(
                f"Unsupported backend '{backend_key}'. Available backends: {sorted(cls._BACKENDS)}"
            )

        backend_cls = cls._BACKENDS[backend_key]
        backend = backend_cls.from_pretrained(model_name_or_path, config, device=device)
        return cls(backend)

    @property
    def tokenizer(self):  # pragma: no cover - passthrough
        return self.adapter.tokenizer

    @property
    def focus_attention(self):  # pragma: no cover - passthrough
        return self.adapter.focus_attention

    def freeze_base(self) -> None:
        return self.adapter.freeze_base()

    def forward_with_focus_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        focus_mask: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        block_map=None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = True,
        **model_kwargs,
    ):
        return self.adapter.forward_with_focus_tokens(
            input_ids=input_ids,
            attention_mask=attention_mask,
            focus_mask=focus_mask,
            causal_mask=causal_mask,
            block_map=block_map,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **model_kwargs,
        )

    def apply_focus_attention(
        self,
        hidden_states: torch.Tensor,
        focus_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        block_map,
    ) -> torch.Tensor:
        return self.adapter.apply_focus_attention(hidden_states, focus_mask, attention_mask, block_map)

    def replace_or_merge_kv(
        self,
        past_key_values: Tuple,
        focus_mask: torch.Tensor,
        new_past: Optional[Tuple] = None,
    ) -> Tuple:
        return self.adapter.replace_or_merge_kv(past_key_values, focus_mask, new_past)

    def generate_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Optional[Tuple] = None,
        **kwargs,
    ):
        return self.adapter.generate_step(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )

    def __getattr__(self, item):  # pragma: no cover - passthrough for convenience
        if item in {"adapter", "_modules"}:
            return super().__getattr__(item)
        return getattr(self.adapter, item)
