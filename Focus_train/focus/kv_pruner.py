from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Union

import torch

PastKeyValue = Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
LayerPast = Union[PastKeyValue, Tuple[PastKeyValue, PastKeyValue]]


@dataclass
class KVStructure:
    has_cross_attention: bool = False
    indices: Sequence[int] | None = None


def _select_indices(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.index_select(tensor, dim=-2, index=indices)


def prune_past_key_values(
    past_key_values: Tuple[LayerPast, ...],
    keep_indices: torch.Tensor,
) -> Tuple[LayerPast, ...]:
    """Keep only entries at `keep_indices` along the sequence dimension of the KV cache."""
    pruned: List[LayerPast] = []
    for layer_past in past_key_values:
        if isinstance(layer_past, tuple) and len(layer_past) in (2, 4):
            pruned.append(_prune_tuple(layer_past, keep_indices))
        else:
            raise TypeError(f"Unsupported past key value structure: {type(layer_past)}")
    return tuple(pruned)


def _prune_tuple(past: Tuple[torch.Tensor, ...], keep_indices: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    buckets: List[torch.Tensor] = []
    for tensor in past:
        if tensor.dim() >= 3:
            buckets.append(_select_indices(tensor, keep_indices))
        else:
            buckets.append(tensor)
    return tuple(buckets)


class FocusTokenKVPruner:
    """Normalizes model-specific past_key_values operations for arbitrary decoder stacks."""

    def __init__(self, structure: KVStructure | None = None) -> None:
        self.structure = structure or KVStructure()

    def gather_focus_indices(self, focus_mask: torch.Tensor) -> torch.Tensor:
        if focus_mask.dim() != 1:
            raise ValueError("focus_mask must be 1-D for gather_focus_indices")
        indices = torch.nonzero(focus_mask, as_tuple=False).flatten()
        if indices.numel() == 0:
            raise ValueError("No focus-token positions found to keep in KV cache")
        return indices

    def prune(self, past_key_values: Tuple[LayerPast, ...], focus_mask: torch.Tensor) -> Tuple[LayerPast, ...]:
        keep_indices = self.gather_focus_indices(focus_mask)
        return prune_past_key_values(past_key_values, keep_indices)

    def merge(self, past_key_values: Tuple[LayerPast, ...], new_past: Tuple[LayerPast, ...]) -> Tuple[LayerPast, ...]:
        if len(past_key_values) != len(new_past):
            raise ValueError("Layer count mismatch during KV merge")
        merged: List[LayerPast] = []
        for old, new in zip(past_key_values, new_past):
            merged.append(self._concat_layer(old, new))
        return tuple(merged)

    @staticmethod
    def _concat_layer(old: LayerPast, new: LayerPast) -> LayerPast:
        if isinstance(old, tuple) and isinstance(new, tuple) and len(old) == len(new):
            return tuple(torch.cat([o, n], dim=-2) if o.dim() >= 3 else n for o, n in zip(old, new))
        raise TypeError("Unsupported layer structure for concatenation")
