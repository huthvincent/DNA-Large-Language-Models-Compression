from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

if __package__ is None or __package__ == "":
    package_root = os.path.dirname(os.path.dirname(__file__))
    if package_root not in sys.path:
        sys.path.append(package_root)
    from focus.utils import FocusPlan, ensure_token  # type: ignore
else:  # pragma: no cover
    from .utils import FocusPlan, ensure_token

FOCUS_TOKEN = "<|focus|>"


@dataclass
class FocusTokenInjectionOutput:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    causal_mask: torch.Tensor
    focus_mask: torch.Tensor
    block_map: List[List[Tuple[int, int]]]


@dataclass
class InferenceInsertion:
    focus_token_ids: List[int]
    focus_token_positions: List[int]
    needs_pruning: bool
    block_index: int


@dataclass
class InferenceState:
    tokens_in_block: int = 0
    block_index: int = 0
    tokens_since_focus: int = 0
    total_tokens: int = 0


class FocusTokenInserter:
    """Handles focus token insertion and block-aware masking for training/inference."""

    def __init__(self, config: Dict, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.plan = FocusPlan(
            block_len=config.get("block_len", config.get("insert_every_n", 512)),
            focus_tokens_per_block=config.get("focus_tokens_per_block", 1),
            insert_every_n=config.get("insert_every_n", config.get("block_len", 512)),
            condense_ratio=config.get("condense_ratio", 64),
        )
        special_tokens = config.get("special_tokens", {})
        focus_token = special_tokens.get("focus_token", FOCUS_TOKEN)
        self.focus_token = focus_token
        if "focus_token_id" in special_tokens:
            self.focus_token_id = int(special_tokens["focus_token_id"])
        else:
            self.focus_token_id = ensure_token(tokenizer, focus_token)

    # -------------------- Training --------------------
    def prepare_training_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], FocusTokenInjectionOutput]:
        """Insert focus tokens into a batch of sequences and build block-aware masks."""

        input_ids: torch.Tensor = batch["input_ids"]
        attention_mask: torch.Tensor = batch.get("attention_mask")
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        batch_size, seq_len = input_ids.shape
        max_extra = self.plan.blocks_for_length(seq_len) * self.plan.focus_tokens_per_block
        padded_len = seq_len + max_extra

        new_input_ids = input_ids.new_full((batch_size, padded_len), self.tokenizer.pad_token_id)
        new_attention_mask = attention_mask.new_zeros((batch_size, padded_len))
        focus_mask = torch.zeros((batch_size, padded_len), dtype=torch.bool, device=input_ids.device)
        causal_masks: List[torch.Tensor] = []
        block_maps: List[List[Tuple[int, int]]] = []

        for i in range(batch_size):
            seq = input_ids[i]
            mask = attention_mask[i]
            valid_len = int(mask.sum().item())
            tokens = seq[:valid_len].tolist()

            augmented, focus_positions, block_map = self._augment_sequence(tokens)
            block_maps.append(block_map)

            new_input_ids[i, : len(augmented)] = torch.tensor(augmented, dtype=input_ids.dtype, device=input_ids.device)
            new_attention_mask[i, : len(augmented)] = 1
            focus_mask[i, focus_positions] = True

            causal_masks.append(self._build_causal_mask(len(augmented), focus_positions, block_map, device=input_ids.device))

        causal_mask = torch.stack(causal_masks, dim=0)

        updated_batch = {
            **batch,
            "input_ids": new_input_ids,
            "attention_mask": new_attention_mask,
        }

        return updated_batch, FocusTokenInjectionOutput(
            input_ids=new_input_ids,
            attention_mask=new_attention_mask,
            causal_mask=causal_mask,
            focus_mask=focus_mask,
            block_map=block_maps,
        )

    # -------------------- Inference --------------------
    def init_inference_state(self) -> InferenceState:
        return InferenceState()

    def observe_token(self, state: InferenceState) -> None:
        state.tokens_in_block += 1
        state.tokens_since_focus += 1
        state.total_tokens += 1

    def maybe_insert_focus_tokens(self, state: InferenceState) -> Optional[InferenceInsertion]:
        if state.tokens_since_focus < self.plan.insert_every_n and state.tokens_in_block < self.plan.block_len:
            return None

        focus_positions = [state.total_tokens + i for i in range(self.plan.focus_tokens_per_block)]
        focus_ids = [self.focus_token_id] * self.plan.focus_tokens_per_block

        state.tokens_since_focus = 0
        state.tokens_in_block = 0
        state.block_index += 1
        state.total_tokens += self.plan.focus_tokens_per_block

        return InferenceInsertion(
            focus_token_ids=focus_ids,
            focus_token_positions=focus_positions,
            needs_pruning=True,
            block_index=state.block_index,
        )

    def build_stream_mask(self, past_focus_mask: torch.Tensor, current_length: int) -> torch.Tensor:
        """Create an attention mask for streaming inference with known focus-token layout."""
        block_ids = self._block_assignments_from_mask(past_focus_mask[:current_length])
        focus_positions = torch.nonzero(past_focus_mask[:current_length], as_tuple=False).flatten().tolist()
        return self._build_causal_mask(current_length, focus_positions, block_ids, device=past_focus_mask.device)

    # -------------------- Internals --------------------
    def _augment_sequence(self, tokens: List[int]) -> Tuple[List[int], List[int], List[Tuple[int, int]]]:
        augmented: List[int] = []
        focus_positions: List[int] = []
        block_map: List[Tuple[int, int]] = []

        cursor = 0
        block_id = 0
        while cursor < len(tokens):
            next_cursor = min(cursor + self.plan.block_len, len(tokens))
            block_tokens = tokens[cursor:next_cursor]
            block_start = len(augmented)
            augmented.extend(block_tokens)
            block_end = len(augmented)
            block_map.append((block_start, block_end))

            for _ in range(self.plan.focus_tokens_per_block):
                focus_positions.append(len(augmented))
                augmented.append(self.focus_token_id)

            cursor = next_cursor
            block_id += 1

        return augmented, focus_positions, block_map

    def _build_causal_mask(
        self,
        length: int,
        focus_positions: List[int],
        block_map: List[Tuple[int, int]] | List[List[Tuple[int, int]]],
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(block_map[0], tuple):
            block_ranges = block_map  # type: ignore
        else:
            block_ranges = block_map[0]  # type: ignore

        block_ids = torch.full((length,), -1, dtype=torch.long, device=device)
        for idx, (start, end) in enumerate(block_ranges):
            block_ids[start:end] = idx
            focus_start = end
            focus_end = min(end + self.plan.focus_tokens_per_block, length)
            block_ids[focus_start:focus_end] = idx

        focus_mask = torch.zeros(length, dtype=torch.bool, device=device)
        if focus_positions:
            focus_mask[focus_positions] = True

        causal = torch.tril(torch.ones((length, length), dtype=torch.bool, device=device))

        for row in range(length):
            row_block = int(block_ids[row].item())
            for col in range(length):
                if not causal[row, col]:
                    continue
                col_block = int(block_ids[col].item())
                if col_block > row_block:
                    causal[row, col] = False
                    continue
                if focus_mask[row] and col_block != row_block:
                    causal[row, col] = False
        return causal

    def _block_assignments_from_mask(self, focus_mask: torch.Tensor) -> List[Tuple[int, int]]:
        block_ranges: List[Tuple[int, int]] = []
        block_start = 0
        block_index = 0
        for idx in range(focus_mask.shape[0]):
            if focus_mask[idx]:
                block_ranges.append((block_start, idx))
                block_start = idx + 1
                block_index += 1
        if block_start < focus_mask.shape[0]:
            block_ranges.append((block_start, focus_mask.shape[0]))
        return block_ranges
