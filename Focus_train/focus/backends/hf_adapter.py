from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..attention import FocusTokenAttention
from ..kv_pruner import FocusTokenKVPruner
from ..utils import ensure_token, set_trainable


@dataclass
class FocusTokenForwardOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor
    focus_states: torch.Tensor
    compressed_states: torch.Tensor
    past_key_values: Optional[Tuple]


class HFAdapter(nn.Module):
    """HuggingFace backend for the focus-token pipeline."""

    def __init__(self, model: AutoModelForCausalLM, tokenizer, config: Dict) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        hidden_size = model.config.hidden_size
        num_heads = getattr(model.config, "num_attention_heads", 16)
        dropout = getattr(model.config, "attention_dropout", 0.0)
        self.focus_attention = FocusTokenAttention(hidden_size, num_heads, dropout=dropout)
        self.pruner = FocusTokenKVPruner()

        self._focus_token = config.get("special_tokens", {}).get("focus_token", "<|focus|>")
        self._focus_token_id = ensure_token(tokenizer, self._focus_token)
        self._register_embedding_hook()

    @classmethod
    def from_pretrained(cls, model_name: str, config: Dict, device: Optional[str] = None) -> "HFAdapter":
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        ensure_token(tokenizer, config.get("special_tokens", {}).get("focus_token", "<|focus|>"))
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        )
        model.resize_token_embeddings(len(tokenizer))
        adapter = cls(model, tokenizer, config)
        adapter.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        return adapter

    # -------------------- Freezing --------------------
    def freeze_base(self) -> None:
        set_trainable(self.model, False)
        set_trainable(self.focus_attention, True)
        self._enable_focus_embedding_training()

    def _enable_focus_embedding_training(self) -> None:
        embedding = self.model.get_input_embeddings()
        embedding.weight.requires_grad = True

    def _register_embedding_hook(self) -> None:
        embedding = self.model.get_input_embeddings()
        mask = torch.zeros(embedding.weight.shape[0], device=embedding.weight.device)
        mask[self._focus_token_id] = 1.0

        def _mask_grad(grad: torch.Tensor) -> torch.Tensor:
            return grad * mask.unsqueeze(-1)

        embedding.weight.register_hook(_mask_grad)

    # -------------------- Forward --------------------
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
    ) -> FocusTokenForwardOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True,
            return_dict=True,
            **model_kwargs,
        )

        hidden_states = outputs.hidden_states[-1]
        focus_states, token_states = self._split_hidden_states(hidden_states, focus_mask, attention_mask)

        stepwise_mask = None

        expanded_causal = None
        if causal_mask is not None:
            expanded_causal = causal_mask.unsqueeze(1).to(hidden_states.dtype)
            expanded_causal = torch.where(
                expanded_causal > 0,
                torch.zeros_like(expanded_causal),
                torch.full_like(expanded_causal, torch.finfo(hidden_states.dtype).min),
            )

        compressed, present = self.focus_attention(
            focus_states,
            token_states,
            attention_mask=expanded_causal,
            stepwise_mask=stepwise_mask,
            past_key_value=None,
            use_cache=False,
        )

        logits = outputs.logits
        return FocusTokenForwardOutput(
            logits=logits,
            hidden_states=hidden_states,
            focus_states=focus_states,
            compressed_states=compressed,
            past_key_values=outputs.past_key_values,
        )

    def apply_focus_attention(
        self,
        hidden_states: torch.Tensor,
        focus_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        block_map,
    ) -> torch.Tensor:
        focus_states, token_states = self._split_hidden_states(hidden_states, focus_mask, attention_mask)
        stepwise_mask = None
        compressed, _ = self.focus_attention(
            focus_states,
            token_states,
            stepwise_mask=stepwise_mask,
        )
        return compressed

    def replace_or_merge_kv(
        self,
        past_key_values: Tuple,
        focus_mask: torch.Tensor,
        new_past: Optional[Tuple] = None,
    ) -> Tuple:
        pruned = self.pruner.prune(past_key_values, focus_mask.squeeze(0))
        if new_past is not None:
            pruned = self.pruner.merge(pruned, new_past)
        return pruned

    # -------------------- Utilities --------------------
    def _split_hidden_states(
        self,
        hidden_states: torch.Tensor,
        focus_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.size(0)
        focus_slices: List[torch.Tensor] = []
        token_slices: List[torch.Tensor] = []
        token_masks = (~focus_mask) & (attention_mask > 0)
        for i in range(batch_size):
            focus_rows = hidden_states[i][focus_mask[i]]
            token_rows = hidden_states[i][token_masks[i]]
            focus_slices.append(focus_rows)
            token_slices.append(token_rows)
        focus_tensor = pad_sequence(focus_slices, batch_first=True)
        token_tensor = pad_sequence(token_slices, batch_first=True)
        return focus_tensor, token_tensor

    def generate_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Optional[Tuple] = None,
        **kwargs,
    ):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            **kwargs,
        )
