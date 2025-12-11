from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocusTokenAttention(nn.Module):
    """Dedicated attention that maps focus-token queries over block token keys/values."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim**-0.5
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    def _shape(self, tensor: torch.Tensor, bsz: int) -> torch.Tensor:
        return tensor.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        focus_states: torch.Tensor,
        token_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        stepwise_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            focus_states: [batch, num_focus_tokens, hidden]
            token_states:  [batch, seq_len, hidden]
            attention_mask: Optional broadcastable mask ([batch, 1, num_focus_tokens, seq_len])
            stepwise_mask: Optional per-focus coverage mask ([batch, num_focus_tokens, seq_len])
            past_key_value: Optional cached KV from previous segments
        Returns:
            compressed states, and optionally cached KV for incremental decoding.
        """

        bsz, num_focus_tokens, _ = focus_states.size()
        seq_len = token_states.size(1)

        target_dtype = self.q_proj.weight.dtype
        if focus_states.dtype != target_dtype:
            focus_states = focus_states.to(target_dtype)
        if token_states.dtype != target_dtype:
            token_states = token_states.to(target_dtype)

        query = self._shape(self.q_proj(focus_states), bsz)
        key = self._shape(self.k_proj(token_states), bsz)
        value = self._shape(self.v_proj(token_states), bsz)

        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)
        present = (key, value) if use_cache else None

        attn_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scaling

        if stepwise_mask is not None:
            expanded_mask = stepwise_mask.unsqueeze(1)  # [batch, 1, num_focus_tokens, seq_len]
            attn_scores = attn_scores.masked_fill(~expanded_mask, torch.finfo(attn_scores.dtype).min)

        if attention_mask is not None:
            attn_scores += attention_mask

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, value)
        context = context.transpose(1, 2).contiguous().view(bsz, num_focus_tokens, self.hidden_size)
        output = self.out_proj(context)
        return output, present

    def stepwise_expansion_mask(
        self,
        block_map,
        focus_tokens_per_block: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Builds a per-focus-token coverage mask implementing stepwise expansion."""
        if not block_map:
            return torch.zeros((1, 0, 0), dtype=torch.bool, device=device)

        max_tokens = block_map[-1][1] if block_map else 0
        total_focus_tokens = len(block_map) * focus_tokens_per_block
        mask = torch.zeros((1, total_focus_tokens, max_tokens), dtype=torch.bool, device=device)
        focus_idx = 0
        for start, end in block_map:
            span = end - start
            tokens = torch.arange(start, end, device=device)
            for local_focus in range(focus_tokens_per_block):
                coverage = min(span, ((local_focus + 1) * span) // focus_tokens_per_block)
                covered_tokens = tokens[:coverage]
                if coverage == 0:
                    mask[0, focus_idx, start] = True
                else:
                    mask[0, focus_idx, covered_tokens] = True
                focus_idx += 1
        return mask
