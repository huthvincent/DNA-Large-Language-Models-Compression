from __future__ import annotations

import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from ..attention import FocusTokenAttention
from ..kv_pruner import FocusTokenKVPruner
from ..utils import set_trainable

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

_PATHS_BOOTSTRAPPED = False


def _bootstrap_bionemo_paths() -> None:
    """Ensure BioNeMo vendor sub-packages are importable."""
    global _PATHS_BOOTSTRAPPED
    if _PATHS_BOOTSTRAPPED:
        return

    root = Path(__file__).resolve().parents[2]
    vendor_root = root / "vendor" / "bionemo" / "sub-packages"
    if not vendor_root.exists():
        return

    for pkg_dir in sorted(vendor_root.iterdir()):
        if not pkg_dir.is_dir():
            continue
        candidate_src = pkg_dir / "src"
        if candidate_src.exists() and str(candidate_src) not in sys.path:
            sys.path.append(str(candidate_src))
    _PATHS_BOOTSTRAPPED = True


@dataclass
class FocusTokenForwardOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor
    focus_states: torch.Tensor
    compressed_states: torch.Tensor
    past_key_values: Optional[Tuple] = None


@dataclass
class _AdapterForwardOutput:
    logits: torch.Tensor
    past_key_values: Optional[Tuple] = None
    hidden_states: Optional[torch.Tensor] = None


class _ForwardModule:
    """Thin wrapper that mimics HF model interface for downstream code."""

    def __init__(self, adapter: "BioNeMoAdapter") -> None:
        self._adapter = adapter

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> _AdapterForwardOutput:
        return self._adapter._forward_base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

    # HuggingFace compatibility -------------------------------------------------
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **_: Dict,
    ) -> torch.Tensor:
        return self._adapter._generate_autoregressive(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

    def get_input_embeddings(self) -> nn.Module:
        return self._adapter.embedding_module


class _TokenizerWrapper:
    """Provide a minimal HF-compatible interface for BioNeMo tokenizers."""

    def __init__(self, base_tokenizer) -> None:
        self._tokenizer = base_tokenizer
        pad_id = getattr(base_tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(base_tokenizer, "pad_id", None)
        if pad_id is None:
            pad_id = getattr(base_tokenizer, "eos_id", 0)
        self.pad_token_id = int(pad_id)

        eos_id = getattr(base_tokenizer, "eos_token_id", None)
        if eos_id is None:
            eos_id = getattr(base_tokenizer, "eos_id", self.pad_token_id)
        self.eos_token_id = int(eos_id)

    # HuggingFace-style API -----------------------------------------------------
    def __call__(
        self,
        text: Iterable[str] | str,
        return_tensors: Optional[str] = None,
        padding: str | bool = "max_length",
        truncation: bool | str = True,
        max_length: Optional[int] = None,
        **_: Dict,
    ):
        if isinstance(text, str):
            text = [text]

        encoded: List[torch.Tensor] = []
        for sample in text:
            ids = self.text_to_ids(sample)
            if max_length is not None and truncation:
                ids = ids[:max_length]
            encoded.append(torch.tensor(ids, dtype=torch.long))

        max_len = max(seq.size(0) for seq in encoded)
        if padding == "max_length" and max_length is not None:
            max_len = max_length

        input_ids: List[torch.Tensor] = []
        attention: List[torch.Tensor] = []
        for seq in encoded:
            pad_len = max(0, max_len - seq.size(0))
            if pad_len:
                pad_tensor = torch.full((pad_len,), self.pad_token_id, dtype=torch.long)
                seq = torch.cat([seq, pad_tensor], dim=0)
            mask = (seq != self.pad_token_id).long()
            input_ids.append(seq)
            attention.append(mask)

        batch_input = torch.stack(input_ids)
        batch_attention = torch.stack(attention)

        if return_tensors == "pt":
            return {"input_ids": batch_input, "attention_mask": batch_attention}
        return {"input_ids": batch_input.tolist(), "attention_mask": batch_attention.tolist()}

    def text_to_ids(self, text: str) -> List[int]:
        if hasattr(self._tokenizer, "text_to_ids"):
            return list(self._tokenizer.text_to_ids(text))
        if hasattr(self._tokenizer, "encode"):
            return list(self._tokenizer.encode(text))
        raise AttributeError("Underlying tokenizer does not expose text_to_ids/encode.")

    def decode(self, ids: Iterable[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if hasattr(self._tokenizer, "ids_to_text"):
            return self._tokenizer.ids_to_text(list(ids))
        if hasattr(self._tokenizer, "decode"):
            return self._tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)
        raise AttributeError("Underlying tokenizer does not expose ids_to_text/decode.")

    # Convenience proxies -------------------------------------------------------
    def get_vocab(self):
        if hasattr(self._tokenizer, "get_vocab"):
            return self._tokenizer.get_vocab()
        if hasattr(self._tokenizer, "tokenizer") and hasattr(self._tokenizer.tokenizer, "get_vocab"):
            return self._tokenizer.tokenizer.get_vocab()
        raise AttributeError("Tokenizer does not provide vocabulary access.")

    def add_special_tokens(self, tokens: Dict) -> None:
        if hasattr(self._tokenizer, "add_special_tokens"):
            self._tokenizer.add_special_tokens(tokens)
            return
        raise AttributeError("Tokenizer does not support add_special_tokens.")

    def convert_tokens_to_ids(self, token: str) -> int:
        if hasattr(self._tokenizer, "token_to_id"):
            return int(self._tokenizer.token_to_id(token))
        vocab = self.get_vocab()
        if token not in vocab:
            raise KeyError(f"Token '{token}' not present in vocabulary.")
        return int(vocab[token])


class BioNeMoAdapter(nn.Module):
    """Adapter that wraps the proprietary BioNeMo Evo-2 Hyena model."""

    def __init__(
        self,
        *,
        base_model: nn.Module,
        tokenizer,
        config: Dict,
        hyena_config,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.device = device
        self.base_model = base_model
        self.add_module("base_model_module", self.base_model)

        self.hyena_config = hyena_config
        self.config = config
        self.tokenizer = _TokenizerWrapper(tokenizer)

        hidden_size = getattr(hyena_config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(hyena_config, "model_dim", None)
        if hidden_size is None:
            raise ValueError("Unable to determine hidden_size from Hyena config.")

        num_heads = getattr(hyena_config, "num_attention_heads", None)
        if num_heads is None:
            num_heads = getattr(hyena_config, "num_mixer_heads", None)
        if num_heads is None:
            raise ValueError("Unable to determine number of attention heads for Hyena model.")

        dropout = getattr(hyena_config, "attention_dropout", 0.0)

        self.focus_attention = FocusTokenAttention(hidden_size, num_heads, dropout=dropout)
        self.pruner = FocusTokenKVPruner()
        self.model = _ForwardModule(self)

        self._last_hidden: Optional[torch.Tensor] = None
        self.embedding_module = self._resolve_embedding_module()

        special_tokens = config.get("special_tokens", {})
        self._focus_token_id = int(special_tokens.get("focus_token_id", 0))
        self._focus_tokens_per_block = int(config.get("focus_tokens_per_block", 1))

        self.to(device)
        self._register_embedding_hook()
        self._register_hidden_hook()

    # ------------------------------------------------------------------ helpers
    def _ensure_device(self, tensor: Optional[torch.Tensor], *, dtype: Optional[torch.dtype] = None) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        if dtype is not None:
            return tensor.to(self.device, dtype=dtype)
        return tensor.to(self.device)

    def _build_position_ids(self, input_ids: torch.Tensor, past_key_values: Optional[Tuple]) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        offset = 0
        if past_key_values:
            if isinstance(past_key_values, (list, tuple)) and past_key_values:
                past_seq = past_key_values[0][0].size(-2)
                offset = past_seq
        position_ids = torch.arange(offset, offset + seq_len, device=self.device, dtype=torch.long).unsqueeze(0)
        return position_ids.expand(batch, -1).contiguous()

    def _forward_base(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = True,
        **model_kwargs,
    ) -> _AdapterForwardOutput:
        input_ids = self._ensure_device(input_ids, dtype=torch.int64)
        attention_mask = self._ensure_device(attention_mask, dtype=torch.int64)
        position_ids = self._build_position_ids(input_ids, past_key_values)

        self._last_hidden = None
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **model_kwargs,
        )

        logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
        hidden_states = self._last_hidden
        if hidden_states is not None and hidden_states.dim() == 3:
            seq_dim, batch_dim = hidden_states.size(0), hidden_states.size(1)
            expected_batch = input_ids.size(0)
            expected_seq = input_ids.size(1)
            if seq_dim == expected_seq and batch_dim == expected_batch:
                hidden_states = hidden_states.transpose(0, 1).contiguous()
            elif seq_dim == expected_batch and batch_dim == expected_seq:
                hidden_states = hidden_states.contiguous()
            else:
                LOGGER.warning(
                    "Unexpected hidden state shape seq=%s batch=%s (expected seq=%s batch=%s)",
                    seq_dim,
                    batch_dim,
                    expected_seq,
                    expected_batch,
                )
        self._last_hidden = hidden_states
        return _AdapterForwardOutput(
            logits=logits,
            past_key_values=None if not use_cache else None,
            hidden_states=hidden_states,
        )

    def _generate_autoregressive(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
    ) -> torch.Tensor:
        input_ids = self._ensure_device(input_ids, dtype=torch.long)
        if attention_mask is not None:
            attention_mask = self._ensure_device(attention_mask, dtype=torch.long)
        else:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=self.device)

        generated = input_ids
        attn_mask = attention_mask

        for _ in range(max_new_tokens):
            outputs = self._forward_base(generated, attention_mask=attn_mask, use_cache=False)
            logits = outputs.logits[:, -1, :]

            if do_sample:
                logits = logits / max(temperature, 1e-6)
                probs = torch.softmax(logits, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cumulative = torch.cumsum(sorted_probs, dim=-1)
                    cutoff = cumulative > top_p
                    cutoff[..., 1:] = cutoff[..., :-1]
                    cutoff[..., 0] = False
                    sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                    next_token_idx = torch.multinomial(sorted_probs, num_samples=1)
                    next_token = sorted_idx.gather(-1, next_token_idx)
                else:
                    next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)
            attn_mask = torch.cat([attn_mask, torch.ones_like(next_token, dtype=torch.long, device=self.device)], dim=1)

        return generated

    def _resolve_embedding_module(self) -> nn.Module:
        candidate_paths = [
            "embedding.word_embeddings",
            "embedding",
            "language_model.embedding.word_embeddings",
            "language_model.embedding",
            "model.embedding.word_embeddings",
            "model.embedding",
        ]
        module = self.base_model
        if hasattr(module, "module"):
            module = module.module

        for path in candidate_paths:
            target = module
            for part in path.split("."):
                if not hasattr(target, part):
                    target = None
                    break
                target = getattr(target, part)
            if isinstance(target, nn.Module) and hasattr(target, "weight"):
                return target

        # Fallback search: pick the first large embedding-like parameter
        for child in module.modules():
            if hasattr(child, "weight"):
                weight = child.weight
                if weight.ndim == 2 and weight.size(0) > 32 and weight.size(1) > 32:
                    LOGGER.warning("Using heuristic match for embedding module: %s", child.__class__.__name__)
                    return child

        raise RuntimeError("Unable to locate token embedding module within the Hyena model.")

    def _register_embedding_hook(self) -> None:
        if self.embedding_module is None or not hasattr(self.embedding_module, "weight"):
            return

        weight = self.embedding_module.weight
        mask = torch.zeros(
            weight.shape[0],
            device=weight.device,
            dtype=weight.dtype,
        )
        mask[self._focus_token_id] = 1.0

        def _mask_grad(grad: torch.Tensor) -> torch.Tensor:
            mask_view = mask
            if mask_view.dtype != grad.dtype:
                mask_view = mask_view.to(grad.dtype)
            return grad * mask_view.unsqueeze(-1)

        self.embedding_module.weight.register_hook(_mask_grad)

    def _register_hidden_hook(self) -> None:
        module = self.base_model
        if hasattr(module, "module"):
            module = module.module

        decoder = None
        for attr in ("decoder", "language_model.decoder", "model.decoder"):
            current = module
            for part in attr.split("."):
                if not hasattr(current, part):
                    current = None
                    break
                current = getattr(current, part)
            if current is not None:
                decoder = current
                break

        if decoder is None:
            LOGGER.warning("Falling back to registering hook on base model; hidden states may be unavailable.")
            target_module = module
        else:
            target_module = decoder

        def _capture_hidden(_: nn.Module, __, output):
            if isinstance(output, tuple):
                self._last_hidden = output[0]
            else:
                self._last_hidden = output

        target_module.register_forward_hook(_capture_hidden)

    # ---------------------------------------------------------------- interface
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        config: Dict,
        device: Optional[str] = None,
    ) -> "BioNeMoAdapter":
        _bootstrap_bionemo_paths()

        try:
            from megatron.core import parallel_state
            from megatron.core.tensor_parallel import random as tp_random
            from megatron.core.transformer.module import Float16Module
            from nemo.collections.llm.gpt.model.hyena import HYENA_MODEL_OPTIONS
            from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
            from bionemo.llm.utils.weight_utils import load_weights_sharded_inplace_nemo2_to_mcore
        except ImportError as exc:  # pragma: no cover - requires proprietary stack
            raise ImportError(
                "BioNeMo backend requires the proprietary BioNeMo/Nemo stack. Ensure PYTHONPATH "
                "includes vendor/bionemo/sub-packages/*/src and install third-party dependencies."
            ) from exc

        device_obj = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_obj.type == "cuda":
            cuda_index = device_obj.index if device_obj.index is not None else 0
            torch.cuda.set_device(cuda_index)

        _initialize_distributed(parallel_state, device_obj)
        seed = config.get("train", {}).get("seed", 42)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            tp_random.model_parallel_cuda_manual_seed(seed)

        bionemo_cfg = config.get("bionemo", {})
        tokenizer_kind = bionemo_cfg.get("tokenizer", "byte-level")
        tokenizer = get_nmt_tokenizer(tokenizer_kind)

        model_size = bionemo_cfg.get("model_size", "7b")
        if model_size not in HYENA_MODEL_OPTIONS:
            raise ValueError(f"Unsupported Hyena model size '{model_size}'. Available: {sorted(HYENA_MODEL_OPTIONS)}")

        hyena_config_builder = HYENA_MODEL_OPTIONS[model_size]
        try:
            hyena_config = hyena_config_builder(apply_rope_fusion=False)
        except TypeError:
            hyena_config = hyena_config_builder()
            if getattr(hyena_config, "apply_rope_fusion", False):
                hyena_config.apply_rope_fusion = False
        hyena_config.use_te = bool(bionemo_cfg.get("use_te", getattr(hyena_config, "use_te", True)))
        hyena_config.seq_length = int(bionemo_cfg.get("seq_length", config.get("block_len", hyena_config.seq_length)))
        hyena_config.fp8 = bool(bionemo_cfg.get("fp8", getattr(hyena_config, "fp8", False)))

        raw_model = hyena_config.configure_model(tokenizer)
        raw_model.eval()
        raw_model.to(device_obj)

        model_name_or_path = os.path.expandvars(model_name_or_path)
        checkpoint_path = Path(model_name_or_path).expanduser()
        weights_dir = checkpoint_path / "weights"
        if weights_dir.exists():
            LOGGER.info("Loading BioNeMo checkpoint from %s", weights_dir)
            load_weights_sharded_inplace_nemo2_to_mcore(raw_model, str(checkpoint_path), set())
        else:
            LOGGER.info("Loading BioNeMo checkpoint from %s", checkpoint_path)
            candidate = checkpoint_path
            if candidate.is_dir() and (candidate / "weights.pt").exists():
                candidate = candidate / "weights.pt"
            state = torch.load(candidate, map_location=device_obj)
            if "module" in state:
                state = state["module"]
            clean_state = {k[len("module.") :]: v for k, v in state.items() if k.startswith("module.")}
            missing, unexpected = raw_model.load_state_dict(clean_state, strict=False)
            if missing:
                LOGGER.warning("Missing keys when loading checkpoint: %s", missing)
            if unexpected:
                LOGGER.warning("Unexpected keys when loading checkpoint: %s", unexpected)

        float16_model = Float16Module(hyena_config, raw_model)
        float16_model.eval()
        float16_model.to(device_obj)

        return cls(
            base_model=float16_model,
            tokenizer=tokenizer,
            config=config,
            hyena_config=hyena_config,
            device=device_obj,
        )

    # ---------------------------------------------------------------- lifecycle
    def freeze_base(self) -> None:
        set_trainable(self.base_model, False)
        set_trainable(self.focus_attention, True)
        if self.embedding_module is not None:
            self.embedding_module.weight.requires_grad_(True)

    # ----------------------------------------------------------------- forward
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
        outputs = self._forward_base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **model_kwargs,
        )

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError(
                "Hidden states are unavailable from the Hyena decoder. Ensure the hidden hook is attached correctly."
            )

        (
            focus_states,
            token_states,
            focus_indices,
            token_indices,
        ) = self._split_hidden_states(hidden_states, focus_mask, attention_mask)

        attention_bias = None
        if causal_mask is not None:
            attention_bias = self._build_attention_bias(
                causal_mask,
                focus_indices,
                token_indices,
                focus_states.size(1),
                token_states.size(1),
                hidden_states.dtype,
            )

        compressed, _ = self.focus_attention(
            focus_states,
            token_states,
            attention_mask=attention_bias,
            stepwise_mask=None,
            past_key_value=None,
            use_cache=False,
        )

        return FocusTokenForwardOutput(
            logits=outputs.logits,
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
        (
            focus_states,
            token_states,
            _,
            _,
        ) = self._split_hidden_states(hidden_states, focus_mask, attention_mask)
        compressed, _ = self.focus_attention(
            focus_states,
            token_states,
            stepwise_mask=None,
        )
        return compressed

    def replace_or_merge_kv(
        self,
        past_key_values: Optional[Tuple],
        focus_mask: torch.Tensor,
        new_past: Optional[Tuple] = None,
    ) -> Optional[Tuple]:
        if past_key_values is None:
            return new_past
        try:
            pruned = self.pruner.prune(past_key_values, focus_mask.squeeze(0))
        except Exception as exc:  # pragma: no cover - defensive path
            LOGGER.warning("KV pruning failed: %s", exc)
            pruned = past_key_values
        if new_past is None:
            return pruned
        try:
            return self.pruner.merge(pruned, new_past)
        except Exception as exc:  # pragma: no cover - defensive path
            LOGGER.warning("KV merge failed: %s", exc)
            return pruned

    def generate_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Optional[Tuple] = None,
        **kwargs,
    ) -> SimpleNamespace:
        outputs = self._forward_base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        return SimpleNamespace(
            logits=outputs.logits[:, -1, :],
            past_key_values=outputs.past_key_values,
        )

    # ---------------------------------------------------------------- utilities
    def _split_hidden_states(
        self,
        hidden_states: torch.Tensor,
        focus_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        batch_size = hidden_states.size(0)
        focus_slices: List[torch.Tensor] = []
        token_slices: List[torch.Tensor] = []
        focus_indices: List[torch.Tensor] = []
        token_indices: List[torch.Tensor] = []
        token_masks = (~focus_mask) & (attention_mask > 0)
        hidden_dim = hidden_states.size(-1)

        for i in range(batch_size):
            focus_idx = torch.nonzero(focus_mask[i], as_tuple=False).flatten()
            token_idx = torch.nonzero(token_masks[i], as_tuple=False).flatten()

            focus_rows = (
                hidden_states[i].index_select(0, focus_idx)
                if focus_idx.numel() > 0
                else hidden_states[i].new_zeros((0, hidden_dim))
            )
            token_rows = (
                hidden_states[i].index_select(0, token_idx)
                if token_idx.numel() > 0
                else hidden_states[i].new_zeros((0, hidden_dim))
            )

            focus_slices.append(focus_rows)
            token_slices.append(token_rows)
            focus_indices.append(focus_idx)
            token_indices.append(token_idx)

        focus_tensor = pad_sequence(focus_slices, batch_first=True)
        token_tensor = pad_sequence(token_slices, batch_first=True)
        return (
            focus_tensor,
            token_tensor,
            focus_indices,
            token_indices,
        )

    def _build_attention_bias(
        self,
        causal_mask: torch.Tensor,
        focus_indices: List[torch.Tensor],
        token_indices: List[torch.Tensor],
        max_focus_tokens: int,
        max_tokens: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch_size = causal_mask.size(0)
        device = causal_mask.device
        finfo = torch.finfo(dtype)
        neg_inf = finfo.min
        bias = torch.full((batch_size, max_focus_tokens, max_tokens), neg_inf, dtype=dtype, device=device)

        for i in range(batch_size):
            focus_idx = focus_indices[i]
            token_idx = token_indices[i]
            if focus_idx.numel() == 0 or token_idx.numel() == 0:
                continue

            mask_slice = causal_mask[i].index_select(0, focus_idx).index_select(1, token_idx)
            allowed = torch.where(
                mask_slice,
                torch.zeros(mask_slice.shape, dtype=dtype, device=device),
                torch.full(mask_slice.shape, neg_inf, dtype=dtype, device=device),
            )
            focus_count, token_count = allowed.size(0), allowed.size(1)
            bias[i, :focus_count, :token_count] = allowed

        return bias.unsqueeze(1)


def _initialize_distributed(parallel_state, device: torch.device) -> None:
    if torch.distributed.is_initialized():
        return

    backend = "nccl" if device.type == "cuda" else "gloo"
    init_file = os.environ.get("TORCH_DIST_INIT_FILE")
    cleanup = False
    if init_file is None:
        fd, init_file = tempfile.mkstemp(prefix="dist_init_", suffix=".pt")
        os.close(fd)
        cleanup = True
    else:
        os.makedirs(os.path.dirname(init_file), exist_ok=True)

    if backend == "gloo":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")

    torch.distributed.init_process_group(
        backend=backend,
        init_method=f"file://{init_file}",
        rank=0,
        world_size=1,
    )

    if cleanup:
        try:
            os.remove(init_file)
        except OSError:
            pass
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        expert_model_parallel_size=1,
    )
