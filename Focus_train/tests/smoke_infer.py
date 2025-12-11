import types

import torch

from focus import infer


class FakeTokenizer:
    def __init__(self):
        self.vocab = {"<pad>": 0, "<|focus|>": 1, "A": 2}
        self.pad_token_id = 0

    def get_vocab(self):
        return self.vocab

    def add_special_tokens(self, tokens):
        for tok in tokens.get("additional_special_tokens", []):
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]

    def __call__(self, text, return_tensors="pt"):
        tokens = [2] * max(1, len(text.split()))
        input_ids = torch.tensor([tokens], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, tokens, skip_special_tokens=True):
        return "X" * len(tokens)

    def to(self, device):
        return self


class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size=8):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=True):
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, self.vocab_size)
        logits[..., 2] = 1.0
        pkv = tuple(
            (
                torch.zeros(1, 1, seq_len, 1),
                torch.zeros(1, 1, seq_len, 1),
            )
            for _ in range(2)
        )
        return types.SimpleNamespace(logits=logits, past_key_values=pkv)

    __call__ = forward

    def generate(self, input_ids, attention_mask=None, max_new_tokens=10, do_sample=False):
        append = torch.full((input_ids.size(0), max_new_tokens), 2, dtype=input_ids.dtype)
        return torch.cat([input_ids, append], dim=1)


class DummyAdapter:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.model = DummyModel()

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def eval(self):
        return self

    def replace_or_merge_kv(self, past_key_values, *_args, **_kwargs):
        return past_key_values


def test_generate_with_focus_tokens_monkeypatch(monkeypatch, tmp_path):
    config = {
        "model_name_or_path": "dummy",
        "block_len": 4,
        "focus_tokens_per_block": 1,
        "insert_every_n": 2,
        "condense_ratio": 2,
        "paths": {
            "checkpoint_dir": str(tmp_path),
            "output_dir": str(tmp_path),
        },
        "infer": {
            "max_new_tokens": 4,
            "temperature": 0.7,
            "top_p": 0.9,
        },
        "metrics": {
            "results_path": str(tmp_path / "results.json"),
        },
    }
    config_path = tmp_path / "config.yaml"
    import yaml

    with open(config_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh)

    monkeypatch.setattr(infer, "LLMAdapter", DummyAdapter)
    monkeypatch.setattr(infer, "load_checkpoint", lambda adapter, *_: None)

    result = infer.generate_with_focus_tokens(str(config_path), "This is a prompt.")
    assert "response" in result
    assert result["generated_tokens"] == config["infer"]["max_new_tokens"]
