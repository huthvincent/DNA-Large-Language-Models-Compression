import torch

from focus.token_injection import FocusTokenInserter


class FakeTokenizer:
    def __init__(self):
        self.vocab = {"<pad>": 0, "A": 1, "B": 2, "<|focus|>": 3}
        self.pad_token_id = 0

    def get_vocab(self):
        return self.vocab

    def add_special_tokens(self, tokens):
        for tok in tokens.get("additional_special_tokens", []):
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]


def test_focus_token_insertion_and_mask():
    tokenizer = FakeTokenizer()
    config = {"block_len": 4, "focus_tokens_per_block": 2, "insert_every_n": 4}
    inserter = FocusTokenInserter(config, tokenizer)

    batch = {
        "input_ids": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }

    updated, meta = inserter.prepare_training_batch(batch)
    total_tokens = updated["attention_mask"].sum().item()
    assert total_tokens == 6

    # Ensure exactly two focus tokens inserted
    assert meta.focus_mask.sum().item() == 2

    # Causal mask should block future access
    causal = meta.causal_mask[0]
    assert not torch.triu(causal, diagonal=1).any()

    # Focus token positions should only see their own block
    focus_token_positions = torch.nonzero(meta.focus_mask[0], as_tuple=False).flatten()
    for pos in focus_token_positions:
        allowed = causal[pos].nonzero(as_tuple=False).flatten()
        assert allowed.max() <= pos
