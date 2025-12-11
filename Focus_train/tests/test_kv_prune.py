import torch

from focus.kv_pruner import FocusTokenKVPruner


def test_prune_keeps_only_focus_tokens():
    pruner = FocusTokenKVPruner()
    seq_len = 6
    head_dim = 4
    past = (
        (
            torch.arange(seq_len * head_dim, dtype=torch.float32).view(1, 1, seq_len, head_dim),
            torch.arange(seq_len * head_dim, dtype=torch.float32).view(1, 1, seq_len, head_dim) + 10,
        ),
    )
    focus_mask = torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.bool)
    pruned = pruner.prune(past, focus_mask)
    kept = pruned[0][0]
    assert kept.size(-2) == 2
    torch.testing.assert_close(
        kept[0, 0, 0],
        torch.tensor([4 * head_dim + i for i in range(head_dim)], dtype=torch.float32),
    )
