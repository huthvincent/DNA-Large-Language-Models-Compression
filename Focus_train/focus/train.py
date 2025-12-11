import argparse
import os
import random
import sys
from typing import Dict, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

if __package__ is None or __package__ == "":
    package_root = os.path.dirname(os.path.dirname(__file__))
    if package_root not in sys.path:
        sys.path.append(package_root)
    from focus.adapter import LLMAdapter  # type: ignore
    from focus.data import (  # type: ignore
        get_dataset_split,
        load_text_dataset_from_config,
    )
    from focus.token_injection import FocusTokenInserter  # type: ignore
    from focus.utils import load_yaml, prepare_output_dir, save_json  # type: ignore
else:  # pragma: no cover
    from .adapter import LLMAdapter
    from .data import get_dataset_split, load_text_dataset_from_config
    from .token_injection import FocusTokenInserter
    from .utils import load_yaml, prepare_output_dir, save_json


def _collate(batch: Iterable[Dict], pad_token_id: int) -> Dict[str, torch.Tensor]:
    input_ids = torch.stack([torch.tensor(sample["input_ids"], dtype=torch.long) for sample in batch])
    attention_mask = torch.stack([torch.tensor(sample["attention_mask"], dtype=torch.long) for sample in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def tokenize_dataset(tokenizer, dataset, text_field: str, max_length: int) -> Iterable:
    def _tokenize(batch):
        outputs = tokenizer(
            batch[text_field],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        return outputs

    return dataset.map(_tokenize, batched=True, remove_columns=dataset.column_names)


def build_dataloader(tokenizer, dataset_cfg: Dict, block_len: int, batch_size: int) -> DataLoader:
    dataset = load_text_dataset_from_config(dataset_cfg)
    split = dataset_cfg.get("split", "train")
    dataset_split = get_dataset_split(dataset, split)
    tokenized = tokenize_dataset(tokenizer, dataset_split, dataset_cfg.get("text_field", "text"), block_len)
    return DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: _collate(batch, tokenizer.pad_token_id),
    )


def randomize_plan(inserter: FocusTokenInserter, train_cfg: Dict) -> None:
    if not train_cfg.get("random_ratio_sampling", False):
        return
    default_high = max(inserter.plan.condense_ratio, inserter.plan.insert_every_n)
    low, high = train_cfg.get("ratio_range", [inserter.plan.condense_ratio, default_high])
    low, high = int(low), int(high)
    if low > high:
        low, high = high, low
    sample_ratio = max(1, random.randint(low, high))
    focus_tokens = max(1, inserter.plan.block_len // sample_ratio)
    inserter.plan.focus_tokens_per_block = focus_tokens


def compute_loss(logits: torch.Tensor, labels: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_labels = shift_labels.masked_fill(shift_labels == pad_token_id, -100)
    return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def train_focus_tokens(config_path: str) -> None:
    config = load_yaml(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adapter = LLMAdapter.from_pretrained(config["model_name_or_path"], config, device=device)
    adapter.freeze_base()
    adapter.train()

    train_cfg = config.get("train", {})
    batch_size = train_cfg.get("batch_size", 1)
    dataloader = build_dataloader(adapter.tokenizer, train_cfg["dataset"], config["block_len"], batch_size)
    inserter = FocusTokenInserter(config, adapter.tokenizer)

    optimizer = torch.optim.AdamW([p for p in adapter.parameters() if p.requires_grad], lr=train_cfg.get("lr", 5e-4), weight_decay=train_cfg.get("weight_decay", 0.0))
    grad_accum = train_cfg.get("grad_accum", 1)

    total_steps = train_cfg.get("steps")
    epochs = train_cfg.get("epochs", 1)

    global_step = 0
    losses = []

    for epoch in range(epochs):
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        optimizer.zero_grad()
        for batch in progress:
            if total_steps and global_step >= total_steps:
                break

            randomize_plan(inserter, train_cfg)

            batch = {k: v.to(device) for k, v in batch.items()}
            updated_batch, meta = inserter.prepare_training_batch(batch)

            outputs = adapter.forward_with_focus_tokens(
                updated_batch["input_ids"],
                updated_batch["attention_mask"],
                meta.focus_mask,
                causal_mask=meta.causal_mask,
                block_map=meta.block_map,
            )

            loss = compute_loss(
                outputs.logits,
                updated_batch["input_ids"],
                adapter.tokenizer.pad_token_id,
            )
            loss = loss / grad_accum
            loss.backward()

            if (global_step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            losses.append(loss.item())
            progress.set_postfix({"loss": sum(losses[-10:]) / min(len(losses), 10)})

            if total_steps and global_step >= total_steps:
                break

        if total_steps and global_step >= total_steps:
            break

    checkpoint_dir = config.get("paths", {}).get("checkpoint_dir", "checkpoints")
    prepare_output_dir(checkpoint_dir)
    torch.save(
        {
            "focus_attention": adapter.focus_attention.state_dict(),
            "embedding": adapter.model.get_input_embeddings().state_dict(),
            "config": config,
        },
        os.path.join(checkpoint_dir, "focus_adapter.pt"),
    )

    save_json({"loss": losses[-1] if losses else None, "steps": global_step}, os.path.join(config.get("paths", {}).get("output_dir", "output"), "train_metrics.json"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train focus-token attention modules")
    parser.add_argument("--config", required=True, help="Path to focus_config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_focus_tokens(args.config)


if __name__ == "__main__":
    main()
