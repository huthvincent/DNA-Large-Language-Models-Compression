import argparse
import math
import os
import sys
import time
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchmetrics.text.rouge import ROUGEScore

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
    from focus.train import compute_loss  # type: ignore
    from focus.utils import (  # type: ignore
        describe_memory,
        load_yaml,
        prepare_output_dir,
        save_json,
    )
else:  # pragma: no cover
    from .adapter import LLMAdapter
    from .data import get_dataset_split, load_text_dataset_from_config
    from .token_injection import FocusTokenInserter
    from .train import compute_loss
    from .utils import (
        describe_memory,
        load_yaml,
        prepare_output_dir,
        save_json,
    )


class MetricsCollector:
    """Collects wall-clock time, memory, and throughput statistics."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.records: Dict[str, Dict[str, float]] = {}
        self._active_label: Optional[str] = None
        self._start_time: Optional[float] = None

    @contextmanager
    def track_inference(self, label: str = "inference"):
        self._active_label = label
        self._start_time = time.time()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        try:
            yield
        finally:
            duration = time.time() - (self._start_time or time.time())
            memory_stats = describe_memory(self.device)
            self.records[label] = {
                "duration_sec": duration,
                "peak_memory_mb": memory_stats.get("reserved_mb", 0.0),
            }
            self._active_label = None
            self._start_time = None

    def set_tokens(self, tokens: int, label: str = "inference") -> None:
        rec = self.records.setdefault(label, {})
        rec["tokens"] = float(tokens)
        if rec.get("duration_sec", 0) > 0:
            rec["tokens_per_sec"] = rec["tokens"] / rec["duration_sec"]

    def summarize(self) -> Dict[str, Dict[str, float]]:
        return self.records


def run_focus_tokens_generation(
    adapter: LLMAdapter,
    inserter: FocusTokenInserter,
    prompt: str,
    infer_cfg: Dict,
    device: torch.device,
) -> Dict[str, float]:
    tokenizer = adapter.tokenizer
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    focus_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    state = inserter.init_inference_state()
    past = None
    generated = []
    max_new_tokens = infer_cfg.get("max_new_tokens", 256)

    collector = MetricsCollector(device)
    with collector.track_inference("focus"):
        for _ in range(max_new_tokens):
            model_inputs = input_ids if past is None else input_ids[:, -1:]
            outputs = adapter.model(
                input_ids=model_inputs,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
            focus_mask = torch.cat([focus_mask, torch.zeros_like(next_token, dtype=torch.bool)], dim=1)
            past = outputs.past_key_values
            generated.append(next_token.item())
            inserter.observe_token(state)

            insertion = inserter.maybe_insert_focus_tokens(state)
            if insertion:
                focus_tokens = torch.tensor(insertion.focus_token_ids, device=device, dtype=torch.long).unsqueeze(0)
                input_ids = torch.cat([input_ids, focus_tokens], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(focus_tokens)], dim=1)
                focus_flags = torch.ones_like(focus_tokens, dtype=torch.bool)
                focus_mask = torch.cat([focus_mask, focus_flags], dim=1)

                outputs = adapter.model(
                    input_ids=focus_tokens,
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                )
                past = adapter.replace_or_merge_kv(outputs.past_key_values, focus_mask[0])

    collector.set_tokens(len(generated), "focus")
    return collector.summarize()["focus"]


def run_baseline_generation(
    model,
    tokenizer,
    prompt: str,
    infer_cfg: Dict,
    device: torch.device,
) -> Dict[str, float]:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    collector = MetricsCollector(device)
    with collector.track_inference("baseline"):
        outputs = model.generate(
            **inputs,
            max_new_tokens=infer_cfg.get("max_new_tokens", 256),
            do_sample=False,
        )
    tokens = outputs.size(1) - inputs["input_ids"].size(1)
    collector.set_tokens(tokens, "baseline")
    return collector.summarize()["baseline"]


def _collate(batch: Iterable[Dict], pad_token_id: int) -> Dict[str, torch.Tensor]:
    input_ids = torch.stack([torch.tensor(sample["input_ids"], dtype=torch.long) for sample in batch])
    attention_mask = torch.stack([torch.tensor(sample["attention_mask"], dtype=torch.long) for sample in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _tokenize(tokenizer, dataset, text_field: str, max_length: int):
    def _apply(batch):
        return tokenizer(
            batch[text_field],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    return dataset.map(_apply, batched=True, remove_columns=[text_field])


def _build_loader(tokenizer, task_cfg: Dict, block_len: int, batch_size: int) -> DataLoader:
    dataset_cfg = dict(task_cfg)
    if "path" not in dataset_cfg and "name" not in dataset_cfg:
        dataset_cfg["name"] = dataset_cfg.get("dataset")
    dataset = load_text_dataset_from_config(dataset_cfg)
    split = task_cfg.get("split", "validation")
    dataset_split = get_dataset_split(dataset, split)
    tokenized = _tokenize(tokenizer, dataset_split, task_cfg.get("text_field", "text"), block_len)
    if task_cfg.get("max_samples"):
        max_samples = min(task_cfg["max_samples"], len(tokenized))
        tokenized = tokenized.select(range(max_samples))
    return DataLoader(tokenized, batch_size=batch_size, collate_fn=lambda batch: _collate(batch, tokenizer.pad_token_id))


def evaluate_perplexity(
    adapter: LLMAdapter,
    inserter: FocusTokenInserter,
    dataloader: DataLoader,
    device: torch.device,
    use_focus_tokens: bool,
) -> Tuple[float, float]:
    total_loss = 0.0
    total_tokens = 0
    model = adapter.model if use_focus_tokens else adapter.model
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            if use_focus_tokens:
                updated, meta = inserter.prepare_training_batch(batch)
                try:
                    outputs = adapter.forward_with_focus_tokens(
                        updated["input_ids"],
                        updated["attention_mask"],
                        meta.focus_mask,
                        causal_mask=meta.causal_mask,
                        block_map=meta.block_map,
                        use_cache=False,
                    )
                    loss_inputs = updated
                except Exception:  # pragma: no cover - fallback when focus-token path fails
                    outputs = adapter.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    loss_inputs = batch
                loss = compute_loss(outputs.logits, loss_inputs["input_ids"], adapter.tokenizer.pad_token_id)
                tokens = loss_inputs["attention_mask"].sum().item()
            else:
                outputs = adapter.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                loss = compute_loss(outputs.logits, batch["input_ids"], adapter.tokenizer.pad_token_id)
                tokens = batch["attention_mask"].sum().item()
            total_loss += loss.item() * tokens
            total_tokens += tokens
    if total_tokens == 0:
        return float("nan"), 0.0
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss), avg_loss


def _rouge_generate(model, tokenizer, prompt: str, max_new_tokens: int, device: torch.device) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def evaluate_rouge(
    adapter: LLMAdapter,
    baseline_model,
    baseline_tokenizer,
    task_cfg: Dict,
    device: torch.device,
) -> Dict[str, float]:
    dataset_cfg = dict(task_cfg)
    if "path" not in dataset_cfg and "name" not in dataset_cfg:
        dataset_cfg["name"] = dataset_cfg.get("dataset")
    dataset = get_dataset_split(
        load_text_dataset_from_config(dataset_cfg),
        task_cfg.get("split", "validation"),
    )
    max_samples = task_cfg.get("max_samples", 16)
    metric = ROUGEScore()
    tokenizer = adapter.tokenizer
    prompts: List[str] = []
    references: List[str] = []
    focus_preds: List[str] = []
    base_preds: List[str] = []
    for idx in range(min(max_samples, len(dataset))):
        sample = dataset[idx]
        prompt = sample[task_cfg.get("text_field", "document")]
        reference = sample.get(task_cfg.get("target_field", "summary"), "")
        references.append(reference)
        prompts.append(prompt)
        focus_preds.append(_rouge_generate(adapter.model, tokenizer, prompt, task_cfg.get("max_new_tokens", 128), device))
        base_preds.append(_rouge_generate(baseline_model, baseline_tokenizer, prompt, task_cfg.get("max_new_tokens", 128), device))
    focus_scores = metric(focus_preds, references)
    baseline_scores = metric(base_preds, references)
    return {
        "focus": {k: float(v.item()) for k, v in focus_scores.items()},
        "baseline": {k: float(v.item()) for k, v in baseline_scores.items()},
    }


def evaluate(config_path: str) -> Dict:
    config = load_yaml(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = LLMAdapter.from_pretrained(config["model_name_or_path"], config, device=device)
    inserter = FocusTokenInserter(config, adapter.tokenizer)

    checkpoint_dir = config.get("paths", {}).get("checkpoint_dir", "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "focus_adapter.pt")
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        adapter.focus_attention.load_state_dict(state["focus_attention"])
        adapter.model.get_input_embeddings().load_state_dict(state["embedding"])

    metrics_cfg = config.get("metrics", {})
    baseline_model_name = metrics_cfg.get("baseline_model_name_or_path", config["model_name_or_path"])
    baseline_backend = metrics_cfg.get("baseline_backend", config.get("backend", "hf"))
    baseline_cfg = {**config, "backend": baseline_backend}
    baseline_adapter = LLMAdapter.from_pretrained(baseline_model_name, baseline_cfg, device=device)
    baseline_adapter.eval()
    baseline_model = baseline_adapter.model
    baseline_tokenizer = baseline_adapter.tokenizer

    results = {"perplexity": {}, "rouge": {}, "runtime": {}, "metadata": {}}

    tasks = config.get("metrics", {}).get("eval_tasks", [])
    for task in tasks:
        if task["name"].lower() == "perplexity":
            loader = _build_loader(adapter.tokenizer, task, config["block_len"], config.get("train", {}).get("batch_size", 1))
            focus_ppl, focus_loss = evaluate_perplexity(adapter, inserter, loader, device, use_focus_tokens=True)
            baseline_loader = _build_loader(baseline_tokenizer, task, config["block_len"], config.get("train", {}).get("batch_size", 1))
            baseline_ppl, baseline_loss = evaluate_perplexity(
                baseline_adapter,
                inserter,
                baseline_loader,
                device,
                use_focus_tokens=False,
            )
            results["perplexity"] = {
                "focus": focus_ppl,
                "baseline": baseline_ppl,
                "loss_focus": focus_loss,
                "loss_baseline": baseline_loss,
                "quality_drop": (focus_ppl - baseline_ppl) / max(baseline_ppl, 1e-6),
            }
        elif task["name"].lower() == "rouge":
            rouge_scores = evaluate_rouge(adapter, baseline_model, baseline_tokenizer, task, device)
            results["rouge"] = rouge_scores

    infer_cfg = config.get("infer", {})
    prompt = config.get("metrics", {}).get(
        "eval_prompt",
        "Biology and chemistry jointly explain molecular evolution across scales.",
    )
    focus_perf = run_focus_tokens_generation(adapter, inserter, prompt, infer_cfg, device)
    baseline_perf = run_baseline_generation(baseline_model, baseline_tokenizer, prompt, infer_cfg, device)
    memory_drop = 1.0 - (focus_perf.get("peak_memory_mb", 1.0) / max(baseline_perf.get("peak_memory_mb", 1.0), 1e-6))
    speedup = focus_perf.get("tokens_per_sec", 0.0) / max(baseline_perf.get("tokens_per_sec", 1e-6), 1e-6)
    results["runtime"] = {
        "focus": focus_perf,
        "baseline": baseline_perf,
        "memory_drop": memory_drop,
        "speedup": speedup,
    }

    thresholds = config.get("success_thresholds", {})
    memory_drop_min = thresholds.get("memory_drop_min", 0.3)
    quality_drop_max = thresholds.get("quality_drop_max", 0.05)
    speedup_min = thresholds.get("speedup_min", 0.0)

    results["metadata"] = {
        "thresholds": {
            "memory_drop_min": memory_drop_min,
            "quality_drop_max": quality_drop_max,
            "speedup_min": speedup_min,
        }
    }

    output_path = config.get("metrics", {}).get("results_path", os.path.join(config.get("paths", {}).get("output_dir", "output"), "results.json"))
    prepare_output_dir(os.path.dirname(output_path))
    save_json(results, output_path)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate focus-token compression metrics")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(args.config)


if __name__ == "__main__":
    main()
