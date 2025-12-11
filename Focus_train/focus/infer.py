import argparse
import os
import sys
from typing import Dict, Optional

import torch

if __package__ is None or __package__ == "":
    package_root = os.path.dirname(os.path.dirname(__file__))
    if package_root not in sys.path:
        sys.path.append(package_root)
    from focus.adapter import LLMAdapter  # type: ignore
    from focus.token_injection import FocusTokenInserter  # type: ignore
    from focus.utils import load_yaml, prepare_output_dir, save_json  # type: ignore
    from focus.metrics import MetricsCollector  # type: ignore
else:  # pragma: no cover
    from .adapter import LLMAdapter
    from .token_injection import FocusTokenInserter
    from .utils import load_yaml, prepare_output_dir, save_json
    from .metrics import MetricsCollector


def _sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    logits = logits / max(temperature, 1e-6)
    probs = torch.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumulative > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        next_idx = torch.multinomial(sorted_probs, num_samples=1)
        next_token = sorted_indices.gather(-1, next_idx)
    else:
        next_token = torch.multinomial(probs, num_samples=1)
    return next_token


def load_checkpoint(adapter: LLMAdapter, checkpoint_dir: str) -> None:
    path = os.path.join(checkpoint_dir, "focus_adapter.pt")
    if not os.path.exists(path):
        return
    state = torch.load(path, map_location=adapter.model.device)
    adapter.focus_attention.load_state_dict(state["focus_attention"])
    adapter.model.get_input_embeddings().load_state_dict(state["embedding"])


def generate_with_focus_tokens(config_path: str, prompt: str, output_path: Optional[str] = None) -> Dict:
    config = load_yaml(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adapter = LLMAdapter.from_pretrained(config["model_name_or_path"], config, device=device)
    load_checkpoint(adapter, config.get("paths", {}).get("checkpoint_dir", "checkpoints"))
    adapter.eval()

    tokenizer = adapter.tokenizer
    inserter = FocusTokenInserter(config, tokenizer)
    collector = MetricsCollector(device=device)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    focus_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    state = inserter.init_inference_state()
    past_key_values = None
    generated_tokens = []

    max_new_tokens = config.get("infer", {}).get("max_new_tokens", 256)
    temperature = config.get("infer", {}).get("temperature", 0.8)
    top_p = config.get("infer", {}).get("top_p", 0.95)

    with collector.track_inference():
        for step in range(max_new_tokens):
            model_inputs = input_ids if past_key_values is None else input_ids[:, -1:]
            outputs = adapter.model(
                input_ids=model_inputs,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = outputs.logits[:, -1, :]
            next_token = _sample_next_token(logits, temperature, top_p)

            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_extension = torch.ones_like(next_token)
            attention_mask = torch.cat([attention_mask, attention_extension], dim=1)
            focus_mask = torch.cat([focus_mask, torch.zeros_like(next_token, dtype=torch.bool)], dim=1)

            past_key_values = outputs.past_key_values
            generated_tokens.append(next_token.item())
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
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                past_key_values = adapter.replace_or_merge_kv(outputs.past_key_values, focus_mask[0])

    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    metrics = collector.summarize()
    results = {
        "prompt": prompt,
        "response": decoded,
        "metrics": metrics,
        "generated_tokens": len(generated_tokens),
    }

    if output_path is None:
        output_path = os.path.join(config.get("paths", {}).get("output_dir", "output"), "infer_results.json")
    prepare_output_dir(os.path.dirname(output_path))
    save_json(results, output_path)

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Focus-token generation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt", required=False, default="Biology helps medicine advance rapidly.")
    parser.add_argument("--output", required=False, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_with_focus_tokens(args.config, args.prompt, args.output)


if __name__ == "__main__":
    main()
