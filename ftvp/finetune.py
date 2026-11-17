"""Fine-tune a small transformer (default: MiniLM-L12, 33M parameters) to classify intents. Runs on CPU.

Needs the [train] extra: torch and transformers.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

from .data import Example

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L12-v2"


def _torch():
    try:
        import torch
        import transformers
    except ImportError as e:
        raise ImportError("Fine-tuning needs torch and transformers: pip install -e '.[train]'") from e
    return torch, transformers


def train(examples: list[Example], labels: list[str], out_dir: str | Path, base_model: str = DEFAULT_MODEL,
          epochs: int = 6, min_steps: int = 300, batch_size: int = 32, lr: float = 8e-5,
          max_length: int = 64, seed: int = 0, log=print) -> dict:
    """Fine-tune and save to out_dir. Small datasets get more epochs so every run takes at least
    min_steps optimizer steps. Returns training stats."""
    torch, transformers = _torch()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(seed)
    torch.manual_seed(seed)
    label2id = {label: i for i, label in enumerate(labels)}
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=label2id)

    steps_per_epoch = math.ceil(len(examples) / batch_size)
    epochs = max(epochs, math.ceil(min_steps / steps_per_epoch))
    total_steps = steps_per_epoch * epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

    log(f"Fine-tuning {base_model} on {len(examples)} examples: {epochs} epochs, {total_steps} steps")
    model.train()
    start = time.perf_counter()
    order = list(range(len(examples)))
    step = 0
    for epoch in range(epochs):
        random.shuffle(order)
        running = 0.0
        for b in range(0, len(order), batch_size):
            batch = [examples[i] for i in order[b:b + batch_size]]
            enc = tokenizer([e.text for e in batch], padding=True, truncation=True, max_length=max_length,
                            return_tensors="pt")
            out = model(**enc, labels=torch.tensor([label2id[e.label] for e in batch]))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += out.loss.item()
            step += 1
        log(f"  epoch {epoch + 1}/{epochs}: loss {running / steps_per_epoch:.4f} "
            f"({time.perf_counter() - start:.0f}s)")
    seconds = time.perf_counter() - start

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    stats = {"base_model": base_model, "examples": len(examples), "epochs": epochs, "steps": total_steps,
             "train_seconds": round(seconds, 1), "parameters": sum(p.numel() for p in model.parameters()),
             "lr": lr, "batch_size": batch_size, "max_length": max_length, "seed": seed}
    (out_dir / "train_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


class FineTunedClassifier:
    """Loads a saved fine-tuned model for prediction."""

    def __init__(self, model_dir: str | Path, threads: int | None = None):
        torch, _ = _torch()
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
        self.id2label = self.model.config.id2label

    def predict(self, texts: list[str], batch_size: int = 64) -> list[str]:
        out = []
        with self.torch.inference_mode():
            for b in range(0, len(texts), batch_size):
                enc = self.tokenizer(texts[b:b + batch_size], padding=True, truncation=True, max_length=64,
                                     return_tensors="pt")
                ids = self.model(**enc).logits.argmax(-1).tolist()
                out.extend(self.id2label[i] for i in ids)
        return out
