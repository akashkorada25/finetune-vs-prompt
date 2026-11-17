"""Fine-tune a small transformer (default: MiniLM-L12, 33M parameters) to classify intents. Runs on CPU.

The model is the pretrained encoder with mean pooling (how all-MiniLM-L12-v2 was trained) and a
linear head. Two choices make it learn well from only a few examples per intent:

- The head starts as a nearest-centroid classifier: each label's weight vector is the normalised
  average embedding of its training examples, so the model is useful before the first step.
- The new head gets a higher learning rate than the pretrained encoder.

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
HEAD_SCALE = 20.0  # temperature for the cosine-initialised head


def _torch():
    try:
        import torch
        import transformers  # noqa: F401
    except ImportError as e:
        raise ImportError("Fine-tuning needs torch and transformers: pip install -e '.[train]'") from e
    return torch


def _mean_pool(torch, hidden, mask):
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def _encode(torch, encoder, tokenizer, texts, max_length, batch_size=64):
    out = []
    with torch.inference_mode():
        for b in range(0, len(texts), batch_size):
            enc = tokenizer(texts[b:b + batch_size], padding=True, truncation=True, max_length=max_length,
                            return_tensors="pt")
            out.append(_mean_pool(torch, encoder(**enc).last_hidden_state, enc["attention_mask"]))
    return torch.cat(out)


def train(examples: list[Example], labels: list[str], out_dir: str | Path, base_model: str = DEFAULT_MODEL,
          epochs: int = 6, min_steps: int = 300, batch_size: int = 32, lr: float = 5e-5, head_lr: float = 1e-3,
          max_length: int = 64, seed: int = 0, log=print) -> dict:
    """Fine-tune and save to out_dir. Small datasets get more epochs so every run takes at least
    min_steps optimizer steps. Returns training stats."""
    torch = _torch()
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(seed)
    torch.manual_seed(seed)
    label2id = {label: i for i, label in enumerate(labels)}
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    encoder = AutoModel.from_pretrained(base_model)
    head = torch.nn.Linear(encoder.config.hidden_size, len(labels))
    dropout = torch.nn.Dropout(0.1)

    start = time.perf_counter()
    # Nearest-centroid initialisation of the head
    emb = torch.nn.functional.normalize(
        _encode(torch, encoder, tokenizer, [e.text for e in examples], max_length), dim=-1).clone()
    targets = torch.tensor([label2id[e.label] for e in examples])
    with torch.no_grad():
        centroids = torch.zeros(len(labels), emb.shape[1]).index_add_(0, targets, emb)
        head.weight.copy_(HEAD_SCALE * torch.nn.functional.normalize(centroids, dim=-1))
        head.bias.zero_()
        init_acc = ((torch.nn.functional.normalize(emb, dim=-1) @ head.weight.T).argmax(-1) == targets).float().mean().item()

    steps_per_epoch = math.ceil(len(examples) / batch_size)
    epochs = max(epochs, math.ceil(min_steps / steps_per_epoch))
    total_steps = steps_per_epoch * epochs
    optimizer = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": lr},
        {"params": head.parameters(), "lr": head_lr},
    ], weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

    log(f"Fine-tuning {base_model} on {len(examples)} examples: {epochs} epochs, {total_steps} steps "
        f"(centroid head starts at {init_acc:.1%} training accuracy)")
    encoder.train()
    order = list(range(len(examples)))
    for epoch in range(epochs):
        random.shuffle(order)
        running = 0.0
        for b in range(0, len(order), batch_size):
            batch = [examples[i] for i in order[b:b + batch_size]]
            enc = tokenizer([e.text for e in batch], padding=True, truncation=True, max_length=max_length,
                            return_tensors="pt")
            pooled = torch.nn.functional.normalize(
                _mean_pool(torch, encoder(**enc).last_hidden_state, enc["attention_mask"]), dim=-1)
            logits = head(dropout(pooled))
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor([label2id[e.label] for e in batch]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
        log(f"  epoch {epoch + 1}/{epochs}: loss {running / steps_per_epoch:.4f} ({time.perf_counter() - start:.0f}s)")
    seconds = time.perf_counter() - start

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    torch.save(head.state_dict(), out_dir / "head.pt")
    (out_dir / "labels.json").write_text(json.dumps(labels))
    stats = {"base_model": base_model, "examples": len(examples), "epochs": epochs, "steps": total_steps,
             "train_seconds": round(seconds, 1), "init_train_accuracy": round(init_acc, 4),
             "parameters": sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in head.parameters()),
             "lr": lr, "head_lr": head_lr, "batch_size": batch_size, "max_length": max_length, "seed": seed}
    (out_dir / "train_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


class FineTunedClassifier:
    """Loads a model saved by train() for prediction."""

    def __init__(self, model_dir: str | Path, threads: int | None = None):
        torch = _torch()
        from transformers import AutoModel, AutoTokenizer

        if threads:
            torch.set_num_threads(threads)
        model_dir = Path(model_dir)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.encoder = AutoModel.from_pretrained(model_dir).eval()
        self.labels = json.loads((model_dir / "labels.json").read_text())
        self.head = torch.nn.Linear(self.encoder.config.hidden_size, len(self.labels))
        self.head.load_state_dict(torch.load(model_dir / "head.pt", map_location="cpu"))
        self.head.eval()

    def predict(self, texts: list[str], batch_size: int = 64) -> list[str]:
        torch = self.torch
        emb = torch.nn.functional.normalize(_encode(torch, self.encoder, self.tokenizer, texts, 64, batch_size), dim=-1)
        with torch.inference_mode():
            ids = self.head(emb).argmax(-1).tolist()
        return [self.labels[i] for i in ids]
