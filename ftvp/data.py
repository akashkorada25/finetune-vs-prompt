"""Banking77: customer banking messages labelled with one of 77 intents (CC-BY 4.0, PolyAI)."""
from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "banking77"


@dataclass(frozen=True)
class Example:
    text: str
    label: str


def load(split: str) -> list[Example]:
    """split: 'train' or 'test'."""
    with (DATA_DIR / f"{split}.csv").open(encoding="utf-8", newline="") as f:
        return [Example(r["text"].strip(), r["category"].strip()) for r in csv.DictReader(f)]


def labels() -> list[str]:
    return json.loads((DATA_DIR / "categories.json").read_text())


def per_class(examples: list[Example], k: int | None, seed: int = 0) -> list[Example]:
    """Up to k examples of each label (all of them if k is None), chosen reproducibly, in original order."""
    if k is None:
        return list(examples)
    rng = random.Random(seed)
    by_label: dict[str, list[int]] = {}
    for i, ex in enumerate(examples):
        by_label.setdefault(ex.label, []).append(i)
    keep = set()
    for idxs in by_label.values():
        keep.update(rng.sample(idxs, min(k, len(idxs))))
    return [ex for i, ex in enumerate(examples) if i in keep]


def eval_subset(seed: int = 0) -> list[Example]:
    """The 770-message test subset (10 per intent) used for the head-to-head comparison."""
    return per_class(load("test"), 10, seed)


def few_shot_examples(seed: int = 0) -> list[Example]:
    """One training example per intent, for few-shot prompts. Never drawn from the test set."""
    chosen = per_class(load("train"), 1, seed=seed + 1000)
    order = {label: i for i, label in enumerate(labels())}
    return sorted(chosen, key=lambda ex: order[ex.label])
