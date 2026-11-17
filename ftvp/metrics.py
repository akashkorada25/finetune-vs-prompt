"""Classification metrics, written out so they are easy to check."""
from __future__ import annotations

from collections import Counter


def accuracy(gold: list[str], pred: list[str]) -> float:
    return sum(g == p for g, p in zip(gold, pred)) / len(gold) if gold else 0.0


def macro_f1(gold: list[str], pred: list[str], labels: list[str] | None = None) -> float:
    """Unweighted mean of per-label F1 over the gold labels (invalid predictions count as wrong)."""
    labels = labels or sorted(set(gold))
    tp, fp, fn = Counter(), Counter(), Counter()
    for g, p in zip(gold, pred):
        if g == p:
            tp[g] += 1
        else:
            fn[g] += 1
            fp[p] += 1
    scores = []
    for label in labels:
        precision = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 0.0
        recall = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def top_confusions(gold: list[str], pred: list[str], n: int = 5) -> list[tuple[str, str, int]]:
    pairs = Counter((g, p) for g, p in zip(gold, pred) if g != p)
    return [(g, p, c) for (g, p), c in pairs.most_common(n)]
