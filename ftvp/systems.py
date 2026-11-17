"""The five systems behind one function, so EvalForge can call each the same way.

EvalForge provider: model "ftvp.systems:classify", params {"system": "<name>", ...}.
"""
from __future__ import annotations

import threading
from pathlib import Path

from . import data, prompts
from .llm import Claude

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

_lock = threading.Lock()
_cache: dict = {}


def _cached(key, build):
    with _lock:
        if key not in _cache:
            _cache[key] = build()
        return _cache[key]


def _tfidf(k):
    import joblib

    path = MODELS / f"tfidf-{k or 'all'}.joblib"
    if path.exists():
        return joblib.load(path)
    from .baseline import train

    model, _ = train(data.per_class(data.load("train"), k))
    path.parent.mkdir(exist_ok=True)
    joblib.dump(model, path)
    return model


def classify(prompt: str, system: str | None = None, kind: str = "tfidf", k: int | None = None,
             model: str = "claude-haiku-5-5", shots: str = "zero", model_dir: str | None = None) -> dict:
    """prompt is the customer message. kind: tfidf | finetuned | claude.

    tfidf:     TF-IDF + logistic regression trained on k examples per intent (None = all)
    finetuned: a model saved by `ftvp train` in model_dir
    claude:    Claude `model`, with shots "zero" (label list) or "few" (one example per label)
    """
    labels = _cached("labels", data.labels)
    if kind == "tfidf":
        clf = _cached(("tfidf", k), lambda: _tfidf(k))
        return {"text": clf.predict([prompt])[0], "metadata": {"system": f"tfidf-k{k or 'all'}"}}
    if kind == "finetuned":
        from .finetune import FineTunedClassifier

        path = Path(model_dir or MODELS / "minilm-all")
        if not path.is_absolute():
            path = ROOT / path
        clf = _cached(("ft", str(path)), lambda: FineTunedClassifier(path, threads=1))
        return {"text": clf.predict([prompt])[0], "metadata": {"system": f"finetuned:{path.name}"}}
    if kind == "claude":
        if shots == "few":
            system_prompt = _cached("few", lambda: prompts.few_shot_system(labels, data.few_shot_examples()))
        else:
            system_prompt = _cached("zero", lambda: prompts.zero_shot_system(labels))
        reply = Claude(model).complete(system_prompt, prompts.user_message(prompt))
        label = prompts.parse_label(reply.text, labels)
        return {"text": label, "input_tokens": reply.input_tokens + reply.cache_read_tokens + reply.cache_write_tokens,
                "output_tokens": reply.output_tokens, "cost_usd": reply.cost_usd,
                "metadata": {"raw": reply.text[:200], "cache_read_tokens": reply.cache_read_tokens,
                             "cache_write_tokens": reply.cache_write_tokens, "valid": label != prompts.INVALID}}
    raise ValueError(f"Unknown kind '{kind}'")
