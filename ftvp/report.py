"""Head-to-head report from an EvalForge run of eval/suite.yaml, plus the learning curve."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from . import data, metrics
from .prompts import INVALID

ORDER = ["tfidf", "finetuned", "haiku-zero", "haiku-few", "sonnet-few"]
NAMES = {
    "tfidf": "TF-IDF + logistic regression",
    "finetuned": "Fine-tuned MiniLM (33M)",
    "haiku-zero": "Claude Haiku 5.5, zero-shot",
    "haiku-few": "Claude Haiku 5.5, few-shot",
    "sonnet-few": "Claude Sonnet 5.5, few-shot",
}


def summarise(results) -> list[dict]:
    """results: EvalForge Results; case ids are 's<index into data.eval_subset()>'."""
    subset = data.eval_subset()
    labels = data.labels()
    rows = []
    by_system: dict[str, list] = {}
    for r in results:
        by_system.setdefault(r.provider_id, []).append(r)
    for system, rs in by_system.items():
        rs = sorted(rs, key=lambda r: r.case_id)
        gold = [subset[int(r.case_id[1:])].label for r in rs]
        pred = [r.output.strip() if not r.error else INVALID for r in rs]
        latencies = [r.latency_ms for r in rs if not r.error]
        rows.append({
            "system": system,
            "n": len(rs),
            "accuracy": metrics.accuracy(gold, pred),
            "macro_f1": metrics.macro_f1(gold, pred, labels),
            "invalid": sum(p == INVALID for p in pred) / len(pred),
            "errors": sum(1 for r in rs if r.error),
            "p50_latency_ms": statistics.median(latencies) if latencies else 0.0,
            "cost_per_1k": 1000 * sum(r.cost_usd for r in rs) / len(rs),
            "cache_read_tokens": sum(r.metadata.get("cache_read_tokens", 0) for r in rs),
            "top_confusions": metrics.top_confusions(gold, pred, 3),
        })
    rows.sort(key=lambda x: ORDER.index(x["system"]) if x["system"] in ORDER else 99)
    return rows


def table(rows: list[dict]) -> str:
    lines = ["| System | Accuracy | Macro-F1 | Invalid outputs | Median latency | API cost per 1,000 messages |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        cost = f"${r['cost_per_1k']:.2f}" if r["cost_per_1k"] else "$0 (runs locally)"
        lines.append(f"| {NAMES.get(r['system'], r['system'])} | {r['accuracy']:.1%} | {r['macro_f1']:.3f} "
                     f"| {r['invalid']:.1%} | {r['p50_latency_ms']:.0f} ms | {cost} |")
    return "\n".join(lines)


def curve_table(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    rows = json.loads(path.read_text())
    lines = ["| Examples per intent | Training examples | TF-IDF + LR | Fine-tuned MiniLM |", "|---|---|---|---|"]
    ks = []
    for r in rows:
        if r["k"] not in ks:
            ks.append(r["k"])
    for k in ks:
        got = {r["method"]: r for r in rows if r["k"] == k}
        n = next(iter(got.values()))["examples"]
        cells = [f"{got[m]['subset_accuracy']:.1%}" if m in got else "-" for m in ("tfidf", "finetuned")]
        lines.append(f"| {k} | {n:,} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run(db: str, run_id: str, out: str | None, curve: str | None) -> int:
    from evalforge.storage import Store

    store = Store(db)
    run_info = store.get_run(run_id)
    rows = summarise(store.load_results(run_info["id"]))
    text = (f"# Head-to-head: 770 test messages (10 per intent)\n\nEvalForge run `{run_info['id']}`.\n\n"
            f"{table(rows)}\n")
    for r in rows:
        if r["top_confusions"]:
            pairs = "; ".join(f"{g} -> {p} ({c})" for g, p, c in r["top_confusions"])
            text += f"\n- {NAMES.get(r['system'], r['system'])}: most common mistakes: {pairs}"
    ct = curve_table(curve) if curve else ""
    if ct:
        text += f"\n\n## Learning curve (accuracy on the same 770 messages)\n\n{ct}\n"
    print(text)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text + "\n", encoding="utf-8")
    Path(out or "results/head_to_head.md").with_suffix(".json").write_text(json.dumps(rows, indent=2))
    return 0
