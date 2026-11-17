"""Command line:

  ftvp baseline [--k 10]                         train and evaluate TF-IDF + logistic regression
  ftvp train [--k 10] [--out models/minilm-k10]  fine-tune MiniLM (needs the [train] extra)
  ftvp curve [--ks 5,10,25,all] [--finetune]     learning curve: accuracy vs. examples per intent
  ftvp predict "my card hasn't arrived" [--system tfidf|finetuned|haiku-zero|haiku-few|sonnet-few]
  ftvp report [RUN_ID] [--db .evalforge/results.db]   accuracy, macro-F1, cost and latency per system
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import data, metrics


def _k(value: str | None):
    return None if value in (None, "all") else int(value)


def evaluate(predict, name: str) -> dict:
    """Accuracy and macro-F1 on the 770-message subset and on the full 3,080-message test set."""
    test, subset = data.load("test"), data.eval_subset()
    start = time.perf_counter()
    pred_full = predict([e.text for e in test])
    ms = (time.perf_counter() - start) * 1000 / len(test)
    subset_texts = {e.text: i for i, e in enumerate(test)}
    pred_sub = [pred_full[subset_texts[e.text]] for e in subset]
    gold_full, gold_sub = [e.label for e in test], [e.label for e in subset]
    row = {"system": name,
           "subset_accuracy": metrics.accuracy(gold_sub, pred_sub), "subset_macro_f1": metrics.macro_f1(gold_sub, pred_sub),
           "full_accuracy": metrics.accuracy(gold_full, pred_full), "full_macro_f1": metrics.macro_f1(gold_full, pred_full),
           "ms_per_example_batched": round(ms, 2),
           "top_confusions": metrics.top_confusions(gold_full, pred_full, 5)}
    print(f"{name}: subset acc {row['subset_accuracy']:.3f} | full acc {row['full_accuracy']:.3f} "
          f"macro-F1 {row['full_macro_f1']:.3f} | {ms:.2f} ms/example (batched)", flush=True)
    return row


def cmd_baseline(args) -> int:
    import joblib

    from .baseline import train

    model, seconds = train(data.per_class(data.load("train"), _k(args.k)))
    out = Path("models") / f"tfidf-{args.k or 'all'}.joblib"
    out.parent.mkdir(exist_ok=True)
    joblib.dump(model, out)
    row = evaluate(lambda texts: list(model.predict(texts)), f"tfidf-k{args.k or 'all'}")
    row["train_seconds"] = round(seconds, 1)
    print(json.dumps({k: v for k, v in row.items() if k != "top_confusions"}))
    return 0


def cmd_train(args) -> int:
    from .finetune import FineTunedClassifier, train

    k = _k(args.k)
    out = args.out or f"models/minilm-{args.k or 'all'}"
    stats = train(data.per_class(data.load("train"), k), data.labels(), out, base_model=args.base_model,
                  epochs=args.epochs, lr=args.lr)
    clf = FineTunedClassifier(out)
    row = evaluate(clf.predict, f"finetuned-k{args.k or 'all'}")
    row.update(stats)
    Path(out, "eval.json").write_text(json.dumps(row, indent=2))
    return 0


def cmd_curve(args) -> int:
    from .baseline import train as train_tfidf

    rows = []
    train_set = data.load("train")
    for kv in args.ks.split(","):
        k = _k(kv)
        subset = data.per_class(train_set, k)
        model, seconds = train_tfidf(subset)
        row = evaluate(lambda t: list(model.predict(t)), f"tfidf-k{kv}")
        row.update({"method": "tfidf", "k": kv, "examples": len(subset), "train_seconds": round(seconds, 1)})
        rows.append(row)
        if args.finetune:
            from .finetune import FineTunedClassifier, train

            out = f"models/minilm-{kv}"
            stats = train(subset, data.labels(), out, base_model=args.base_model)
            row = evaluate(FineTunedClassifier(out).predict, f"finetuned-k{kv}")
            row.update({"method": "finetuned", "k": kv, "examples": len(subset),
                        "train_seconds": stats["train_seconds"], "parameters": stats["parameters"]})
            rows.append(row)
            Path(out, "eval.json").write_text(json.dumps(row, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\n| Method | Examples per intent | Training examples | Subset accuracy | Full-test accuracy | Macro-F1 | Train time |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['method']} | {r['k']} | {r['examples']} | {r['subset_accuracy']:.1%} | {r['full_accuracy']:.1%} "
              f"| {r['full_macro_f1']:.3f} | {r['train_seconds']:.0f} s |")
    return 0


SYSTEMS = {
    "tfidf": {"kind": "tfidf"},
    "finetuned": {"kind": "finetuned"},
    "haiku-zero": {"kind": "claude", "model": "claude-haiku-5-5", "shots": "zero"},
    "haiku-few": {"kind": "claude", "model": "claude-haiku-5-5", "shots": "few"},
    "sonnet-few": {"kind": "claude", "model": "claude-sonnet-5-5", "shots": "few"},
}


def cmd_predict(args) -> int:
    from .systems import classify

    out = classify(args.text, **SYSTEMS[args.system])
    print(out["text"])
    if out.get("cost_usd"):
        print(f"(cost ${out['cost_usd']:.6f})", file=sys.stderr)
    return 0


def cmd_report(args) -> int:
    from .report import run

    return run(args.db, args.run_id, args.out, args.curve)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ftvp", description="Fine-tuned small model vs. prompted LLM on Banking77.")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("baseline")
    b.add_argument("--k", default=None, help="examples per intent (default: all)")
    b.set_defaults(fn=cmd_baseline)

    t = sub.add_parser("train")
    t.add_argument("--k", default=None)
    t.add_argument("--out")
    t.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L12-v2")
    t.add_argument("--epochs", type=int, default=6)
    t.add_argument("--lr", type=float, default=8e-5)
    t.set_defaults(fn=cmd_train)

    c = sub.add_parser("curve")
    c.add_argument("--ks", default="5,10,25,all")
    c.add_argument("--finetune", action="store_true")
    c.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L12-v2")
    c.add_argument("--out", default="results/curve.json")
    c.set_defaults(fn=cmd_curve)

    pr = sub.add_parser("predict")
    pr.add_argument("text")
    pr.add_argument("--system", default="tfidf", choices=sorted(SYSTEMS))
    pr.set_defaults(fn=cmd_predict)

    r = sub.add_parser("report")
    r.add_argument("run_id", nargs="?", default="latest")
    r.add_argument("--db", default=".evalforge/results.db")
    r.add_argument("--out", default="results/head_to_head.md")
    r.add_argument("--curve", default="results/curve.json")
    r.set_defaults(fn=cmd_report)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except (ImportError, FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
