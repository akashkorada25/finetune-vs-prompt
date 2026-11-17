import json
import sys
from pathlib import Path

import pytest

from ftvp import data, report, systems
from ftvp.prompts import INVALID

ROOT = Path(__file__).resolve().parents[1]


def test_tfidf_system_through_the_provider_function(tmp_path, monkeypatch):
    monkeypatch.setattr(systems, "MODELS", tmp_path)
    out = systems.classify("My card still hasn't arrived after two weeks", kind="tfidf", k=10)
    assert out["text"] in data.labels()
    assert (tmp_path / "tfidf-10.joblib").exists()
    with pytest.raises(ValueError):
        systems.classify("x", kind="magic")


def test_suite_builds_and_loads():
    evalforge_config = pytest.importorskip("evalforge.config")
    sys.path.insert(0, str(ROOT / "eval"))
    import build_suite
    import yaml

    for local_only, n in ((True, 2), (False, 5)):
        suite_dict = build_suite.build(local_only)
        path = ROOT / "eval" / f"_test_suite_{local_only}.yaml"
        try:
            path.write_text(yaml.safe_dump(suite_dict))
            suite = evalforge_config.load_suite(path)
        finally:
            path.unlink(missing_ok=True)
        assert len(suite.providers) == n and len(suite.cases) == 770
        assert suite.cases[0].expected == data.eval_subset()[0].label


class FakeResult:
    def __init__(self, provider, case_id, output, latency=10.0, cost=0.0, error=None, metadata=None):
        self.provider_id, self.case_id, self.output = provider, case_id, output
        self.latency_ms, self.cost_usd, self.error, self.metadata = latency, cost, error, metadata or {}


def test_report_summarise():
    subset = data.eval_subset()
    results = [FakeResult("haiku-few", "s000", subset[0].label, cost=0.0002),
               FakeResult("haiku-few", "s001", INVALID, cost=0.0002),
               FakeResult("tfidf", "s000", subset[0].label),
               FakeResult("tfidf", "s001", "wrong", error="boom")]
    rows = {r["system"]: r for r in report.summarise(results)}
    assert rows["haiku-few"]["accuracy"] == 0.5 and rows["haiku-few"]["invalid"] == 0.5
    assert rows["haiku-few"]["cost_per_1k"] == pytest.approx(0.2)
    assert rows["tfidf"]["errors"] == 1 and rows["tfidf"]["invalid"] == 0.5
    assert [r["system"] for r in report.summarise(results)] == ["tfidf", "haiku-few"]
    assert "$0 (runs locally)" in report.table(report.summarise(results))


def test_curve_table(tmp_path):
    path = tmp_path / "curve.json"
    path.write_text(json.dumps([
        {"method": "tfidf", "k": "5", "examples": 385, "subset_accuracy": 0.67},
        {"method": "finetuned", "k": "5", "examples": 385, "subset_accuracy": 0.8},
        {"method": "tfidf", "k": "all", "examples": 10003, "subset_accuracy": 0.918},
    ]))
    t = report.curve_table(path)
    assert "| 5 | 385 | 67.0% | 80.0% |" in t and "| all | 10,003 | 91.8% | - |" in t
    assert report.curve_table(tmp_path / "missing.json") == ""


def test_finetune_smoke(tmp_path):
    """Runs only where torch is installed (the benchmark workflow)."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from ftvp.finetune import FineTunedClassifier, train

    examples = [e for e in data.load("train") if e.label in ("card_arrival", "exchange_rate")][:40]
    # The real base model, so the smoke test also checks that it downloads and loads
    stats = train(examples, ["card_arrival", "exchange_rate"], tmp_path / "m", epochs=1, min_steps=4,
                  batch_size=16, lr=1e-4, log=lambda *_: None)
    assert stats["steps"] >= 4 and (tmp_path / "m" / "train_stats.json").exists()
    preds = FineTunedClassifier(tmp_path / "m").predict(["Where is my card?", "What exchange rate do you use?"])
    assert set(preds) <= {"card_arrival", "exchange_rate"}
