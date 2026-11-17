import io
import json
import urllib.request
from collections import Counter

import pytest

from ftvp import data, metrics, prompts
from ftvp.baseline import train
from ftvp.llm import Claude, LLMError, cost


def test_dataset_shape():
    train_set, test = data.load("train"), data.load("test")
    assert len(train_set) == 10003 and len(test) == 3080
    assert len(data.labels()) == 77
    assert set(e.label for e in test) == set(data.labels())


def test_per_class_is_reproducible_and_balanced():
    train_set = data.load("train")
    a, b = data.per_class(train_set, 5, seed=0), data.per_class(train_set, 5, seed=0)
    assert a == b and len(a) == 77 * 5
    assert set(Counter(e.label for e in a).values()) == {5}
    assert data.per_class(train_set, 5, seed=1) != a
    assert data.per_class(train_set, None) == train_set


def test_eval_subset_and_few_shot_do_not_leak():
    subset = data.eval_subset()
    assert len(subset) == 770 and set(Counter(e.label for e in subset).values()) == {10}
    shots = data.few_shot_examples()
    assert [e.label for e in shots] == data.labels()
    test_texts = {e.text for e in data.load("test")}
    assert not any(e.text in test_texts for e in shots)


def test_metrics():
    gold = ["a", "a", "b", "c"]
    pred = ["a", "b", "b", "x"]
    assert metrics.accuracy(gold, pred) == 0.5
    # F1: a = 2*1*0.5/1.5 = 0.667, b = 2*0.5*1/1.5 = 0.667, c = 0
    assert metrics.macro_f1(gold, pred) == pytest.approx((2 / 3 + 2 / 3 + 0) / 3)
    assert metrics.top_confusions(gold, pred, 1)[0][2] == 1


@pytest.mark.parametrize("reply,expected", [
    ("card_arrival", "card_arrival"),
    ("  `Card_Arrival`.", "card_arrival"),
    ("Label: lost_or_stolen_card", "lost_or_stolen_card"),
    ("The intent is card_payment_fee_charged because the fee", "card_payment_fee_charged"),
    ("card arrival", "card_arrival"),
    ("card_arival", "card_arrival"),
    ("I'm not sure which applies", prompts.INVALID),
    ("", prompts.INVALID),
])
def test_parse_label(reply, expected):
    assert prompts.parse_label(reply, data.labels()) == expected


def test_prompts_list_every_label():
    labels = data.labels()
    zero = prompts.zero_shot_system(labels)
    few = prompts.few_shot_system(labels, data.few_shot_examples())
    assert all(label in zero and label in few for label in labels)
    assert len(few) > len(zero)
    assert prompts.user_message("hi").startswith("Customer message:")


def test_baseline_learns_from_few_examples():
    model, seconds = train(data.per_class(data.load("train"), 5))
    subset = data.eval_subset()
    acc = metrics.accuracy([e.label for e in subset], list(model.predict([e.text for e in subset])))
    assert acc > 0.55 and seconds < 60


def test_cost_with_cache():
    # Haiku 5.5: $0.10 in, $0.50 out per 1M; cache read 0.1x, write 1.25x
    assert cost("claude-haiku-5-5", 1_000_000, 0) == pytest.approx(0.10)
    assert cost("claude-haiku-5-5", 0, 0, cache_read=1_000_000) == pytest.approx(0.01)
    assert cost("claude-haiku-5-5", 0, 0, cache_write=1_000_000) == pytest.approx(0.125)
    assert cost("unknown-model", 1000, 1000) == 0.0


def test_claude_client_sends_cached_system_and_parses_usage(monkeypatch):
    sent = {}

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=0):
        sent["body"] = json.loads(req.data)
        sent["key"] = req.get_header("X-api-key")
        return Resp(json.dumps({"content": [{"type": "text", "text": "card_arrival"}],
                                "usage": {"input_tokens": 20, "output_tokens": 3,
                                          "cache_read_input_tokens": 1500, "cache_creation_input_tokens": 0}}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    reply = Claude("claude-haiku-5-5", api_key=" sk-test \n").complete("SYSTEM", "USER")
    assert sent["key"] == "sk-test"  # whitespace from a pasted secret is stripped
    assert sent["body"]["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert reply.text == "card_arrival" and reply.cache_read_tokens == 1500
    assert reply.cost_usd == pytest.approx(cost("claude-haiku-5-5", 20, 3, cache_read=1500))


def test_claude_needs_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError):
        Claude()
