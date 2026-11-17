"""Prompts for Claude and parsing of its replies into one of the 77 labels."""
from __future__ import annotations

import difflib
import re

from .data import Example

INVALID = "__invalid__"

INSTRUCTIONS = """You classify customer messages sent to a bank's support team into exactly one intent.

The intents are the labels below. Pick the single label that best describes what the customer needs.
Reply with the label only, exactly as written (lowercase with underscores), with no other text."""


def zero_shot_system(labels: list[str]) -> str:
    return f"{INSTRUCTIONS}\n\nLabels:\n" + "\n".join(labels)


def few_shot_system(labels: list[str], examples: list[Example]) -> str:
    """One real training message per label, shown next to its label."""
    lines = [f"{ex.label}: {ex.text}" for ex in examples]
    return (f"{INSTRUCTIONS}\n\nLabels:\n" + "\n".join(labels) +
            "\n\nOne example message for each label (label: message):\n" + "\n".join(lines))


def user_message(text: str) -> str:
    return f"Customer message:\n{text}\n\nLabel:"


def parse_label(reply: str, labels: list[str]) -> str:
    """Map a reply to a valid label. Accepts quotes, backticks, spaces or capitals around the label;
    a near-miss spelling is matched to the closest label; anything else is INVALID."""
    text = reply.strip().strip("`'\".").strip()
    first = text.splitlines()[0].strip() if text else ""
    candidate = re.sub(r"[\s-]+", "_", first.lower()).strip("_:. ")
    candidate = re.sub(r"^(label|intent)_?:?_?", "", candidate)
    valid = set(labels)
    if candidate in valid:
        return candidate
    # A valid label mentioned anywhere in the reply (longest first, so "card_payment_fee_charged"
    # wins over "card_payment")
    for label in sorted(labels, key=len, reverse=True):
        if re.search(rf"\b{re.escape(label)}\b", text.lower()):
            return label
    close = difflib.get_close_matches(candidate, labels, n=1, cutoff=0.85)
    return close[0] if close else INVALID
