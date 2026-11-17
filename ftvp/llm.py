"""Anthropic Messages API client (standard library only) with prompt caching and cost tracking."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

# USD per 1M tokens, Anthropic list prices (Nov 2026). Haiku 5.5's rate is for prompts under 100K tokens.
# Cache writes cost 1.25x the input price and cache reads 0.1x.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}
RETRYABLE = {408, 409, 429, 500, 502, 503, 504, 529}


@dataclass
class Reply:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0


class LLMError(Exception):
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


def cost(model: str, input_tokens: int, output_tokens: int, cache_read: int = 0, cache_write: int = 0) -> float:
    price_in, price_out = PRICES.get(model, (0.0, 0.0))
    return (input_tokens * price_in + cache_write * price_in * 1.25 + cache_read * price_in * 0.1
            + output_tokens * price_out) / 1_000_000


class Claude:
    def __init__(self, model: str = "claude-haiku-5-5", api_key: str | None = None, retries: int = 5):
        self.model = model
        self.api_key = (api_key or os.environ.get("ANTHROPIC_API_KEY") or "").strip()
        if not self.api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self.retries = retries

    def complete(self, system: str, user: str, max_tokens: int = 50, cache_system: bool = True) -> Reply:
        system_block = {"type": "text", "text": system}
        if cache_system:  # the long label list is identical on every call, so cache it
            system_block["cache_control"] = {"type": "ephemeral"}
        payload = {"model": self.model, "max_tokens": max_tokens, "system": [system_block],
                   "messages": [{"role": "user", "content": user}]}
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=json.dumps(payload).encode(), method="POST",
            headers={"content-type": "application/json", "x-api-key": self.api_key,
                     "anthropic-version": "2023-06-01"})
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read())
                break
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")[:300]
                if e.code not in RETRYABLE or attempt == self.retries:
                    raise LLMError(f"HTTP {e.code}: {body}", retryable=e.code in RETRYABLE) from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == self.retries:
                    raise LLMError(f"Network error: {e}", retryable=True) from e
            time.sleep(min(30, 2 ** attempt))
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        u = data.get("usage", {})
        in_tok, out_tok = u.get("input_tokens", 0), u.get("output_tokens", 0)
        read, write = u.get("cache_read_input_tokens", 0) or 0, u.get("cache_creation_input_tokens", 0) or 0
        return Reply(text, in_tok, out_tok, read, write, cost(self.model, in_tok, out_tok, read, write))
