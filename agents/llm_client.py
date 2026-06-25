"""Provider-agnostic LLM client with call counting (Phase 1).

Why this exists
---------------
The original code created an OpenAI client at *import time*
(`client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))`), which:
  * crashed the import when no API key was set (breaks tests/CI), and
  * scattered the model name + provider across modules.

This module centralises every chat-JSON call so that:
  * the provider/model is swappable in ONE place (config / env),
  * every call is COUNTED (the Phase 1 cost-reduction verification reads this),
  * tests can inject a mock client with ``set_client`` — no network, no key.

Public surface (module-level, intentionally tiny):
    chat_json(prompt)  -> dict      # the only call the agents need
    call_count()       -> int       # for the cost counter / verification
    reset_count()                    # zero the counter at cycle start
    get_client() / set_client(c)     # swap the active client (tests)
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, Optional

DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


class LLMClient:
    """Thin wrapper over the OpenAI chat-completions JSON mode.

    The underlying SDK client is created lazily on the first real call, so
    importing this module (and the agents) never requires an API key.
    """

    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None):
        self.model = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None
        self._lock = threading.Lock()
        self._call_count = 0

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI  # imported lazily so no key needed at import
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def chat_json(self, prompt: str) -> Dict[str, Any]:
        """Send a single prompt, return parsed JSON. Counts the call."""
        with self._lock:
            self._call_count += 1
        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)

    @property
    def call_count(self) -> int:
        return self._call_count

    def reset_count(self) -> None:
        with self._lock:
            self._call_count = 0


# ----------------------------------------------------------------------------
# Process-wide active client + module-level convenience functions.
# Tests call set_client(MockLLM()) to redirect every agent's LLM traffic.
# ----------------------------------------------------------------------------
_ACTIVE: LLMClient = LLMClient()


def get_client() -> LLMClient:
    return _ACTIVE


def set_client(client: LLMClient) -> None:
    global _ACTIVE
    _ACTIVE = client


def chat_json(prompt: str) -> Dict[str, Any]:
    return _ACTIVE.chat_json(prompt)


def call_count() -> int:
    return _ACTIVE.call_count


def reset_count() -> None:
    _ACTIVE.reset_count()
