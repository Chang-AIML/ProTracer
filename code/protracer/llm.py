"""OpenRouter chat client: every model call of the pipeline and the judge goes here."""
from __future__ import annotations

import base64
import http.client
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
_RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504, 524}


def text(content: str) -> dict:
    return {"type": "text", "text": content}


def image(jpeg: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}}


def message(role: str, content: str | list[dict]) -> dict:
    return {"role": role, "content": content}


class OpenRouterError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status

    @property
    def fatal(self) -> bool:
        """Invalid key or no credit: every further call would fail as well."""
        return self.status in (401, 402)


@dataclass
class Reply:
    text: str
    usage: dict = field(default_factory=dict)

    @property
    def cost(self) -> float:
        return float(self.usage.get("cost") or 0.0)


class OpenRouter:
    """Chat completions on OpenRouter; the key is read from OPENROUTER_API_KEY.

    Transient failures (rate limits, 5xx, broken connections, empty replies)
    are retried with jittered exponential backoff. `spent` sums the billed cost
    of every call, retries included.
    """

    def __init__(self, model: str, reasoning_effort: str | None = "high",
                 max_tokens: int | None = None, retries: int = 8, timeout: float = 900):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise OpenRouterError("set OPENROUTER_API_KEY")
        self.model, self.reasoning_effort, self.max_tokens = model, reasoning_effort, max_tokens
        self.retries, self.timeout = retries, timeout
        self.spent, self._lock = 0.0, threading.Lock()

    def chat(self, messages: list[dict]) -> Reply:
        payload = {"model": self.model, "messages": messages, "usage": {"include": True}}
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        request = urllib.request.Request(ENDPOINT, data=json.dumps(payload).encode(), method="POST",
                                         headers={"Authorization": f"Bearer {self.api_key}",
                                                  "Content-Type": "application/json"})
        delay, cost = 8.0, 0.0
        for attempt in range(1, self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read())
                usage = body.get("usage") or {}
                cost += float(usage.get("cost") or 0.0)
                content = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
                if content.strip():
                    with self._lock:
                        self.spent += cost
                    return Reply(content, {**usage, "cost": cost})
                problem = f"empty reply: {body.get('error') or body.get('choices')}"
            except urllib.error.HTTPError as e:
                if e.code not in _RETRYABLE:
                    detail = e.read().decode(errors="replace")[:500]
                    raise OpenRouterError(f"OpenRouter HTTP {e.code}: {detail}", e.code) from e
                problem = f"HTTP {e.code}"
            except (OSError, http.client.HTTPException, json.JSONDecodeError) as e:   # dropped or truncated
                problem = f"{type(e).__name__}: {e}"
            if attempt == self.retries:
                with self._lock:
                    self.spent += cost
                raise OpenRouterError(f"OpenRouter {self.model}: {problem} (gave up after {attempt} attempts)")
            time.sleep(delay * random.uniform(0.7, 1.4))
            delay = min(delay * 1.6, 90.0)
