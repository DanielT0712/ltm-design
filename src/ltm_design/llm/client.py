from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class DeepSeekClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 60.0,
        stream: bool = False,
        log_stream_progress: bool = False,
        thinking: str | None = None,
        max_retries: int = 3,
    ) -> None:
        env = self._load_env()
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or env.get("DEEPSEEK_API_KEY")
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL") or env.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/")
        self.timeout_s = timeout_s
        self.stream = stream
        self.log_stream_progress = log_stream_progress
        self.thinking = thinking
        self.max_retries = max_retries

    def chat(self, *, model: str, messages: list[ChatMessage], temperature: float = 0.0) -> str:
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required for model calls")
        payload = {
            "model": model,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "temperature": temperature,
            "stream": self.stream,
        }
        if self.thinking is not None:
            payload["thinking"] = {"type": self.thinking}
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    if self.stream:
                        return self._read_stream(response)
                    data: dict[str, Any] = json.loads(response.read().decode("utf-8"))
                    return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code not in {429, 500, 502, 503, 504} or attempt >= self.max_retries:
                    raise RuntimeError(f"DeepSeek request failed: HTTP {exc.code}: {body}") from exc
                time.sleep(2 ** attempt)
            except urllib.error.URLError:
                if attempt >= self.max_retries:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("DeepSeek request failed after retries")

    def _read_stream(self, response: Any) -> str:
        start = time.monotonic()
        last_log = start
        chunks = 0
        content_parts: list[str] = []
        reasoning_chars = 0
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunks += 1
            obj = json.loads(data)
            delta = obj["choices"][0].get("delta", {})
            content = delta.get("content") or ""
            reasoning = delta.get("reasoning_content") or ""
            if content:
                content_parts.append(content)
            reasoning_chars += len(reasoning)
            now = time.monotonic()
            if self.log_stream_progress and now - last_log >= 10:
                print(
                    "deepseek stream",
                    f"elapsed_s={now - start:.1f}",
                    f"chunks={chunks}",
                    f"reasoning_chars={reasoning_chars}",
                    f"content_chars={sum(len(part) for part in content_parts)}",
                    flush=True,
                )
                last_log = now
        return "".join(content_parts)

    def _load_env_key(self) -> str | None:
        return self._load_env().get("DEEPSEEK_API_KEY")

    def _load_env(self) -> dict[str, str]:
        env_path = Path.cwd() / ".env"
        if not env_path.exists():
            return {}
        values: dict[str, str] = {}
        for line in env_path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
        return values


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return json.loads(stripped)
