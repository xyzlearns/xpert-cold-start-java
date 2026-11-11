"""OpenAI-compatible LLM client with .env autoloading, disk caching, retries, and usage logging."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def load_dotenv_if_present(env_path: str | Path = ".env") -> None:
    """Load key=value pairs from .env into os.environ if not already set in the environment."""
    path = Path(env_path)
    if not path.exists():
        # Also check parent/repo root if running from another working directory
        alt = Path(__file__).resolve().parents[1] / ".env"
        if alt.exists():
            path = alt
        else:
            return
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except OSError:
        pass


class LLMClient:
    """OpenAI-compatible LLM client with disk caching, retries, and usage.jsonl logging."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        cache_dir: Optional[str | Path] = ".cache/llm",
        usage_file: str | Path = "usage.jsonl",
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        load_dotenv_if_present()
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or "").rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or ""
        self.model = model or os.environ.get("LLM_MODEL") or ""
        use_cache_env = os.environ.get("XPERT_USE_CACHE", "1").lower() not in ("0", "false", "no")
        self.cache_dir = Path(cache_dir) if (cache_dir and use_cache_env) else None
        self.usage_file = Path(usage_file)
        self.timeout = timeout or int(os.environ.get("XPERT_LLM_TIMEOUT", "60"))
        self.max_retries = max_retries or max(1, int(os.environ.get("XPERT_LLM_RETRIES", "2")))

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0

    @property
    def is_available(self) -> bool:
        return bool(
            self.base_url
            and self.api_key
            and self.model
            and self.api_key != "replace-me"
            and "your-openai-compatible-endpoint" not in self.base_url
        )

    def _cache_key(self, payload: Dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if not self.is_available:
            return None

        local_ollama = self.base_url in {"http://127.0.0.1:11434/v1", "http://localhost:11434/v1"}
        disable_json_mode = os.environ.get("XPERT_DISABLE_JSON_MODE", "").lower() in {"1", "true", "yes"}

        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if response_format is not None and not disable_json_mode:
            body["response_format"] = response_format
        if os.environ.get("XPERT_DISABLE_THINKING", "").lower() in {"1", "true", "yes"}:
            body["reasoning_effort"] = "none"

        if self.cache_dir is not None:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                ckey = self._cache_key(body)
                cfile = self.cache_dir / f"{ckey}.json"
                if cfile.exists():
                    cached = json.loads(cfile.read_text(encoding="utf-8"))
                    if isinstance(cached, dict) and "content" in cached:
                        return cached["content"]
            except OSError:
                pass

        for attempt in range(self.max_retries):
            try:
                if local_ollama:
                    endpoint = self.base_url[:-3] + "/api/chat"
                    req_body = {
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                        "format": "json",
                        "think": False,
                        "options": {"num_predict": max_tokens or 1024, "temperature": temperature},
                    }
                else:
                    endpoint = f"{self.base_url}/chat/completions"
                    req_body = body

                req = urllib.request.Request(
                    endpoint,
                    data=json.dumps(req_body, ensure_ascii=False).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://goals.xpert.chat",
                        "X-Title": "Xpert Cold Start",
                    },
                )
                started = time.time()
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))

                elapsed = round(time.time() - started, 2)
                if local_ollama:
                    msg_obj = data.get("message") or {}
                    content = msg_obj.get("content")
                    pt = int(data.get("prompt_eval_count") or 0)
                    ct = int(data.get("eval_count") or 0)
                else:
                    usage = data.get("usage") or {}
                    pt = int(usage.get("prompt_tokens") or 0)
                    ct = int(usage.get("completion_tokens") or 0)
                    msg_obj = data["choices"][0]["message"]
                    content = msg_obj.get("content")
                    if isinstance(content, list):
                        content = "".join(
                            part.get("text", "")
                            for part in content
                            if isinstance(part, dict) and isinstance(part.get("text"), str)
                        )

                self.total_prompt_tokens += pt
                self.total_completion_tokens += ct
                self.total_calls += 1

                try:
                    with open(self.usage_file, "a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                {
                                    "model": self.model,
                                    "prompt_tokens": pt,
                                    "completion_tokens": ct,
                                    "seconds": elapsed,
                                }
                            )
                            + "\n"
                        )
                except OSError:
                    pass

                if not isinstance(content, str) or not content.strip():
                    return None

                if self.cache_dir is not None:
                    try:
                        ckey = self._cache_key(body)
                        cfile = self.cache_dir / f"{ckey}.json"
                        cfile.write_text(
                            json.dumps({"content": content}, ensure_ascii=False),
                            encoding="utf-8",
                        )
                    except OSError:
                        pass
                return content

            except urllib.error.HTTPError as exc:
                if exc.code in (400, 422) and "response_format" in body:
                    body.pop("response_format", None)
                    continue
                if exc.code == 429 or exc.code >= 500:
                    if attempt + 1 < self.max_retries:
                        time.sleep(2 ** attempt)
                        continue
                logger.warning("LLM HTTP error %s: %s", exc.code, exc)
                return None
            except Exception as exc:
                if attempt + 1 < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                logger.warning("LLM request failed after retries: %s", exc)
                return None
        return None

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1500,
    ) -> Optional[Any]:
        """Request a structured JSON response from the LLM and parse it safely."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        raw = self.chat(
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        if not raw:
            return None
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start_brace = text.find("{")
            end_brace = text.rfind("}")
            if start_brace != -1 and end_brace > start_brace:
                try:
                    return json.loads(text[start_brace : end_brace + 1])
                except json.JSONDecodeError:
                    pass
            return None
