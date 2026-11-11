"""Small, guarded OpenAI-compatible client with on-disk request caching."""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


class LlmError(RuntimeError):
    pass


def configured() -> bool:
    return all(os.environ.get(name) for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"))


def ask_json(system: str, prompt: str, *, max_tokens: int = 4000, timeout: int | None = None, use_cache: bool = True) -> object:
    """Return a JSON response, retrying transient model failures. Never logs message content."""
    if not configured():
        raise LlmError("LLM environment variables are not configured")
    if timeout is None:
        timeout = int(os.environ.get("XPERT_LLM_TIMEOUT", "45"))
    retries = max(1, int(os.environ.get("XPERT_LLM_RETRIES", "1")))
    base_url = os.environ["LLM_BASE_URL"].rstrip("/")
    local_ollama = base_url in {"http://127.0.0.1:11434/v1", "http://localhost:11434/v1"}
    gemini_interactions = os.environ.get("XPERT_GEMINI_INTERACTIONS", "").lower() in {"1", "true", "yes"}
    body = {
        "model": os.environ["LLM_MODEL"],
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
    }
    if os.environ.get("XPERT_DISABLE_JSON_MODE", "").lower() not in {"1", "true", "yes"}:
        body["response_format"] = {"type": "json_object"}
    if os.environ.get("XPERT_DISABLE_THINKING", "").lower() in {"1", "true", "yes"}:
        # Ollama maps the OpenAI-compatible "none" effort to disabled thinking for Qwen3.
        body["reasoning_effort"] = "none"
    if local_ollama:
        # Ollama's native endpoint reliably honours this boolean for Qwen3.
        body = {
            "model": body["model"], "messages": body["messages"], "stream": False,
            "format": "json", "think": False,
            "options": {"num_predict": max_tokens, "temperature": 0},
        }
    if gemini_interactions:
        # Gemini 3.x generation is served reliably through its native Interactions API.
        body = {
            "model": body["model"], "store": False,
            "input": system + "\n\n" + prompt,
        }
    encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    cache_path = Path(".cache") / "llm" / f"{digest}.json"
    if use_cache and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    if gemini_interactions:
        api_root = base_url.split("/v1beta", 1)[0]
        endpoint = api_root + "/v1beta/interactions"
        headers = {"x-goog-api-key": os.environ["LLM_API_KEY"], "Content-Type": "application/json"}
    else:
        endpoint = (base_url[:-3] + "/api/chat") if local_ollama else (base_url + "/chat/completions")
        headers = {"Authorization": "Bearer " + os.environ["LLM_API_KEY"], "Content-Type": "application/json"}
    request = urllib.request.Request(
        endpoint,
        data=encoded,
        headers=headers,
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            started = time.monotonic()
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if gemini_interactions:
                texts = []
                for step in payload.get("steps", []):
                    if step.get("type") == "model_output":
                        texts.extend(part.get("text", "") for part in step.get("content", []) if isinstance(part, dict))
                message = {"content": "".join(texts)}
                raw_usage = payload.get("usage") or {}
                usage = {"prompt_tokens": raw_usage.get("total_input_tokens"), "completion_tokens": raw_usage.get("total_output_tokens")}
                finish_reason = payload.get("status")
            elif local_ollama:
                message = payload["message"]
                usage = {"prompt_tokens": payload.get("prompt_eval_count"), "completion_tokens": payload.get("eval_count")}
                finish_reason = "stop" if payload.get("done") else None
            else:
                message = payload["choices"][0]["message"]
                usage = payload.get("usage") or {}
                finish_reason = payload["choices"][0].get("finish_reason")
            content = message.get("content")
            # Some reasoning providers use text parts rather than a plain string.
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") for part in content
                    if isinstance(part, dict) and isinstance(part.get("text"), str)
                )
            if not isinstance(content, str) or not content.strip():
                raise LlmError(
                    "Model returned no text content "
                    f"(finish_reason={finish_reason!r})"
                )
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json.loads(content)
            if use_cache:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            with Path("usage.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps({"model": body["model"], "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"), "seconds": round(time.monotonic() - started, 2)}) + "\n")
            return result
        except urllib.error.HTTPError as exc:
            # Providers put actionable model/parameter details in the JSON error body.
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
            except OSError:
                detail = ""
            last_error = LlmError(f"HTTP {exc.code}: {detail or exc.reason}")
            if attempt + 1 < retries:
                time.sleep(2**attempt)
        except (urllib.error.URLError, urllib.error.HTTPError, http.client.HTTPException, TimeoutError, KeyError, TypeError, json.JSONDecodeError, LlmError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise LlmError(f"LLM request failed after retries: {last_error}")
