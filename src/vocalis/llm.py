from __future__ import annotations

from typing import Iterator

from .logger import get_logger

log = get_logger(__name__)


class History:
    """Chat history with a fixed system prompt, sliding window, and optional compaction."""

    def __init__(self, system_prompt: str, max_messages: int = 40, compact_after: int = 30):
        self._system = {"role": "system", "content": system_prompt}
        self._max = max(2, max_messages)
        self._compact_after = max(10, compact_after)
        self._turns: list[dict] = []
        self._summary: str | None = None  # compacted summary of older turns

    def add_user(self, text: str) -> None:
        self._turns.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, text: str) -> None:
        self._turns.append({"role": "assistant", "content": text})
        self._trim()

    def messages(self) -> list[dict]:
        # If we have a summary, inject it as a system note after the main system prompt
        if self._summary:
            return [self._system, {"role": "system", "content": self._summary}] + self._turns
        return [self._system] + self._turns

    def clear(self) -> None:
        """Forget all previous turns, keep only system prompt."""
        self._turns.clear()
        self._summary = None

    def compact(self, keep: int | None = None) -> str | None:
        """Compact older history into a summary, keeping `keep` most recent turns. Returns summary."""
        keep = keep if keep is not None else self._compact_after
        if len(self._turns) <= keep:
            return None
        # Older turns to compact
        older = self._turns[:-keep]
        recent = self._turns[-keep:]
        # Build a deterministic compact summary (no LLM call — keeps it offline/fast).
        # Summarize as: last N turns + key facts.
        lines: list[str] = []
        for m in older[-12:]:  # last 12 of the older chunk to avoid huge summary
            role = m.get("role", "?")
            content = (m.get("content") or "").strip().replace("\n", " ")
            if len(content) > 140:
                content = content[:137] + "..."
            if content:
                lines.append(f"{role}: {content}")
        summary = (
            f"[Conversation so far — {len(older)} earlier messages compacted. "
            f"Recent context kept: {len(recent)} messages. Key excerpts: "
            + " | ".join(lines)
            + "]"
        )
        self._summary = summary
        self._turns = recent
        log.debug("History compacted: %d -> %d turns (compact_after=%d)", len(older)+len(recent), len(recent), self._compact_after)
        return summary

    def _trim(self) -> None:
        # Compact when we exceed compact_after (preserve mode), otherwise trim to max
        if len(self._turns) > self._compact_after:
            # Keep at most compact_after messages; compact older ones into summary
            self.compact(keep=self._compact_after)
        overflow = len(self._turns) - self._max
        if overflow > 0:
            del self._turns[:overflow]


def _is_placeholder_key(k: str) -> bool:
    return (k or "").strip().lower() in ("", "none", "vocalis-local", "ollama", "lm-studio", "unsloth", "test")


class LLMClient:
    """Streaming client for any OpenAI-compatible chat completions endpoint."""

    def __init__(self, base_url: str, model: str, api_key: str = "", temperature: float = 0.7):
        from openai import OpenAI

        # API key is optional for local providers (Ollama/LM Studio/Unsloth) — don't send placeholder tokens
        # OpenAI client requires *some* key, so use "not-needed" internally but remember if it was placeholder
        raw_key = (api_key or "").strip()
        self._api_key_raw = raw_key
        self._is_placeholder = _is_placeholder_key(raw_key)
        wire_key = raw_key if raw_key and not self._is_placeholder else "not-needed"
        self._client = OpenAI(base_url=base_url, api_key=wire_key)
        # Normalize model id like config does (handles leading slash, Windows paths)
        self.model = self._norm_id(model)
        if not self.model and model:
            self.model = (model or "").strip()
        self.temperature = temperature
        self._base_url_raw = base_url
        log.debug("LLMClient base_url=%s model=%s temp=%.2f api_key=%s", base_url, self.model, temperature, "set" if raw_key else "none")

    def _ensure_lmstudio_model_loaded(self, timeout: float = 60.0) -> bool:
        """If base_url looks like LM Studio, ensure the selected model is loaded.

        LM Studio exposes POST {origin}/api/v1/models/load (REST API) which is separate
        from the OpenAI-compatible /v1/chat/completions. The OpenAI endpoint does not
        auto-load evicted models (TTL/auto-evict). We proactively load here so the
        prompt request doesn't fail with 'model not loaded' / 404.

        Returns True if we attempted and succeeded (or already loaded), False if we
        skipped (not LM Studio) or failed — caller should still try the chat request.
        """
        # Heuristic: LM Studio default is 1234, but user may use any port. Try the REST load
        # endpoint only if the origin responds to it; otherwise no-op.
        try:
            from urllib.parse import urlparse
            import httpx

            raw = str(self._base_url_raw).rstrip("/")
            parsed = urlparse(raw if "://" in raw else f"http://{raw}")
            origin = f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else raw
            # LM Studio REST load is at /api/v1/models/load (not /v1/models)
            # Derive origin from base_url: e.g. http://localhost:1234/v1 -> http://localhost:1234
            load_url = f"{origin}/api/v1/models/load"
            # Don't spam if model is empty
            if not self.model or self.model.strip().lower() in ("", "none"):
                return False
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if not self._is_placeholder and self._api_key_raw:
                headers["Authorization"] = f"Bearer {self._api_key_raw}"
            # Quick check: if origin is clearly Ollama (11434) we still try, but it will 404 and we ignore.
            # Use short connect timeout to avoid hanging headless.
            payload = {"model": self.model}
            log.info("LM Studio ensure-loaded: POST %s model=%r", load_url, self.model)
            with httpx.Client(timeout=timeout) as c:
                r = c.post(load_url, json=payload, headers=headers)
                if r.status_code == 200:
                    try:
                        data = r.json()
                        log.info("LM Studio model loaded: %s (%.1fs)", data.get("instance_id") or self.model, data.get("load_time_seconds", 0))
                    except Exception:
                        log.info("LM Studio model loaded: %s", self.model)
                    return True
                elif r.status_code == 404:
                    # Not LM Studio (Ollama/Unsloth) — silently skip
                    log.debug("LM Studio load endpoint not found (404) — not LM Studio, skipping")
                    return False
                else:
                    # 400 may mean already loaded, downloading, or unknown model — log and still try chat
                    txt = r.text[:400] if hasattr(r, "text") else ""
                    log.warning("LM Studio load returned %s: %s — will still try chat", r.status_code, txt[:200])
                    return r.status_code in (200, 400)  # 400 sometimes means already loaded with different config
        except Exception as e:
            log.debug("LM Studio ensure-loaded skipped/failed: %s", e)
        return False

    def stream_reply(self, messages: list[dict], tools: list[dict] | None = None) -> Iterator[str]:
        # LM Studio: ensure selected model is loaded before the prompt (auto-evict/TTL may have unloaded it)
        # Do this once per turn, before the chat request, so the prompt doesn't fail with 404/model_not_loaded
        try:
            self._ensure_lmstudio_model_loaded()
        except Exception:
            pass
        # tools for timer, etc. Stored for caller to inspect after stream
        self._pending_tool_calls: list[dict] = []
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        log.info("LLM stream start model=%s msgs=%d tools=%s", self.model, len(messages), bool(tools))
        tool_calls: dict[int, dict] = {}
        try:
            with self._client.chat.completions.create(**kwargs) as stream:  # type: ignore[arg-type]
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta  # type: ignore[attr-defined]
                    # content
                    content = getattr(delta, "content", None)
                    if content:
                        log.debug("LLM delta: %r", content[:80])
                        yield content
                    # tool calls (streaming)
                    tcs = getattr(delta, "tool_calls", None)
                    if tcs:
                        for tc in tcs:
                            idx = getattr(tc, "index", 0)
                            if idx not in tool_calls:
                                tool_calls[idx] = {"id": getattr(tc, "id", f"call_{idx}"), "type": "function", "function": {"name": "", "arguments": ""}}
                            fn = getattr(tc, "function", None)
                            if fn:
                                if getattr(fn, "name", None):
                                    tool_calls[idx]["function"]["name"] = fn.name  # type: ignore
                                if getattr(fn, "arguments", None):
                                    tool_calls[idx]["function"]["arguments"] += fn.arguments  # type: ignore
            if tool_calls:
                self._pending_tool_calls = list(tool_calls.values())
                log.info("LLM tool calls collected: %s", self._pending_tool_calls)
            else:
                self._pending_tool_calls = []
            log.info("LLM stream done")
        except BaseException as e:
            msg = str(e).lower()
            # If model was evicted (LM Studio TTL) the chat may 404. Try one load+retry.
            if any(s in msg for s in ("model not found", "model not loaded", "model_not_loaded", "no model loaded", "failed to load model", "404")):
                log.warning("LLM stream failed with model-not-loaded, trying ensure-loaded + retry: %s", e)
                try:
                    if self._ensure_lmstudio_model_loaded(timeout=90.0):
                        # retry once without recursion on tools
                        for chunk in self.stream_reply(messages, tools=tools):
                            yield chunk
                        return
                except Exception as re:
                    log.debug("Retry after load also failed: %s", re)
            # If tools not supported, retry without tools
            if tools and ("tools" in str(e).lower() or "tool" in str(e).lower()):
                log.warning("LLM tools not supported, retrying without tools: %s", e)
                self._pending_tool_calls = []
                # retry without tools
                for chunk in self.stream_reply(messages, tools=None):
                    yield chunk
                return
            log.exception("LLM stream failed base_url=%s model=%s: %s", self._client.base_url, self.model, e)
            raise

    @property
    def pending_tool_calls(self) -> list[dict]:
        return getattr(self, "_pending_tool_calls", [])

    def check(self) -> str:
        log.info("LLM check base_url=%s model=%s", self._client.base_url, self.model)
        # Proactive load for LM Studio so Test LLM doesn't fail when model is cold
        try:
            self._ensure_lmstudio_model_loaded()
        except Exception:
            pass
        try:
            r = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                max_tokens=8,
                temperature=0.0,
            )
            out = (r.choices[0].message.content or "").strip()
            log.info("LLM check ok: %r", out)
            return out
        except BaseException as e:
            msg = str(e).lower()
            # If model was cold, try one load+retry before falling back to auth handling
            if any(s in msg for s in ("model not found", "model not loaded", "no model loaded", "failed to load model", "404")):
                log.warning("LLM check failed with model-not-loaded, trying ensure-loaded + retry: %s", e)
                try:
                    if self._ensure_lmstudio_model_loaded(timeout=90.0):
                        # retry once
                        r = self._client.chat.completions.create(
                            model=self.model,
                            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                            max_tokens=8,
                            temperature=0.0,
                        )
                        out = (r.choices[0].message.content or "").strip()
                        if out:
                            log.info("LLM check ok after load retry: %r", out)
                            return out
                except Exception as re:
                    log.debug("LLM check retry after load failed: %s", re)
            msg2 = str(e)
            is_auth = "401" in msg2 or "authentication" in msg2.lower() or "Invalid token" in msg2
            if is_auth and self._is_placeholder:
                log.info("LLM check got 401 with placeholder key — retrying without auth header")
                try:
                    import httpx

                    url = str(self._client.base_url).rstrip("/") + "/chat/completions"
                    payload = {
                        "model": self.model,
                        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
                        "max_tokens": 8,
                        "temperature": 0.0,
                    }
                    with httpx.Client(timeout=10.0) as c:
                        r = c.post(url, json=payload)
                        r.raise_for_status()
                        data = r.json()
                        out = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
                        if out:
                            log.info("LLM check ok via fallback: %r", out)
                            return out
                except Exception as fe:
                    log.debug("LLM check fallback also failed: %s", fe)
            log.exception("LLM check failed: %s", e)
            raise

    @staticmethod
    def _norm_id(raw: str) -> str:
        """Normalize provider model ids: strip leading slashes, handle file paths, keep namespace like 'unsloth/...'."""
        s = (raw or "").strip()
        if not s:
            return ""
        s = s.lstrip("/\\").strip()
        low = s.lower()
        if low.endswith((".gguf", ".bin", ".onnx")):
            if "\\" in s:
                s = s.replace("\\", "/").split("/")[-1]
            elif "/" in s and ":" not in s:
                s = s.split("/")[-1]
        return s.strip()

    def list_models(self) -> list[str]:
        """Fetch available model ids from the provider via /v1/models."""
        log.info("LLM list_models base_url=%s", self._client.base_url)
        # Try via OpenAI client first (sends placeholder key as "not-needed" — local servers ignore it)
        try:
            resp = self._client.models.list()
            ids = sorted({self._norm_id(m.id) for m in resp.data if getattr(m, "id", None) and self._norm_id(m.id)})
            log.info("LLM list_models got %d: %s", len(ids), ids[:20])
            if ids:
                return ids
            raise ValueError("No ids in /v1/models")
        except BaseException as e:
            # If 401 with placeholder key, retry without auth header via raw httpx
            msg = str(e)
            is_auth_err = "401" in msg or "authentication" in msg.lower() or "Invalid token" in msg
            if is_auth_err and self._is_placeholder:
                log.info("LLM list_models got 401 with placeholder key — retrying without auth")
            else:
                log.warning("LLM list_models failed (will try fallback): %s", e)
            try:
                import httpx

                url = str(self._client.base_url).rstrip("/") + "/models"
                # For fallback, don't send placeholder tokens — many local servers reject them
                headers: dict[str, str] = {}
                if not self._is_placeholder and self._api_key_raw:
                    headers["Authorization"] = f"Bearer {self._api_key_raw}"
                with httpx.Client(timeout=5.0) as c:
                    r = c.get(url, headers=headers)
                    r.raise_for_status()
                    data = r.json()
                    ids: list[str] = []
                    if isinstance(data, dict) and "data" in data:
                        ids = sorted({self._norm_id(m.get("id") or m.get("name") or "") for m in data["data"] if m.get("id") or m.get("name")})
                        ids = [i for i in ids if i]
                        if ids:
                            log.info("LLM fallback /v1/models got %d", len(ids))
                            return ids
                    if isinstance(data, dict) and "models" in data:
                        ids = sorted({self._norm_id(m.get("name") or "") for m in data["models"] if m.get("name")})
                        ids = [i for i in ids if i]
                        if ids:
                            log.info("LLM fallback /v1/models (Ollama) got %d", len(ids))
                            return ids
            except Exception as fe:
                log.debug("LLM list_models fallback also failed: %s", fe)
            raise
