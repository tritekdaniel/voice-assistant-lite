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
        older = self._turns[:-keep]
        recent = self._turns[-keep:]
        lines: list[str] = []
        for m in older[-12:]:
            role = m.get("role", "?")
            content = (m.get("content") or "").strip().replace("\n", " ")
            if len(content) > 140:
                content = content[:137] + "..."
            if content:
                lines.append(f"{role}: {content}")
        new_part = " | ".join(lines)
        if self._summary:
            # accumulate, don't overwrite history older than window
            summary = self._summary + " | " + new_part if new_part else self._summary
            # trim summary to avoid unbounded growth
            if len(summary) > 2000:
                summary = summary[-2000:]
        else:
            summary = (
                f"[Conversation so far — {len(older)} earlier messages compacted. "
                f"Recent context kept: {len(recent)} messages. Key excerpts: "
                + new_part
                + "]"
            )
        self._summary = summary
        self._turns = recent
        log.debug("History compacted: %d -> %d turns (compact_after=%d)", len(older)+len(recent), len(recent), self._compact_after)
        return summary

    def _trim(self) -> None:
        # Only compact when we exceed both compact_after and max; prefer trimming to max
        if len(self._turns) > self._max:
            # If we would lose too much, compact first to preserve summary
            if len(self._turns) > self._compact_after:
                self.compact(keep=self._compact_after)
            overflow = len(self._turns) - self._max
            if overflow > 0:
                del self._turns[:overflow]
        elif len(self._turns) > self._compact_after:
            self.compact(keep=self._compact_after)


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
        # LM Studio: remember last successful ensure to avoid re-loading every turn
        self._lmstudio_last_ok: float = 0.0
        self._lmstudio_last_model: str | None = None
        log.debug("LLMClient base_url=%s model=%s temp=%.2f api_key=%s", base_url, self.model, temperature, "set" if raw_key else "none")

    def _is_lmstudio_model_already_loaded(self, origin: str, headers: dict[str, str]) -> bool:
        """Check via LM Studio REST whether model is already loaded — avoids redundant POST."""
        import httpx
        # LM Studio exposes several endpoints depending on version; try in order
        candidates = [
            f"{origin}/api/v0/models",          # LM Studio 0.x
            f"{origin}/api/v1/models",          # LM Studio 1.x
            f"{origin}/v1/models",              # OpenAI-compatible (also works)
        ]
        for url in candidates:
            try:
                with httpx.Client(timeout=5.0) as c:
                    r = c.get(url, headers=headers)
                    if r.status_code != 200:
                        continue
                    data = r.json()
                    ids: set[str] = set()
                    if isinstance(data, dict):
                        if "data" in data and isinstance(data["data"], list):
                            for m in data["data"]:
                                mid = m.get("id") or m.get("name") or ""
                                if mid:
                                    ids.add(self._norm_id(str(mid)).lower())
                                    ids.add(str(mid).lower())
                        if "models" in data and isinstance(data["models"], list):
                            for m in data["models"]:
                                mid = m.get("name") or m.get("id") or ""
                                if mid:
                                    ids.add(self._norm_id(str(mid)).lower())
                                    ids.add(str(mid).lower())
                    # also handle bare list
                    if isinstance(data, list):
                        for m in data:
                            if isinstance(m, dict):
                                mid = m.get("id") or m.get("name") or ""
                                if mid:
                                    ids.add(self._norm_id(str(mid)).lower())
                    want = self._norm_id(self.model).lower()
                    if want and (want in ids or self.model.lower() in ids):
                        log.debug("LM Studio model %r already loaded (via %s)", self.model, url)
                        return True
            except Exception:
                continue
        return False

    def _ensure_lmstudio_model_loaded(self, timeout: float = 60.0, force: bool = False) -> bool:
        """If base_url looks like LM Studio, ensure the selected model is loaded.

        LM Studio exposes POST {origin}/api/v1/models/load (REST API) which is separate
        from the OpenAI-compatible /v1/chat/completions. The OpenAI endpoint does not
        auto-load evicted models (TTL/auto-evict). We load on error (404/model_not_loaded)
        so the prompt request doesn't fail. Proactive calls are throttled + skip if already loaded.

        Returns True if we attempted and succeeded (or already loaded), False if we
        skipped (not LM Studio) or failed — caller should still try the chat request.
        """
        try:
            import time as _time
            from urllib.parse import urlparse
            import httpx

            raw = str(self._base_url_raw).rstrip("/")
            parsed = urlparse(raw if "://" in raw else f"http://{raw}")
            origin = f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else raw
            load_url = f"{origin}/api/v1/models/load"
            if not self.model or self.model.strip().lower() in ("", "none"):
                return False
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if not self._is_placeholder and self._api_key_raw:
                headers["Authorization"] = f"Bearer {self._api_key_raw}"

            # Throttle: don't POST again if we succeeded recently for same model
            if not force:
                age = _time.monotonic() - self._lmstudio_last_ok
                if self._lmstudio_last_model == self.model and age < 120:
                    log.debug("LM Studio ensure-loaded throttled (%.0fs ago) for %r — skipping", age, self.model)
                    return True
                # Check if already loaded via GET before POST — avoids reload every turn
                if self._is_lmstudio_model_already_loaded(origin, headers):
                    self._lmstudio_last_ok = _time.monotonic()
                    self._lmstudio_last_model = self.model
                    return True

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
                    self._lmstudio_last_ok = _time.monotonic()
                    self._lmstudio_last_model = self.model
                    return True
                elif r.status_code == 404:
                    log.debug("LM Studio load endpoint not found (404) — not LM Studio, skipping")
                    return False
                else:
                    txt = r.text[:400] if hasattr(r, "text") else ""
                    log.warning("LM Studio load returned %s: %s — will still try chat", r.status_code, txt[:200])
                    ok = r.status_code in (200, 400)
                    if ok:
                        self._lmstudio_last_ok = _time.monotonic()
                        self._lmstudio_last_model = self.model
                    return ok
        except Exception as e:
            log.debug("LM Studio ensure-loaded skipped/failed: %s", e)
        return False

    def _stream_once(self, messages: list[dict], tools: list[dict] | None, tool_calls: dict[int, dict]) -> Iterator[str]:
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        with self._client.chat.completions.create(**kwargs) as stream:  # type: ignore[arg-type]
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta  # type: ignore[attr-defined]
                content = getattr(delta, "content", None)
                if content:
                    log.debug("LLM delta: %r", content[:80])
                    yield content
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

    def _stream_fallback_httpx(self, messages: list[dict]) -> Iterator[str]:
        """Fallback streaming via raw httpx without auth (for 401 placeholder case)."""
        import httpx
        import json

        url = str(self._client.base_url).rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
        }
        self._pending_tool_calls = []
        with httpx.Client(timeout=None) as c:
            with c.stream("POST", url, json=payload, headers={"Content-Type": "application/json"}) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                        delta = obj.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
                    except Exception:
                        continue

    def stream_reply(self, messages: list[dict], tools: list[dict] | None = None) -> Iterator[str]:
        # No proactive POST — only load on demand if chat returns 404/model_not_loaded.
        # This avoids reloading every turn when LM Studio already has the model warm.
        self._pending_tool_calls: list[dict] = []
        log.info("LLM stream start model=%s msgs=%d tools=%s", self.model, len(messages), bool(tools))
        tool_calls: dict[int, dict] = {}
        cur_tools = tools
        for attempt in range(3):
            try:
                yield from self._stream_once(messages, cur_tools, tool_calls)
                if tool_calls:
                    self._pending_tool_calls = list(tool_calls.values())
                    log.info("LLM tool calls collected: %s", self._pending_tool_calls)
                else:
                    self._pending_tool_calls = []
                log.info("LLM stream done")
                # Mark LM Studio model as warm — avoids redundant GET/POST next turn
                try:
                    import time as _t
                    self._lmstudio_last_ok = _t.monotonic()
                    self._lmstudio_last_model = self.model
                except Exception:
                    pass
                return
            except BaseException as e:
                msg = str(e).lower()
                msg_raw = str(e)
                is_model_err = any(s in msg for s in ("model not found", "model not loaded", "model_not_loaded", "no model loaded", "failed to load model", "404"))
                is_auth_err = "401" in msg_raw or "authentication" in msg or "invalid token" in msg
                is_tool_err = cur_tools and ("tools" in msg or "tool" in msg)

                if is_model_err and attempt == 0:
                    log.warning("LLM stream model-not-loaded, trying ensure-loaded + retry: %s", e)
                    try:
                        if self._ensure_lmstudio_model_loaded(timeout=90.0, force=True):
                            continue
                    except Exception as re:
                        log.debug("Retry after load also failed: %s", re)

                if is_auth_err and self._is_placeholder and attempt == 0:
                    log.warning("LLM stream 401 with placeholder key — retrying without auth via httpx")
                    try:
                        # stream without tools via fallback (tools not supported in fallback)
                        yield from self._stream_fallback_httpx(messages)
                        log.info("LLM stream fallback done")
                        return
                    except Exception as fe:
                        log.debug("Fallback stream also failed: %s", fe)

                if is_tool_err and cur_tools is not None:
                    log.warning("LLM tools not supported, retrying without tools: %s", e)
                    self._pending_tool_calls = []
                    cur_tools = None
                    tool_calls.clear()
                    continue

                log.exception("LLM stream failed base_url=%s model=%s: %s", self._client.base_url, self.model, e)
                raise
        log.error("LLM stream exhausted retries")
        return

    @property
    def pending_tool_calls(self) -> list[dict]:
        return getattr(self, "_pending_tool_calls", [])

    def check(self) -> str:
        log.info("LLM check base_url=%s model=%s", self._client.base_url, self.model)
        try:
            r = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                max_tokens=8,
                temperature=0.0,
            )
            out = (r.choices[0].message.content or "").strip()
            log.info("LLM check ok: %r", out)
            try:
                import time as _t2
                self._lmstudio_last_ok = _t2.monotonic()
                self._lmstudio_last_model = self.model
            except Exception:
                pass
            return out
        except BaseException as e:
            msg = str(e).lower()
            # If model was cold, try one load+retry before falling back to auth handling
            if any(s in msg for s in ("model not found", "model not loaded", "no model loaded", "failed to load model", "404")):
                log.warning("LLM check failed with model-not-loaded, trying ensure-loaded + retry: %s", e)
                try:
                    if self._ensure_lmstudio_model_loaded(timeout=90.0, force=True):
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
