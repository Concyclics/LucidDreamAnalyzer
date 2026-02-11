"""Rolling shot summarizer."""

from __future__ import annotations

import time
from typing import Any

from .config import LLMConfig, build_thinking_extra_body
from .logging_utils import CallLogger
from .prompt_loader import build_summarizer_prompt

try:
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover - only triggered when dependency is missing.
    AsyncOpenAI = None


class ShotSummarizer:
    """Maintains factual rolling summary across ordered shots."""

    def __init__(
        self,
        *,
        llm_config: LLMConfig,
        logger: CallLogger,
        dream_id: str,
        max_chars: int = 1200,
    ) -> None:
        if AsyncOpenAI is None:
            raise RuntimeError(
                "openai dependency is required. Install with: pip install openai"
            )

        client_kwargs: dict[str, Any] = {}
        if llm_config.api_key:
            client_kwargs["api_key"] = llm_config.api_key
        if llm_config.base_url:
            client_kwargs["base_url"] = llm_config.base_url

        self._client = AsyncOpenAI(**client_kwargs)
        self._llm_config = llm_config
        self._logger = logger
        self._dream_id = dream_id
        self._max_chars = max_chars

    async def update(
        self,
        *,
        shot_id: int,
        trait_layer: list[str],
        background_layer: str,
        prev_summary: str,
        current_shot: str,
    ) -> str:
        """Update rolling summary after one processed shot."""

        prompt_text = build_summarizer_prompt(
            trait_layer=trait_layer,
            background_layer=background_layer,
            prev_summary=prev_summary,
            current_shot=current_shot,
            max_chars=self._max_chars,
        )

        started_at = time.perf_counter()
        raw_output = ""
        reasoning_content: str | None = None
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        error: str | None = None

        try:
            request: dict[str, Any] = {
                "model": self._llm_config.summarizer_model,
                "messages": [{"role": "user", "content": prompt_text}],
                "temperature": self._llm_config.summarizer_temperature,
                "timeout": self._llm_config.timeout_s,
            }
            extra_body = build_thinking_extra_body(
                self._llm_config.summarizer_disable_thinking
            )
            if extra_body is not None:
                request["extra_body"] = extra_body

            response = await self._client.chat.completions.create(
                **request,
            )
            if response.choices and response.choices[0].message is not None:
                message = response.choices[0].message
                raw_output = _coerce_to_text(getattr(message, "content", None))
                reasoning_content = _extract_reasoning_content(message)

            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
        except Exception as exc:  # pragma: no cover - network path in tests.
            error = f"summarizer_failed:{exc.__class__.__name__}:{exc}"

        latency_ms = int((time.perf_counter() - started_at) * 1000)

        if error:
            summary = self._deterministic_fallback(prev_summary, current_shot)
        else:
            summary = self._normalize_summary(raw_output)
            if not summary:
                summary = self._deterministic_fallback(prev_summary, current_shot)
                error = "summarizer_empty_output_fallback_used"

        summary = self._truncate(summary, self._max_chars)

        self._logger.log_call(
            {
                "dream_id": self._dream_id,
                "shot_id": shot_id,
                "agent_id": "rolling_summarizer",
                "agent_tag": "SUM",
                "agent_version": "v1",
                "vector_length_expected": None,
                "model": self._llm_config.summarizer_model,
                "retry_count": 0,
                "latency_ms": latency_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": (
                    (prompt_tokens or 0) + (completion_tokens or 0)
                    if prompt_tokens is not None or completion_tokens is not None
                    else None
                ),
                "prompt_text": prompt_text,
                "raw_output_text": raw_output,
                "reasoning_content": reasoning_content,
                "parsed_vector": None,
                "parse_ok": error is None,
                "error": error,
            }
        )

        return summary

    @staticmethod
    def _normalize_summary(text: str) -> str:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        cleaned: list[str] = []
        for line in lines:
            line = line.lstrip("- ")
            if not line:
                continue
            cleaned.append(f"- {line}")
        return "\n".join(cleaned)

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _deterministic_fallback(prev_summary: str, current_shot: str) -> str:
        shot_line = current_shot.strip().replace("\n", " ")
        if len(shot_line) > 260:
            shot_line = shot_line[:257].rstrip() + "..."
        addition = f"- Shot event: {shot_line}"
        if not prev_summary.strip():
            return addition
        return f"{prev_summary.rstrip()}\n{addition}"


def _coerce_to_text(value: Any) -> str:
    """Convert provider response fragments into plain text."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                if "text" in item and isinstance(item["text"], str):
                    parts.append(item["text"])
                elif "content" in item and isinstance(item["content"], str):
                    parts.append(item["content"])
                continue

            text_attr = getattr(item, "text", None)
            if isinstance(text_attr, str):
                parts.append(text_attr)
                continue
            content_attr = getattr(item, "content", None)
            if isinstance(content_attr, str):
                parts.append(content_attr)
        return "\n".join(part for part in parts if part)
    return str(value)


def _extract_reasoning_content(message: Any) -> str | None:
    """Extract reasoning content from vendor-specific message fields."""

    direct_reasoning = getattr(message, "reasoning_content", None)
    if direct_reasoning:
        text = _coerce_to_text(direct_reasoning).strip()
        if text:
            return text

    reasoning_attr = getattr(message, "reasoning", None)
    if reasoning_attr:
        text = _coerce_to_text(reasoning_attr).strip()
        if text:
            return text

    content = getattr(message, "content", None)
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                item_type = str(item.get("type", "")).lower()
                if item_type in {"reasoning", "reasoning_content"}:
                    chunk = _coerce_to_text(item.get("text") or item.get("content")).strip()
                    if chunk:
                        chunks.append(chunk)
                    continue
            item_type = str(getattr(item, "type", "")).lower()
            if item_type in {"reasoning", "reasoning_content"}:
                chunk = _coerce_to_text(
                    getattr(item, "text", None) or getattr(item, "content", None)
                ).strip()
                if chunk:
                    chunks.append(chunk)
        if chunks:
            return "\n".join(chunks)

    return None
