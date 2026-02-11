"""Analyzer execution and vector parsing."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
import re
import time
from typing import Any

from .config import LLMConfig, build_thinking_extra_body
from .io_schema import AgentResult, ShotContext
from .logging_utils import CallLogger
from .prompt_loader import build_analyzer_prompt
from .registry import AnalyzerRegistry, AnalyzerSpec

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - only triggered when dependency is missing.
    OpenAI = None


@dataclass
class ModelCallResult:
    """Normalized model call payload."""

    text: str
    reasoning_content: str | None
    prompt_tokens: int | None
    completion_tokens: int | None



def parse_vector(raw_text: str, spec: AnalyzerSpec) -> tuple[list[int], bool, str | None]:
    """Parse one analyzer vector from potentially messy model output."""

    parse_ok = True
    errors: list[str] = []
    text = raw_text or ""

    tagged_pattern = re.compile(
        rf"\b{re.escape(spec.tag)}\s*:\s*\(([^)]*)\)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    tagged_match = tagged_pattern.search(text)

    vector_blob: str | None = None
    if tagged_match:
        vector_blob = tagged_match.group(1)
    else:
        fallback_matches = re.findall(r"\(([^)]*)\)", text, flags=re.DOTALL)
        for candidate in fallback_matches:
            if re.search(r"-?\d", candidate):
                vector_blob = candidate
                break
        if vector_blob is None:
            return [0] * spec.vector_length, False, "unable_to_parse_vector"
        parse_ok = False
        errors.append("tag_not_found_used_parenthesized_fallback")

    parsed_numbers = [int(token) for token in re.findall(r"-?\d+", vector_blob)]
    if not parsed_numbers:
        return [0] * spec.vector_length, False, "unable_to_parse_numeric_values"

    clamped: list[int] = []
    for val in parsed_numbers:
        corrected = min(max(val, spec.value_min), spec.value_max)
        if corrected != val:
            parse_ok = False
            errors.append(f"out_of_range_clamped:{val}->{corrected}")
        clamped.append(corrected)

    expected_len = spec.vector_length
    if len(clamped) < expected_len:
        parse_ok = False
        errors.append(
            f"length_too_short:{len(clamped)}_expected:{expected_len}_padded_with_0"
        )
        clamped.extend([0] * (expected_len - len(clamped)))
    elif len(clamped) > expected_len:
        parse_ok = False
        errors.append(
            f"length_too_long:{len(clamped)}_expected:{expected_len}_truncated"
        )
        clamped = clamped[:expected_len]

    error_text = "; ".join(sorted(set(errors))) if errors else None
    return clamped, parse_ok, error_text



def _is_transient_error(exc: Exception) -> bool:
    """Heuristic classification for retryable model errors."""

    retryable_names = {
        "APITimeoutError",
        "APIConnectionError",
        "RateLimitError",
        "InternalServerError",
        "TimeoutError",
    }
    return exc.__class__.__name__ in retryable_names


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

    # Some providers embed reasoning chunks inside content arrays.
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


class AnalyzerRunner:
    """Executes analyzer agents against shot contexts."""

    def __init__(self, llm_config: LLMConfig, logger: CallLogger, dream_id: str) -> None:
        if OpenAI is None:
            raise RuntimeError(
                "openai dependency is required. Install with: pip install openai"
            )

        client_kwargs: dict[str, Any] = {}
        if llm_config.api_key:
            client_kwargs["api_key"] = llm_config.api_key
        if llm_config.base_url:
            client_kwargs["base_url"] = llm_config.base_url

        self._client = OpenAI(**client_kwargs)
        self._llm_config = llm_config
        self._logger = logger
        self._dream_id = dream_id

    def _call_model(
        self,
        *,
        prompt: str,
        model: str,
        temperature: float,
        max_output_tokens: int | None,
    ) -> ModelCallResult:
        request: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "timeout": self._llm_config.timeout_s,
        }
        if max_output_tokens is not None:
            request["max_tokens"] = max_output_tokens
        extra_body = build_thinking_extra_body(self._llm_config.analyzer_disable_thinking)
        if extra_body is not None:
            request["extra_body"] = extra_body

        response = self._client.chat.completions.create(**request)
        text = ""
        reasoning_content: str | None = None
        if response.choices:
            first = response.choices[0]
            if getattr(first, "message", None) is not None:
                text = _coerce_to_text(getattr(first.message, "content", None))
                reasoning_content = _extract_reasoning_content(first.message)

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)

        return ModelCallResult(
            text=text,
            reasoning_content=reasoning_content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def run_single_analyzer(
        self,
        *,
        spec: AnalyzerSpec,
        role_prompt: str,
        ctx: ShotContext,
    ) -> AgentResult:
        """Run one analyzer with retry/backoff and parse enforcement."""

        prompt_text = build_analyzer_prompt(spec, role_prompt, ctx)
        retries = 0
        started_at = time.perf_counter()

        last_error: str | None = None
        raw_text = ""
        reasoning_content: str | None = None
        prompt_tokens: int | None = None
        completion_tokens: int | None = None

        for attempt in range(self._llm_config.max_retries + 1):
            try:
                model = spec.model or self._llm_config.analyzer_model
                temperature = (
                    spec.temperature
                    if spec.temperature is not None
                    else self._llm_config.analyzer_temperature
                )
                call_result = self._call_model(
                    prompt=prompt_text,
                    model=model,
                    temperature=temperature,
                    max_output_tokens=spec.max_output_tokens,
                )
                raw_text = call_result.text
                reasoning_content = call_result.reasoning_content
                prompt_tokens = call_result.prompt_tokens
                completion_tokens = call_result.completion_tokens
                break
            except Exception as exc:
                last_error = f"model_call_failed:{exc.__class__.__name__}:{exc}"
                should_retry = (
                    attempt < self._llm_config.max_retries and _is_transient_error(exc)
                )
                if not should_retry:
                    break
                retries += 1
                delay = self._llm_config.backoff_base_s * (
                    self._llm_config.backoff_factor**attempt
                )
                time.sleep(delay)

        latency_ms = int((time.perf_counter() - started_at) * 1000)

        if raw_text:
            vector, parse_ok, parse_error = parse_vector(raw_text, spec)
            error = parse_error
        else:
            vector = [0] * spec.vector_length
            parse_ok = False
            error = last_error or "empty_model_output"

        result = AgentResult(
            agent=spec.tag,
            vector=vector,
            raw_text=raw_text,
            parse_ok=parse_ok,
            error=error,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        self._logger.log_call(
            {
                "dream_id": self._dream_id,
                "shot_id": ctx.shot_id,
                "agent_id": spec.id,
                "agent_tag": spec.tag,
                "agent_version": spec.version,
                "vector_length_expected": spec.vector_length,
                "model": spec.model or self._llm_config.analyzer_model,
                "retry_count": retries,
                "latency_ms": latency_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": (
                    (prompt_tokens or 0) + (completion_tokens or 0)
                    if prompt_tokens is not None or completion_tokens is not None
                    else None
                ),
                "prompt_text": prompt_text,
                "raw_output_text": raw_text,
                "reasoning_content": reasoning_content,
                "parsed_vector": vector,
                "parse_ok": parse_ok,
                "error": error,
            }
        )

        return result

    async def run_all_analyzers_for_shot(
        self,
        *,
        registry: AnalyzerRegistry,
        prompts: dict[str, str],
        ctx: ShotContext,
    ) -> dict[str, AgentResult]:
        """Run all enabled analyzers concurrently using a thread pool."""

        specs = registry.enabled_analyzers()
        if not specs:
            return {}

        loop = asyncio.get_running_loop()
        max_workers = max(1, len(specs))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = [
                loop.run_in_executor(
                    executor,
                    partial(
                        self.run_single_analyzer,
                        spec=spec,
                        role_prompt=prompts[spec.tag],
                        ctx=ctx,
                    ),
                )
                for spec in specs
            ]
            outputs = await asyncio.gather(*tasks)
        return {result.agent: result for result in outputs}

    async def run_all_analyzers_for_contexts(
        self,
        *,
        registry: AnalyzerRegistry,
        prompts: dict[str, str],
        contexts: list[ShotContext],
    ) -> dict[int, dict[str, AgentResult]]:
        """Run analyzers in parallel across every (shot, analyzer) pair."""

        specs = registry.enabled_analyzers()
        if not contexts:
            return {}
        if not specs:
            return {ctx.shot_id: {} for ctx in contexts}

        loop = asyncio.get_running_loop()
        max_workers = max(1, len(specs) * len(contexts))
        tasks: list[tuple[int, asyncio.Future[AgentResult]]] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for ctx in contexts:
                for spec in specs:
                    future = loop.run_in_executor(
                        executor,
                        partial(
                            self.run_single_analyzer,
                            spec=spec,
                            role_prompt=prompts[spec.tag],
                            ctx=ctx,
                        ),
                    )
                    tasks.append((ctx.shot_id, future))

            ordered_futures = [future for _, future in tasks]
            ordered_results = await asyncio.gather(*ordered_futures)

        by_shot: dict[int, dict[str, AgentResult]] = {ctx.shot_id: {} for ctx in contexts}
        for (shot_id, _), result in zip(tasks, ordered_results):
            by_shot[shot_id][result.agent] = result

        return by_shot
