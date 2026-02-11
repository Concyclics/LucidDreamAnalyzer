"""Prompt loading and rendering utilities."""

from __future__ import annotations

import json
from pathlib import Path

from .io_schema import ShotContext
from .registry import AnalyzerRegistry, AnalyzerSpec



def load_analyzer_prompts(
    registry: AnalyzerRegistry,
    prompt_dir: Path,
) -> dict[str, str]:
    """Load prompt text for each analyzer tag from files."""

    prompts: dict[str, str] = {}
    for spec in registry.enabled_analyzers():
        prompt_path = prompt_dir / spec.prompt_file
        prompts[spec.tag] = prompt_path.read_text(encoding="utf-8")
    return prompts



def build_analyzer_prompt(spec: AnalyzerSpec, role_prompt: str, ctx: ShotContext) -> str:
    """Build a standardized analyzer prompt with injected JSON context."""

    context_json = json.dumps(
        {
            "trait_layer": ctx.trait_layer,
            "background_layer": ctx.background_layer,
            "prev_shots_summary": ctx.prev_shots_summary,
            "shot": ctx.current_shot,
        },
        ensure_ascii=False,
        indent=2,
    )

    return (
        f"{role_prompt.strip()}\n\n"
        "You are evaluating exactly one shot with additional context.\n"
        "Use the JSON context below as the only analysis input.\n\n"
        "JSON context:\n"
        f"{context_json}\n\n"
        "Output rules:\n"
        f"1) Output exactly one line in this format: {spec.tag}:(v1, v2, ..., v{spec.vector_length})\n"
        "2) Do not output any extra prose or explanation.\n"
        "3) Use integers only.\n"
    )



def build_summarizer_prompt(
    *,
    trait_layer: list[str],
    background_layer: str,
    prev_summary: str,
    current_shot: str,
    max_chars: int,
) -> str:
    """Build the rolling-summary prompt."""

    context_json = json.dumps(
        {
            "trait_layer": trait_layer,
            "background_layer": background_layer,
            "prev_shots_summary": prev_summary,
            "current_shot": current_shot,
        },
        ensure_ascii=False,
        indent=2,
    )

    return (
        "Summarize dream progression as concise factual bullet points.\n"
        "Keep only stable facts; do not add interpretation, diagnosis, or speculation.\n"
        "Capture: key events, emotions, threats/rewards, social interactions, "
        "bodily sensations, awakenings.\n"
        f"Hard limit: {max_chars} characters.\n\n"
        "JSON context:\n"
        f"{context_json}\n\n"
        "Output only bullets."
    )
