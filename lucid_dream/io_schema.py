"""Input/output schema models and parsing helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from .registry import AnalyzerRegistry


@dataclass
class DreamInput:
    """A full dream report input."""

    trait_layer: list[str]
    background_layer: str
    shots: list[str]
    dream_id: str | None = None


@dataclass
class ShotContext:
    """Context passed into each analyzer for a shot."""

    shot_id: int
    trait_layer: list[str]
    background_layer: str
    prev_shots_summary: str
    current_shot: str


@dataclass
class AgentResult:
    """Analyzer result for one shot."""

    agent: str
    vector: list[int]
    raw_text: str
    parse_ok: bool
    error: str | None
    latency_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass
class ShotAnalysis:
    """All analyzer outputs for one shot."""

    shot_id: int
    shot_text: str
    results: dict[str, AgentResult]
    summary_after_shot: str


@dataclass
class RunArtifacts:
    """Paths for generated run outputs."""

    run_dir: Path
    report_md: Path
    matrix_expanded_csv: Path | None
    matrix_long_csv: Path | None
    calls_jsonl: Path



def validate_dream_input(dream: DreamInput) -> None:
    """Validate required dream input fields."""

    if not isinstance(dream.trait_layer, list):
        raise ValueError("trait_layer must be a list of strings")

    if not all(isinstance(item, str) for item in dream.trait_layer):
        raise ValueError("trait_layer must only contain strings")

    if not isinstance(dream.background_layer, str) or not dream.background_layer.strip():
        raise ValueError("background_layer must be a non-empty string")

    if not isinstance(dream.shots, list) or not dream.shots:
        raise ValueError("shots must be a non-empty list of strings")

    if not all(isinstance(item, str) and item.strip() for item in dream.shots):
        raise ValueError("shots must only contain non-empty strings")



def validate_shot_vectors(analysis: ShotAnalysis, registry: AnalyzerRegistry) -> None:
    """Validate shot result vectors against registry vector constraints."""

    for spec in registry.enabled_analyzers():
        result = analysis.results.get(spec.tag)
        if result is None:
            raise ValueError(f"Missing analyzer result for {spec.tag}")
        if len(result.vector) != spec.vector_length:
            raise ValueError(
                f"{spec.tag} vector length mismatch: expected {spec.vector_length}, "
                f"got {len(result.vector)}"
            )
        for val in result.vector:
            if val < spec.value_min or val > spec.value_max:
                raise ValueError(
                    f"{spec.tag} value out of bounds [{spec.value_min}, {spec.value_max}]: {val}"
                )



def load_dream_input(path: Path) -> DreamInput:
    """Load dream input from JSON or template-like plain text."""

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return parse_dream_json(text)
    return parse_dream_template_text(text)



def parse_dream_json(text: str) -> DreamInput:
    """Parse dream input from JSON payload."""

    payload = json.loads(text)
    dream = DreamInput(
        dream_id=payload.get("dream_id"),
        trait_layer=list(payload.get("trait_layer", [])),
        background_layer=str(payload.get("background_layer", "")).strip(),
        shots=list(payload.get("shots", [])),
    )
    validate_dream_input(dream)
    return dream



def parse_dream_template_text(text: str) -> DreamInput:
    """Parse template text into DreamInput.

    The parser is intentionally permissive for headings like:
    - ## Trait Layer
    - ## Background Layer
    - ## Shot1 / ## Shot 1 / shot 1
    """

    lines = text.splitlines()

    trait_layer: list[str] = []
    background_lines: list[str] = []
    shot_map: dict[int, list[str]] = {}

    section: str | None = None
    current_shot_id: int | None = None

    heading_re = re.compile(r"^\s*#{0,3}\s*([A-Za-z ]+)\s*$")
    shot_re = re.compile(r"^\s*#{0,3}\s*shot\s*(\d+)\s*$", re.IGNORECASE)

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        shot_match = shot_re.match(stripped)
        if shot_match:
            current_shot_id = int(shot_match.group(1))
            shot_map.setdefault(current_shot_id, [])
            section = "shot"
            continue

        heading_match = heading_re.match(stripped)
        if heading_match:
            heading = heading_match.group(1).strip().lower()
            if "trait layer" in heading:
                section = "trait"
                current_shot_id = None
                continue
            if "background layer" in heading:
                section = "background"
                current_shot_id = None
                continue
            if "sleep content" in heading:
                section = None
                current_shot_id = None
                continue
            if stripped.startswith("##"):
                section = None
                current_shot_id = None

        if section == "trait":
            if stripped:
                cleaned = re.sub(r"^[-*]\s*", "", stripped)
                trait_layer.append(cleaned)
            continue

        if section == "background":
            background_lines.append(line)
            continue

        if section == "shot" and current_shot_id is not None:
            shot_map[current_shot_id].append(line)

    shots: list[str] = []
    for shot_id in sorted(shot_map):
        body = "\n".join(shot_map[shot_id]).strip()
        if body:
            shots.append(body)

    dream = DreamInput(
        trait_layer=trait_layer,
        background_layer="\n".join(background_lines).strip(),
        shots=shots,
        dream_id=None,
    )
    validate_dream_input(dream)
    return dream
