"""Analyzer registry models and loader."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AnalyzerSpec:
    """Configuration for one analyzer agent."""

    id: str
    tag: str
    name: str
    prompt_file: str
    vector_length: int
    dimension_labels: list[str]
    enabled: bool = True
    model: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    value_min: int = -2
    value_max: int = 2
    version: str | None = None

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "AnalyzerSpec":
        return AnalyzerSpec(
            id=str(payload["id"]),
            tag=str(payload["tag"]),
            name=str(payload.get("name", payload["tag"])),
            prompt_file=str(payload["prompt_file"]),
            vector_length=int(payload["vector_length"]),
            dimension_labels=[str(x) for x in payload.get("dimension_labels", [])],
            enabled=bool(payload.get("enabled", True)),
            model=payload.get("model"),
            temperature=(
                float(payload["temperature"])
                if payload.get("temperature") is not None
                else None
            ),
            max_output_tokens=(
                int(payload["max_output_tokens"])
                if payload.get("max_output_tokens") is not None
                else None
            ),
            value_min=int(payload.get("value_min", -2)),
            value_max=int(payload.get("value_max", 2)),
            version=payload.get("version"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tag": self.tag,
            "name": self.name,
            "prompt_file": self.prompt_file,
            "vector_length": self.vector_length,
            "dimension_labels": self.dimension_labels,
            "enabled": self.enabled,
            "model": self.model,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "value_min": self.value_min,
            "value_max": self.value_max,
            "version": self.version,
        }


@dataclass
class AnalyzerRegistry:
    """Ordered analyzer registry used at runtime."""

    analyzers: list[AnalyzerSpec] = field(default_factory=list)

    def enabled_analyzers(self) -> list[AnalyzerSpec]:
        return [spec for spec in self.analyzers if spec.enabled]

    def by_tag(self, tag: str) -> AnalyzerSpec:
        for spec in self.analyzers:
            if spec.tag == tag:
                return spec
        raise KeyError(f"Unknown analyzer tag: {tag}")

    def by_id(self, analyzer_id: str) -> AnalyzerSpec:
        for spec in self.analyzers:
            if spec.id == analyzer_id:
                return spec
        raise KeyError(f"Unknown analyzer id: {analyzer_id}")

    def validate(self, prompt_dir: Path, strict: bool = True) -> list[str]:
        errors: list[str] = []
        seen_ids: set[str] = set()
        seen_tags: set[str] = set()

        for spec in self.analyzers:
            if spec.id in seen_ids:
                errors.append(f"Duplicate analyzer id: {spec.id}")
            seen_ids.add(spec.id)

            if spec.tag in seen_tags:
                errors.append(f"Duplicate analyzer tag: {spec.tag}")
            seen_tags.add(spec.tag)

            if spec.vector_length <= 0:
                errors.append(f"{spec.tag}: vector_length must be > 0")

            if len(spec.dimension_labels) != spec.vector_length:
                errors.append(
                    f"{spec.tag}: dimension_labels length ({len(spec.dimension_labels)}) "
                    f"must equal vector_length ({spec.vector_length})"
                )

            if spec.value_min > spec.value_max:
                errors.append(
                    f"{spec.tag}: value_min ({spec.value_min}) cannot exceed "
                    f"value_max ({spec.value_max})"
                )

            prompt_path = prompt_dir / spec.prompt_file
            if not prompt_path.exists():
                errors.append(f"{spec.tag}: prompt file not found: {prompt_path}")

        if strict and not self.enabled_analyzers():
            errors.append("Registry has no enabled analyzers")

        return errors

    def to_dict(self) -> dict[str, Any]:
        return {"analyzers": [spec.to_dict() for spec in self.analyzers]}



def load_registry(path: Path, prompt_dir: Path, strict: bool = True) -> AnalyzerRegistry:
    """Load and validate analyzer registry from JSON file."""

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    analyzers_payload = payload.get("analyzers")
    if not isinstance(analyzers_payload, list):
        raise ValueError("Registry JSON must contain an 'analyzers' list")

    analyzers = [AnalyzerSpec.from_dict(item) for item in analyzers_payload]
    registry = AnalyzerRegistry(analyzers=analyzers)

    errors = registry.validate(prompt_dir=prompt_dir, strict=strict)
    if errors:
        raise ValueError("Invalid analyzer registry:\n- " + "\n- ".join(errors))

    return registry
