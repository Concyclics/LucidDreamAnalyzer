"""Markdown and CSV report generation."""

from __future__ import annotations

import csv
from pathlib import Path

from .config import DEFAULT_CANONICAL_TAGS
from .io_schema import DreamInput, ShotAnalysis
from .registry import AnalyzerRegistry



def build_report_markdown(
    *,
    dream: DreamInput,
    analyses: list[ShotAnalysis],
    final_summary: str,
    registry: AnalyzerRegistry,
    include_compatibility_section: bool = True,
) -> str:
    """Build report markdown with per-shot vectors and matrix views."""

    enabled_specs = registry.enabled_analyzers()

    lines: list[str] = []
    lines.append("# Lucid Dream Multi-Agent Analysis Report")
    lines.append("")

    lines.append("## Trait Layer")
    for trait in dream.trait_layer:
        lines.append(f"- {trait}")
    lines.append("")

    lines.append("## Background Layer")
    lines.append(dream.background_layer)
    lines.append("")

    lines.append("## Final Rolling Summary")
    lines.append(final_summary or "(empty)")
    lines.append("")

    lines.append("## Per-Shot Analysis")
    lines.append("")
    for shot in analyses:
        if shot.shot_id == 0:
            lines.append("### Background Layer (Pre-Shot)")
        else:
            lines.append(f"### Shot {shot.shot_id}")
        lines.append(shot.shot_text)
        lines.append("")
        for spec in enabled_specs:
            result = shot.results.get(spec.tag)
            vector = result.vector if result else []
            parse_note = "" if (result and result.parse_ok) else " (parse corrected/failure)"
            lines.append(f"- {spec.tag}: {_vector_to_text(vector)}{parse_note}")
        lines.append("")

    if include_compatibility_section and _has_all_tags(registry, DEFAULT_CANONICAL_TAGS):
        lines.append("## Canonical Matrix (NVS/PVS/CS/SSP/ARS/SMS)")
        lines.append("")
        canonical_tags = list(DEFAULT_CANONICAL_TAGS)
        lines.extend(_matrix_table_lines(analyses=analyses, tags=canonical_tags))
        lines.append("")

    lines.append("## Dynamic Matrix (Registry-Driven)")
    lines.append("")
    lines.extend(_matrix_table_lines(analyses=analyses, tags=[spec.tag for spec in enabled_specs]))
    lines.append("")

    return "\n".join(lines).strip() + "\n"



def export_matrix_expanded_csv(
    *,
    analyses: list[ShotAnalysis],
    registry: AnalyzerRegistry,
    out_path: Path,
) -> None:
    """Export dynamic expanded-wide matrix CSV."""

    enabled_specs = registry.enabled_analyzers()
    header: list[str] = ["shot_id"]
    for spec in enabled_specs:
        for idx in range(1, spec.vector_length + 1):
            header.append(f"{spec.tag}_{idx}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for shot in analyses:
            row: list[int | str] = [shot.shot_id]
            for spec in enabled_specs:
                vector = shot.results[spec.tag].vector
                row.extend(vector)
            writer.writerow(row)



def export_matrix_long_csv(
    *,
    analyses: list[ShotAnalysis],
    registry: AnalyzerRegistry,
    out_path: Path,
) -> None:
    """Export normalized long-format matrix CSV."""

    enabled_specs = registry.enabled_analyzers()
    header = ["shot_id", "agent_tag", "agent_id", "dim_index", "dim_label", "value"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for shot in analyses:
            for spec in enabled_specs:
                vector = shot.results[spec.tag].vector
                for idx, value in enumerate(vector, start=1):
                    label = spec.dimension_labels[idx - 1]
                    writer.writerow([shot.shot_id, spec.tag, spec.id, idx, label, value])



def _has_all_tags(registry: AnalyzerRegistry, tags: tuple[str, ...]) -> bool:
    present = {spec.tag for spec in registry.enabled_analyzers()}
    return all(tag in present for tag in tags)



def _matrix_table_lines(*, analyses: list[ShotAnalysis], tags: list[str]) -> list[str]:
    lines = [
        "| shot_id | " + " | ".join(tags) + " |",
        "|" + "---|" * (len(tags) + 1),
    ]

    for shot in analyses:
        values = [_vector_to_text(shot.results[tag].vector) for tag in tags]
        lines.append(f"| {shot.shot_id} | " + " | ".join(values) + " |")
    return lines



def _vector_to_text(vector: list[int]) -> str:
    return "[" + ", ".join(str(x) for x in vector) + "]"
