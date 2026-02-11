from __future__ import annotations

import csv
from pathlib import Path

from lucid_dream.io_schema import AgentResult, DreamInput, ShotAnalysis
from lucid_dream.registry import AnalyzerRegistry, AnalyzerSpec
from lucid_dream.report import (
    build_report_markdown,
    export_matrix_expanded_csv,
    export_matrix_long_csv,
)



def _spec(tag: str, length: int) -> AnalyzerSpec:
    return AnalyzerSpec(
        id=f"{tag.lower()}_id",
        tag=tag,
        name=tag,
        prompt_file="unused.txt",
        vector_length=length,
        dimension_labels=[f"{tag}_d{i}" for i in range(1, length + 1)],
    )



def _result(tag: str, vector: list[int]) -> AgentResult:
    return AgentResult(
        agent=tag,
        vector=vector,
        raw_text="",
        parse_ok=True,
        error=None,
        latency_ms=1,
    )



def test_expanded_csv_columns_and_rows(tmp_path: Path) -> None:
    specs = [_spec("A", 2), _spec("B", 3)]
    registry = AnalyzerRegistry(analyzers=specs)

    analyses = [
        ShotAnalysis(
            shot_id=1,
            shot_text="s1",
            results={"A": _result("A", [1, 0]), "B": _result("B", [2, 2, 1])},
            summary_after_shot="",
        ),
        ShotAnalysis(
            shot_id=2,
            shot_text="s2",
            results={"A": _result("A", [0, -1]), "B": _result("B", [1, 0, -2])},
            summary_after_shot="",
        ),
    ]

    out = tmp_path / "matrix_expanded.csv"
    export_matrix_expanded_csv(analyses=analyses, registry=registry, out_path=out)

    with out.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert rows[0] == ["shot_id", "A_1", "A_2", "B_1", "B_2", "B_3"]
    assert len(rows) == 3
    assert rows[1] == ["1", "1", "0", "2", "2", "1"]
    assert rows[2] == ["2", "0", "-1", "1", "0", "-2"]



def test_long_csv_row_count_matches_vector_dimensions(tmp_path: Path) -> None:
    specs = [_spec("A", 2), _spec("B", 3)]
    registry = AnalyzerRegistry(analyzers=specs)

    analyses = [
        ShotAnalysis(
            shot_id=1,
            shot_text="s1",
            results={"A": _result("A", [1, 0]), "B": _result("B", [2, 2, 1])},
            summary_after_shot="",
        ),
        ShotAnalysis(
            shot_id=2,
            shot_text="s2",
            results={"A": _result("A", [0, -1]), "B": _result("B", [1, 0, -2])},
            summary_after_shot="",
        ),
    ]

    out = tmp_path / "matrix_long.csv"
    export_matrix_long_csv(analyses=analyses, registry=registry, out_path=out)

    with out.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert rows[0] == ["shot_id", "agent_tag", "agent_id", "dim_index", "dim_label", "value"]
    expected_data_rows = len(analyses) * sum(spec.vector_length for spec in specs)
    assert len(rows) - 1 == expected_data_rows



def test_report_compatibility_section_emitted_only_when_canonical_present() -> None:
    canonical_specs = [_spec(tag, 1) for tag in ["NVS", "PVS", "CS", "SSP", "ARS", "SMS"]]
    registry_canonical = AnalyzerRegistry(analyzers=canonical_specs)

    result_map = {tag: _result(tag, [0]) for tag in ["NVS", "PVS", "CS", "SSP", "ARS", "SMS"]}
    analyses = [
        ShotAnalysis(
            shot_id=1,
            shot_text="shot",
            results=result_map,
            summary_after_shot="sum",
        )
    ]
    dream = DreamInput(trait_layer=["t"], background_layer="b", shots=["shot"], dream_id="d1")

    report1 = build_report_markdown(
        dream=dream,
        analyses=analyses,
        final_summary="sum",
        registry=registry_canonical,
        include_compatibility_section=True,
    )
    assert "Canonical Matrix (NVS/PVS/CS/SSP/ARS/SMS)" in report1

    generic_specs = [_spec("A", 1), _spec("B", 1)]
    registry_generic = AnalyzerRegistry(analyzers=generic_specs)
    analyses_generic = [
        ShotAnalysis(
            shot_id=1,
            shot_text="shot",
            results={"A": _result("A", [0]), "B": _result("B", [1])},
            summary_after_shot="sum",
        )
    ]

    report2 = build_report_markdown(
        dream=dream,
        analyses=analyses_generic,
        final_summary="sum",
        registry=registry_generic,
        include_compatibility_section=True,
    )
    assert "Canonical Matrix (NVS/PVS/CS/SSP/ARS/SMS)" not in report2



def test_updatable_registry_additional_analyzer_reflected_in_exports(tmp_path: Path) -> None:
    specs = [_spec("NVS", 2), _spec("NEW", 3)]
    registry = AnalyzerRegistry(analyzers=specs)

    analyses = [
        ShotAnalysis(
            shot_id=1,
            shot_text="s1",
            results={
                "NVS": _result("NVS", [1, 0]),
                "NEW": _result("NEW", [2, 1, -1]),
            },
            summary_after_shot="",
        )
    ]

    expanded_path = tmp_path / "matrix_expanded.csv"
    export_matrix_expanded_csv(analyses=analyses, registry=registry, out_path=expanded_path)

    with expanded_path.open("r", encoding="utf-8") as f:
        header = next(csv.reader(f))

    assert header == ["shot_id", "NVS_1", "NVS_2", "NEW_1", "NEW_2", "NEW_3"]
