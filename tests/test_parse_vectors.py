from __future__ import annotations

import json
from pathlib import Path

import pytest

from lucid_dream.analyzers import parse_vector
from lucid_dream.registry import AnalyzerSpec, load_registry



def _spec(tag: str = "NVS", length: int = 5) -> AnalyzerSpec:
    return AnalyzerSpec(
        id=f"{tag.lower()}_id",
        tag=tag,
        name=tag,
        prompt_file="dummy.txt",
        vector_length=length,
        dimension_labels=[f"d{i}" for i in range(length)],
    )



def test_parse_vector_exact_tag_format() -> None:
    spec = _spec("NVS", 5)
    vector, parse_ok, error = parse_vector("NVS:(1, 0, -1, 2, -2)", spec)
    assert vector == [1, 0, -1, 2, -2]
    assert parse_ok is True
    assert error is None



def test_parse_vector_with_extra_text_still_parses() -> None:
    spec = _spec("PVS", 3)
    raw = "analysis note\nPVS:(1,2,0)\nthanks"
    vector, parse_ok, error = parse_vector(raw, spec)
    assert vector == [1, 2, 0]
    assert parse_ok is True
    assert error is None



def test_parse_vector_fallback_when_tag_missing_marks_failure() -> None:
    spec = _spec("CS", 4)
    vector, parse_ok, error = parse_vector("WRONG:(1,2,0,-1)", spec)
    assert vector == [1, 2, 0, -1]
    assert parse_ok is False
    assert "fallback" in (error or "")



def test_parse_vector_clamps_out_of_range_values() -> None:
    spec = _spec("ARS", 3)
    vector, parse_ok, error = parse_vector("ARS:(9,-9,2)", spec)
    assert vector == [2, -2, 2]
    assert parse_ok is False
    assert "out_of_range_clamped" in (error or "")



def test_parse_vector_short_length_padded() -> None:
    spec = _spec("SMS", 5)
    vector, parse_ok, error = parse_vector("SMS:(1,2)", spec)
    assert vector == [1, 2, 0, 0, 0]
    assert parse_ok is False
    assert "length_too_short" in (error or "")



def test_parse_vector_long_length_truncated() -> None:
    spec = _spec("NVS", 2)
    vector, parse_ok, error = parse_vector("NVS:(1,2,0,-1)", spec)
    assert vector == [1, 2]
    assert parse_ok is False
    assert "length_too_long" in (error or "")



def test_parse_vector_unparseable_returns_zeros() -> None:
    spec = _spec("NVS", 3)
    vector, parse_ok, error = parse_vector("no vector here", spec)
    assert vector == [0, 0, 0]
    assert parse_ok is False
    assert error == "unable_to_parse_vector"



def test_registry_validation_rejects_duplicate_tags(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("prompt", encoding="utf-8")
    registry_path = tmp_path / "analyzers.json"
    payload = {
        "analyzers": [
            {
                "id": "a1",
                "tag": "A",
                "name": "A",
                "prompt_file": "a.txt",
                "vector_length": 1,
                "dimension_labels": ["x"],
                "enabled": True,
            },
            {
                "id": "a2",
                "tag": "A",
                "name": "A2",
                "prompt_file": "a.txt",
                "vector_length": 1,
                "dimension_labels": ["x"],
                "enabled": True,
            },
        ]
    }
    registry_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate analyzer tag"):
        load_registry(registry_path, prompt_dir=tmp_path, strict=True)



def test_registry_validation_rejects_missing_prompt(tmp_path: Path) -> None:
    registry_path = tmp_path / "analyzers.json"
    payload = {
        "analyzers": [
            {
                "id": "a1",
                "tag": "A",
                "name": "A",
                "prompt_file": "missing.txt",
                "vector_length": 1,
                "dimension_labels": ["x"],
                "enabled": True,
            }
        ]
    }
    registry_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="prompt file not found"):
        load_registry(registry_path, prompt_dir=tmp_path, strict=True)



def test_registry_validation_rejects_dimension_length_mismatch(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("prompt", encoding="utf-8")
    registry_path = tmp_path / "analyzers.json"
    payload = {
        "analyzers": [
            {
                "id": "a1",
                "tag": "A",
                "name": "A",
                "prompt_file": "a.txt",
                "vector_length": 2,
                "dimension_labels": ["x"],
                "enabled": True,
            }
        ]
    }
    registry_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="dimension_labels length"):
        load_registry(registry_path, prompt_dir=tmp_path, strict=True)
