from __future__ import annotations

from pathlib import Path
import time

import pytest

from lucid_dream.config import LLMConfig, RuntimeConfig
from lucid_dream.io_schema import DreamInput, RunArtifacts
from lucid_dream.web_jobs import (
    JobManager,
    compute_calls_per_second_and_eta,
    compute_expected_calls,
    read_completed_calls,
)



def _dream() -> DreamInput:
    return DreamInput(
        trait_layer=["t1"],
        background_layer="background",
        shots=["s1", "s2"],
        dream_id=None,
    )



def _runtime() -> RuntimeConfig:
    return RuntimeConfig(
        prompt_dir=Path("prompts"),
        registry_path=Path("prompts/analyzers.json"),
        canonical_tags=("NVS", "PVS", "CS", "SSP", "ARS", "SMS"),
        strict_registry=True,
        emit_long_csv=True,
        emit_expanded_csv=True,
        compatibility_layer=True,
        summarizer_max_chars=1200,
    )



def _llm() -> LLMConfig:
    return LLMConfig(
        api_key=None,
        base_url=None,
        analyzer_model="m",
        summarizer_model="m",
        analyzer_temperature=0.0,
        summarizer_temperature=0.0,
        timeout_s=10.0,
        max_retries=1,
        backoff_base_s=0.1,
        backoff_factor=2.0,
        analyzer_disable_thinking=None,
        summarizer_disable_thinking=None,
    )



def test_compute_expected_calls_formula() -> None:
    dream = _dream()
    assert compute_expected_calls(dream, enabled_analyzers_count=6) == 21



def test_compute_expected_calls_rejects_invalid_dream() -> None:
    dream = DreamInput(trait_layer=["t1"], background_layer="b", shots=[], dream_id=None)
    with pytest.raises(ValueError):
        compute_expected_calls(dream, enabled_analyzers_count=6)



def test_read_completed_calls_counts_lines(tmp_path: Path) -> None:
    calls = tmp_path / "calls.jsonl"
    calls.write_text('{"a":1}\n{"b":2}\n', encoding="utf-8")
    assert read_completed_calls(calls) == 2



def test_compute_calls_per_second_and_eta_running() -> None:
    rate, eta = compute_calls_per_second_and_eta(
        completed_calls=10,
        expected_calls=30,
        started_at=100.0,
        status="running",
        now=110.0,
    )
    assert rate == pytest.approx(1.0)
    assert eta == pytest.approx(20.0)



def test_compute_calls_per_second_and_eta_completed() -> None:
    rate, eta = compute_calls_per_second_and_eta(
        completed_calls=30,
        expected_calls=30,
        started_at=100.0,
        status="completed",
        now=115.0,
    )
    assert rate == pytest.approx(2.0)
    assert eta == pytest.approx(0.0)



def test_job_manager_success_transition(tmp_path: Path) -> None:
    def fake_analyze(*, dream, outdir, runtime_config, llm_config):
        run_dir = outdir / (dream.dream_id or "missing")
        run_dir.mkdir(parents=True, exist_ok=True)
        calls = run_dir / "calls.jsonl"
        calls.write_text("\n".join(["{}"] * 5) + "\n", encoding="utf-8")
        report = run_dir / "report.md"
        report.write_text("# report", encoding="utf-8")
        expanded = run_dir / "matrix_expanded.csv"
        expanded.write_text("shot_id\n0\n", encoding="utf-8")
        long_csv = run_dir / "matrix_long.csv"
        long_csv.write_text("shot_id\n0\n", encoding="utf-8")
        return RunArtifacts(
            run_dir=run_dir,
            report_md=report,
            matrix_expanded_csv=expanded,
            matrix_long_csv=long_csv,
            calls_jsonl=calls,
        )

    manager = JobManager(
        max_concurrent_jobs=1,
        analyze_fn=fake_analyze,
        enabled_count_fn=lambda _: 2,
    )

    job_id = manager.submit_job(
        dream=_dream(),
        runtime_config=_runtime(),
        llm_config=_llm(),
        outdir=tmp_path,
        username="alice",
    )
    early = manager.get_job(job_id)
    assert early is not None
    assert early.calls_jsonl is not None

    deadline = time.time() + 5
    status = manager.get_job(job_id)
    while status is not None and status.status in {"pending", "running"} and time.time() < deadline:
        time.sleep(0.05)
        status = manager.get_job(job_id)

    assert status is not None
    assert status.status == "completed"
    assert status.run_dir is not None
    assert status.calls_jsonl is not None
    assert status.completed_calls >= status.expected_calls



def test_job_manager_failure_transition(tmp_path: Path) -> None:
    def fake_analyze_fail(*, dream, outdir, runtime_config, llm_config):
        raise RuntimeError("boom")

    manager = JobManager(
        max_concurrent_jobs=1,
        analyze_fn=fake_analyze_fail,
        enabled_count_fn=lambda _: 2,
    )

    job_id = manager.submit_job(
        dream=_dream(),
        runtime_config=_runtime(),
        llm_config=_llm(),
        outdir=tmp_path,
        username="alice",
    )

    deadline = time.time() + 5
    status = manager.get_job(job_id)
    while status is not None and status.status in {"pending", "running"} and time.time() < deadline:
        time.sleep(0.05)
        status = manager.get_job(job_id)

    assert status is not None
    assert status.status == "failed"
    assert status.error is not None
    assert "boom" in status.error
