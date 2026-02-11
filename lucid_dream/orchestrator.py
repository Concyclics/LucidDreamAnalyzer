"""Top-level orchestration for dream analysis runs."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path

from .analyzers import AnalyzerRunner
from .config import LLMConfig, RuntimeConfig, load_llm_config_from_env, load_runtime_config
from .io_schema import (
    DreamInput,
    RunArtifacts,
    ShotAnalysis,
    ShotContext,
    validate_dream_input,
    validate_shot_vectors,
)
from .logging_utils import CallLogger
from .prompt_loader import load_analyzer_prompts
from .registry import load_registry
from .report import build_report_markdown, export_matrix_expanded_csv, export_matrix_long_csv
from .summarizer import ShotSummarizer



def _resolve_dream_id(dream: DreamInput) -> str:
    if dream.dream_id and dream.dream_id.strip():
        return dream.dream_id.strip()
    ts = datetime.now(timezone.utc).strftime("dream_%Y%m%dT%H%M%SZ")
    return ts


async def analyze_dream_async(
    *,
    dream: DreamInput,
    outdir: Path,
    runtime_config: RuntimeConfig,
    llm_config: LLMConfig,
) -> RunArtifacts:
    """Analyze one dream and generate all run artifacts."""

    validate_dream_input(dream)

    registry = load_registry(
        path=runtime_config.registry_path,
        prompt_dir=runtime_config.prompt_dir,
        strict=runtime_config.strict_registry,
    )
    prompts = load_analyzer_prompts(registry, runtime_config.prompt_dir)

    dream_id = _resolve_dream_id(dream)
    run_dir = outdir / dream_id
    run_dir.mkdir(parents=True, exist_ok=True)

    calls_path = run_dir / "calls.jsonl"
    call_logger = CallLogger(calls_path)

    analyzer_runner = AnalyzerRunner(llm_config=llm_config, logger=call_logger, dream_id=dream_id)
    summarizer = ShotSummarizer(
        llm_config=llm_config,
        logger=call_logger,
        dream_id=dream_id,
        max_chars=runtime_config.summarizer_max_chars,
    )

    analyses: list[ShotAnalysis] = []
    contexts: list[ShotContext] = []
    prev_summary = ""

    # Phase 1: build contexts + rolling summaries sequentially.
    summary_after_shot: dict[int, str] = {}
    ordered_inputs: list[tuple[int, str]] = [(0, dream.background_layer)] + [
        (idx, shot_text) for idx, shot_text in enumerate(dream.shots, start=1)
    ]
    for shot_id, shot_text in ordered_inputs:
        ctx = ShotContext(
            shot_id=shot_id,
            trait_layer=dream.trait_layer,
            background_layer=dream.background_layer,
            prev_shots_summary=prev_summary,
            current_shot=shot_text,
        )
        contexts.append(ctx)

        # Keep rolling summary semantics unchanged.
        updated_summary = await summarizer.update(
            shot_id=shot_id,
            trait_layer=dream.trait_layer,
            background_layer=dream.background_layer,
            prev_summary=prev_summary,
            current_shot=shot_text,
        )
        summary_after_shot[shot_id] = updated_summary
        prev_summary = updated_summary

    # Phase 2: run all analyzers in parallel across (shot * analyzer).
    results_by_shot = await analyzer_runner.run_all_analyzers_for_contexts(
        registry=registry,
        prompts=prompts,
        contexts=contexts,
    )

    for ctx in contexts:
        shot_analysis = ShotAnalysis(
            shot_id=ctx.shot_id,
            shot_text=ctx.current_shot,
            results=results_by_shot[ctx.shot_id],
            summary_after_shot=summary_after_shot[ctx.shot_id],
        )
        validate_shot_vectors(shot_analysis, registry)
        analyses.append(shot_analysis)

    snapshot_path = run_dir / "registry_snapshot.json"
    snapshot_path.write_text(
        json.dumps(registry.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_text = build_report_markdown(
        dream=dream,
        analyses=analyses,
        final_summary=prev_summary,
        registry=registry,
        include_compatibility_section=runtime_config.compatibility_layer,
    )
    report_path = run_dir / "report.md"
    report_path.write_text(report_text, encoding="utf-8")

    expanded_csv_path: Path | None = None
    if runtime_config.emit_expanded_csv:
        expanded_csv_path = run_dir / "matrix_expanded.csv"
        export_matrix_expanded_csv(
            analyses=analyses,
            registry=registry,
            out_path=expanded_csv_path,
        )

    long_csv_path: Path | None = None
    if runtime_config.emit_long_csv:
        long_csv_path = run_dir / "matrix_long.csv"
        export_matrix_long_csv(
            analyses=analyses,
            registry=registry,
            out_path=long_csv_path,
        )

    return RunArtifacts(
        run_dir=run_dir,
        report_md=report_path,
        matrix_expanded_csv=expanded_csv_path,
        matrix_long_csv=long_csv_path,
        calls_jsonl=calls_path,
    )



def analyze_dream(
    *,
    dream: DreamInput,
    outdir: Path,
    runtime_config: RuntimeConfig | None = None,
    llm_config: LLMConfig | None = None,
) -> RunArtifacts:
    """Synchronous wrapper for the async pipeline."""

    runtime = runtime_config or load_runtime_config()
    llm = llm_config or load_llm_config_from_env()
    return asyncio.run(
        analyze_dream_async(
            dream=dream,
            outdir=outdir,
            runtime_config=runtime,
            llm_config=llm,
        )
    )
