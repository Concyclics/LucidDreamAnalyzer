"""Background job manager for web-triggered analysis runs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import os
from pathlib import Path
import threading
import time
from typing import Callable, Literal
import uuid

from .config import LLMConfig, RuntimeConfig
from .io_schema import DreamInput, RunArtifacts, validate_dream_input
from .orchestrator import analyze_dream
from .registry import load_registry


JobState = Literal["pending", "running", "completed", "failed"]


@dataclass
class WebJobStatus:
    """Status container for one web-submitted analysis job."""

    job_id: str
    username: str
    status: JobState
    submitted_at: float
    started_at: float | None
    finished_at: float | None
    error: str | None
    run_dir: Path | None
    calls_jsonl: Path | None
    expected_calls: int
    completed_calls: int



def compute_expected_calls(dream: DreamInput, enabled_analyzers_count: int) -> int:
    """Compute expected total model calls for progress tracking.

    Formula: (shots + 1) * (enabled_analyzers_count + 1)
    - +1 shot for background pre-shot
    - +1 call each shot for summarizer
    """

    validate_dream_input(dream)
    if enabled_analyzers_count <= 0:
        raise ValueError("enabled_analyzers_count must be > 0")
    return (len(dream.shots) + 1) * (enabled_analyzers_count + 1)



def read_completed_calls(calls_jsonl_path: Path | None) -> int:
    """Count completed calls from calls.jsonl."""

    if calls_jsonl_path is None or not calls_jsonl_path.exists():
        return 0
    with calls_jsonl_path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def compute_calls_per_second_and_eta(
    *,
    completed_calls: int,
    expected_calls: int,
    started_at: float | None,
    status: JobState,
    now: float | None = None,
) -> tuple[float, float | None]:
    """Compute calls/sec and ETA seconds similar to tqdm metrics."""

    if started_at is None:
        return 0.0, None

    now_ts = time.time() if now is None else now
    elapsed = max(0.0, now_ts - started_at)
    if elapsed <= 0 or completed_calls <= 0:
        if status == "completed":
            return 0.0, 0.0
        return 0.0, None

    rate = completed_calls / elapsed
    remaining = max(0, expected_calls - completed_calls)

    if status == "completed":
        return rate, 0.0
    if rate <= 0:
        return 0.0, None
    return rate, (remaining / rate)


class JobManager:
    """Thread-safe in-memory job manager."""

    def __init__(
        self,
        *,
        max_concurrent_jobs: int = 2,
        analyze_fn: Callable[..., RunArtifacts] | None = None,
        enabled_count_fn: Callable[[RuntimeConfig], int] | None = None,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max(1, max_concurrent_jobs))
        self._analyze_fn = analyze_fn or analyze_dream
        self._enabled_count_fn = enabled_count_fn or self._default_enabled_count
        self._time_fn = time_fn or time.time
        self._lock = threading.Lock()
        self._jobs: dict[str, WebJobStatus] = {}

    @staticmethod
    def _default_enabled_count(runtime_config: RuntimeConfig) -> int:
        registry = load_registry(
            path=runtime_config.registry_path,
            prompt_dir=runtime_config.prompt_dir,
            strict=runtime_config.strict_registry,
        )
        return len(registry.enabled_analyzers())

    def submit_job(
        self,
        *,
        dream: DreamInput,
        runtime_config: RuntimeConfig,
        llm_config: LLMConfig,
        outdir: Path,
        username: str,
    ) -> str:
        """Submit one background analysis job and return job_id."""

        user = (username or "").strip()
        if not user:
            raise ValueError("username is required")

        dream_copy = DreamInput(
            dream_id=None,
            trait_layer=list(dream.trait_layer),
            background_layer=dream.background_layer,
            shots=list(dream.shots),
        )

        enabled_count = self._enabled_count_fn(runtime_config)
        expected_calls = compute_expected_calls(dream_copy, enabled_count)

        job_id = uuid.uuid4().hex
        dream_copy.dream_id = job_id

        run_dir = outdir / job_id
        calls_jsonl = run_dir / "calls.jsonl"

        status = WebJobStatus(
            job_id=job_id,
            username=user,
            status="pending",
            submitted_at=self._time_fn(),
            started_at=None,
            finished_at=None,
            error=None,
            run_dir=run_dir,
            calls_jsonl=calls_jsonl,
            expected_calls=expected_calls,
            completed_calls=0,
        )

        with self._lock:
            self._jobs[job_id] = status

        self._executor.submit(
            self._run_job,
            job_id=job_id,
            dream=dream_copy,
            runtime_config=runtime_config,
            llm_config=llm_config,
            outdir=outdir,
        )

        return job_id

    def _run_job(
        self,
        *,
        job_id: str,
        dream: DreamInput,
        runtime_config: RuntimeConfig,
        llm_config: LLMConfig,
        outdir: Path,
    ) -> None:
        with self._lock:
            status = self._jobs[job_id]
            status.status = "running"
            status.started_at = self._time_fn()

        try:
            artifacts = self._analyze_fn(
                dream=dream,
                outdir=outdir,
                runtime_config=runtime_config,
                llm_config=llm_config,
            )
            with self._lock:
                status = self._jobs[job_id]
                status.status = "completed"
                status.finished_at = self._time_fn()
                status.run_dir = artifacts.run_dir
                status.calls_jsonl = artifacts.calls_jsonl
                status.completed_calls = max(
                    status.expected_calls,
                    read_completed_calls(artifacts.calls_jsonl),
                )
        except Exception as exc:
            with self._lock:
                status = self._jobs[job_id]
                status.status = "failed"
                status.finished_at = self._time_fn()
                status.error = f"{exc.__class__.__name__}: {exc}"
                status.completed_calls = read_completed_calls(status.calls_jsonl)

    def _refresh_progress(self, status: WebJobStatus) -> None:
        if status.calls_jsonl is None and status.run_dir is not None:
            candidate = status.run_dir / "calls.jsonl"
            if candidate.exists():
                status.calls_jsonl = candidate

        if status.calls_jsonl is not None:
            status.completed_calls = read_completed_calls(status.calls_jsonl)

        if status.status == "completed":
            status.completed_calls = max(status.completed_calls, status.expected_calls)

    def get_job(self, job_id: str) -> WebJobStatus | None:
        """Get one job status by job id."""

        with self._lock:
            status = self._jobs.get(job_id)
            if status is None:
                return None
            self._refresh_progress(status)
            return replace(status)

    def list_jobs_for_user(self, username: str) -> list[WebJobStatus]:
        """List jobs for one username, newest first."""

        user = (username or "").strip()
        with self._lock:
            rows = [job for job in self._jobs.values() if job.username == user]
            for row in rows:
                self._refresh_progress(row)
            rows.sort(key=lambda item: item.submitted_at, reverse=True)
            return [replace(item) for item in rows]



def default_max_concurrent_jobs() -> int:
    """Read max concurrent web jobs from env."""

    raw = os.getenv("LUCID_WEB_MAX_CONCURRENT_JOBS", "2")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("LUCID_WEB_MAX_CONCURRENT_JOBS must be an integer") from exc
    return max(1, value)
