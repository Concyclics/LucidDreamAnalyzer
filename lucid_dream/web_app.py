"""Streamlit web app for lucid dream multi-agent analysis."""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

import streamlit as st

try:
    from .config import load_llm_config_from_env, load_runtime_config
    from .io_schema import DreamInput, parse_dream_json, parse_dream_template_text
    from .web_auth import verify_login
    from .web_jobs import (
        JobManager,
        compute_calls_per_second_and_eta,
        default_max_concurrent_jobs,
    )
except ImportError:
    # Supports `streamlit run lucid_dream/web_app.py` script execution.
    from lucid_dream.config import load_llm_config_from_env, load_runtime_config
    from lucid_dream.io_schema import DreamInput, parse_dream_json, parse_dream_template_text
    from lucid_dream.web_auth import verify_login
    from lucid_dream.web_jobs import (
        JobManager,
        compute_calls_per_second_and_eta,
        default_max_concurrent_jobs,
    )


DEFAULT_OUTDIR = Path("runs")


@st.cache_resource
def get_job_manager() -> JobManager:
    """Build a singleton job manager for the Streamlit process."""

    return JobManager(max_concurrent_jobs=default_max_concurrent_jobs())



def _max_upload_bytes() -> int:
    raw = os.getenv("LUCID_WEB_MAX_UPLOAD_BYTES", str(1024 * 1024))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("LUCID_WEB_MAX_UPLOAD_BYTES must be an integer") from exc
    return max(1024, value)



def _progress_poll_seconds() -> float:
    raw = os.getenv("LUCID_WEB_PROGRESS_POLL_SECONDS", "0.3")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("LUCID_WEB_PROGRESS_POLL_SECONDS must be a number") from exc
    return min(max(value, 0.1), 5.0)



def _format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    total = max(0, int(seconds))
    mins, secs = divmod(total, 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"



def _parse_user_input(mode: str, text_input: str, uploaded: Any) -> DreamInput:
    """Parse dream input from text area or file upload."""

    if mode == "Paste template text":
        raw = (text_input or "").strip()
        if not raw:
            raise ValueError("Please paste template text before starting analysis")
        return parse_dream_template_text(raw)

    if uploaded is None:
        raise ValueError("Please upload a file before starting analysis")

    content = uploaded.getvalue()
    if len(content) > _max_upload_bytes():
        raise ValueError("Uploaded file is too large")

    decoded = content.decode("utf-8")
    suffix = Path(uploaded.name).suffix.lower()
    if suffix == ".json":
        return parse_dream_json(decoded)
    return parse_dream_template_text(decoded)



def _show_login() -> None:
    st.subheader("Login")
    username = st.text_input("Username", key="login_username")
    provided_hash = st.text_input("SHA256 Hash", type="password", key="login_hash")

    if st.button("Login", type="primary"):
        if verify_login(username, provided_hash):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username.strip()
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid username or hash")



def _show_download_button(path: Path, label: str, mime: str) -> None:
    if not path.exists():
        return
    data = path.read_bytes()
    st.download_button(
        label=label,
        data=data,
        file_name=path.name,
        mime=mime,
    )



def _render_job_result(run_dir: Path) -> None:
    report_path = run_dir / "report.md"
    expanded_path = run_dir / "matrix_expanded.csv"
    long_path = run_dir / "matrix_long.csv"

    st.subheader("Report")
    if report_path.exists():
        st.markdown(report_path.read_text(encoding="utf-8"))
    else:
        st.warning("report.md not found")

    st.subheader("Downloads")
    _show_download_button(report_path, "Download report.md", "text/markdown")
    _show_download_button(expanded_path, "Download matrix_expanded.csv", "text/csv")
    if long_path.exists():
        _show_download_button(long_path, "Download matrix_long.csv", "text/csv")



def main() -> None:
    """Streamlit app entrypoint."""

    st.set_page_config(page_title="Lucid Dream Multi-Agent Analyzer", layout="wide")
    st.title("Lucid Dream Multi-Agent Analyzer")

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = ""
    if "active_job_id" not in st.session_state:
        st.session_state["active_job_id"] = None

    if not st.session_state["authenticated"]:
        _show_login()
        return

    username = st.session_state["username"]
    with st.sidebar:
        st.write(f"Logged in as: `{username}`")
        if st.button("Logout"):
            st.session_state["authenticated"] = False
            st.session_state["username"] = ""
            st.session_state["active_job_id"] = None
            st.rerun()

    st.subheader("Input")
    mode = st.radio("Input mode", ["Paste template text", "Upload file"], horizontal=True)

    text_input = ""
    uploaded = None
    if mode == "Paste template text":
        text_input = st.text_area("Paste template-like text", height=300)
    else:
        uploaded = st.file_uploader("Upload template/json file", type=["txt", "md", "json"])

    parsed: DreamInput | None = None
    parse_error: str | None = None

    has_content = bool(text_input.strip()) if mode == "Paste template text" else (uploaded is not None)
    if has_content:
        try:
            parsed = _parse_user_input(mode, text_input, uploaded)
            st.success("Input format verified")
            with st.expander("Parsed preview", expanded=True):
                st.write(f"Trait items: {len(parsed.trait_layer)}")
                st.write(f"Shots: {len(parsed.shots)}")
                st.write("First shot snippet:")
                st.code(parsed.shots[0][:300] if parsed.shots else "")
        except Exception as exc:
            parse_error = f"{exc.__class__.__name__}: {exc}"
            st.error(parse_error)

    if st.button("Start Analysis", type="primary", disabled=(parsed is None or parse_error is not None)):
        if parsed is None:
            st.error("Input is not ready")
        else:
            runtime = load_runtime_config()
            llm = load_llm_config_from_env()
            manager = get_job_manager()
            job_id = manager.submit_job(
                dream=parsed,
                runtime_config=runtime,
                llm_config=llm,
                outdir=DEFAULT_OUTDIR,
                username=username,
            )
            st.session_state["active_job_id"] = job_id
            st.success(f"Job submitted: {job_id}")
            st.rerun()

    manager = get_job_manager()
    jobs = manager.list_jobs_for_user(username)
    if not jobs:
        return

    active_job_id = st.session_state.get("active_job_id")
    active = manager.get_job(active_job_id) if active_job_id else None
    if active is None:
        active = jobs[0]
        st.session_state["active_job_id"] = active.job_id

    st.subheader("Progress")
    expected = max(1, active.expected_calls)
    if active.status == "completed":
        progress = 1.0
    elif active.status == "running":
        progress = min(active.completed_calls / expected, 0.99)
    else:
        progress = min(active.completed_calls / expected, 1.0)

    st.progress(progress)
    calls_per_sec, eta_seconds = compute_calls_per_second_and_eta(
        completed_calls=active.completed_calls,
        expected_calls=active.expected_calls,
        started_at=active.started_at,
        status=active.status,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Calls", f"{active.completed_calls}/{active.expected_calls}")
    c2.metric("Calls/s", f"{calls_per_sec:.2f}")
    c3.metric("ETA", _format_eta(eta_seconds))

    st.write(
        {
            "job_id": active.job_id,
            "status": active.status,
            "completed_calls": active.completed_calls,
            "expected_calls": active.expected_calls,
            "submitted_at": active.submitted_at,
            "started_at": active.started_at,
            "finished_at": active.finished_at,
        }
    )

    if active.status == "failed":
        st.error(active.error or "Job failed")
    elif active.status == "completed" and active.run_dir is not None:
        st.caption(f"Run directory: {active.run_dir}")
        _render_job_result(active.run_dir)
    elif active.status in {"pending", "running"}:
        time.sleep(_progress_poll_seconds())
        st.rerun()


if __name__ == "__main__":
    main()
