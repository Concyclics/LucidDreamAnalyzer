"""CLI entrypoint for lucid dream multi-agent analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_llm_config_from_env, load_runtime_config
from .io_schema import load_dream_input
from .orchestrator import analyze_dream
from .registry import load_registry



def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(description="Lucid dream multi-agent analyzer")
    parser.add_argument("--input", type=Path, help="Path to dream input (.json or template text)")
    parser.add_argument("--outdir", type=Path, default=Path("runs"), help="Output root directory")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("prompts/analyzers.json"),
        help="Analyzer registry JSON path",
    )
    parser.add_argument(
        "--validate-registry",
        action="store_true",
        help="Validate analyzer registry and exit",
    )
    parser.add_argument(
        "--no-compat-section",
        action="store_true",
        help="Disable canonical compatibility matrix section in report",
    )
    return parser



def main(argv: list[str] | None = None) -> int:
    """CLI main function."""

    parser = build_parser()
    args = parser.parse_args(argv)

    runtime = load_runtime_config(
        registry_path=args.registry,
        compatibility_layer=not args.no_compat_section,
    )

    try:
        _ = load_registry(
            path=runtime.registry_path,
            prompt_dir=runtime.prompt_dir,
            strict=runtime.strict_registry,
        )
    except Exception as exc:
        print(f"Registry validation failed: {exc}", file=sys.stderr)
        return 2

    if args.validate_registry:
        print(f"Registry OK: {runtime.registry_path}")
        return 0

    if args.input is None:
        parser.error("--input is required unless --validate-registry is used")

    try:
        dream = load_dream_input(args.input)
        llm = load_llm_config_from_env()
        artifacts = analyze_dream(
            dream=dream,
            outdir=args.outdir,
            runtime_config=runtime,
            llm_config=llm,
        )
    except Exception as exc:
        print(f"Run failed: {exc}", file=sys.stderr)
        return 1

    print(f"Run directory: {artifacts.run_dir}")
    print(f"Report: {artifacts.report_md}")
    if artifacts.matrix_expanded_csv is not None:
        print(f"Expanded CSV: {artifacts.matrix_expanded_csv}")
    if artifacts.matrix_long_csv is not None:
        print(f"Long CSV: {artifacts.matrix_long_csv}")
    print(f"Calls log: {artifacts.calls_jsonl}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
