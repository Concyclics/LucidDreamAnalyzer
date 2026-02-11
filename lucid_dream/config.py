"""Runtime and LLM configuration."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_CANONICAL_TAGS = ("NVS", "PVS", "CS", "SSP", "ARS", "SMS")
DEFAULT_LLM_SETTINGS_PATH = Path("config/llm_settings.json")


@dataclass(frozen=True)
class LLMConfig:
    """Settings for OpenAI-compatible model calls."""

    api_key: str | None
    base_url: str | None
    analyzer_model: str
    summarizer_model: str
    analyzer_temperature: float
    summarizer_temperature: float
    timeout_s: float
    max_retries: int
    backoff_base_s: float
    backoff_factor: float
    analyzer_disable_thinking: bool | None
    summarizer_disable_thinking: bool | None


@dataclass(frozen=True)
class RuntimeConfig:
    """Settings for orchestrator behavior."""

    prompt_dir: Path
    registry_path: Path
    canonical_tags: tuple[str, ...]
    strict_registry: bool
    emit_long_csv: bool
    emit_expanded_csv: bool
    compatibility_layer: bool
    summarizer_max_chars: int


@dataclass(frozen=True)
class RegistryConfig:
    """Controls registry loading and validation."""

    path: Path
    prompt_dir: Path
    strict: bool = True



def load_llm_config_from_env() -> LLMConfig:
    """Load LLM settings from JSON + env overrides.

    Priority:
    1) Environment variable (if set)
    2) JSON settings file value
    3) Built-in default
    """

    settings_path = Path(
        os.getenv("LUCID_LLM_SETTINGS_JSON", str(DEFAULT_LLM_SETTINGS_PATH))
    )
    json_settings = _load_llm_settings_json(settings_path)

    analyzer_model = _pick_setting(
        env_key="LUCID_ANALYZER_MODEL",
        json_settings=json_settings,
        json_key="analyzer_model",
        default="gpt-4.1-mini",
    )
    summarizer_model = _pick_setting(
        env_key="LUCID_SUMMARIZER_MODEL",
        json_settings=json_settings,
        json_key="summarizer_model",
        default=analyzer_model,
    )

    return LLMConfig(
        api_key=_pick_setting(
            env_key="OPENAI_API_KEY",
            json_settings=json_settings,
            json_key="api_key",
            default=None,
        ),
        base_url=_pick_setting(
            env_key="OPENAI_BASE_URL",
            json_settings=json_settings,
            json_key="base_url",
            default=None,
        ),
        analyzer_model=analyzer_model,
        summarizer_model=summarizer_model,
        analyzer_temperature=float(
            _pick_setting(
                env_key="LUCID_ANALYZER_TEMPERATURE",
                json_settings=json_settings,
                json_key="analyzer_temperature",
                default="0.0",
            )
        ),
        summarizer_temperature=float(
            _pick_setting(
                env_key="LUCID_SUMMARIZER_TEMPERATURE",
                json_settings=json_settings,
                json_key="summarizer_temperature",
                default="0.0",
            )
        ),
        timeout_s=float(
            _pick_setting(
                env_key="LUCID_TIMEOUT_S",
                json_settings=json_settings,
                json_key="timeout_s",
                default="60",
            )
        ),
        max_retries=int(
            _pick_setting(
                env_key="LUCID_MAX_RETRIES",
                json_settings=json_settings,
                json_key="max_retries",
                default="2",
            )
        ),
        backoff_base_s=float(
            _pick_setting(
                env_key="LUCID_BACKOFF_BASE_S",
                json_settings=json_settings,
                json_key="backoff_base_s",
                default="0.8",
            )
        ),
        backoff_factor=float(
            _pick_setting(
                env_key="LUCID_BACKOFF_FACTOR",
                json_settings=json_settings,
                json_key="backoff_factor",
                default="2.0",
            )
        ),
        analyzer_disable_thinking=_parse_optional_bool(
            _pick_setting(
                env_key="LUCID_ANALYZER_DISABLE_THINKING",
                json_settings=json_settings,
                json_key="analyzer_disable_thinking",
                default=None,
            )
        ),
        summarizer_disable_thinking=_parse_optional_bool(
            _pick_setting(
                env_key="LUCID_SUMMARIZER_DISABLE_THINKING",
                json_settings=json_settings,
                json_key="summarizer_disable_thinking",
                default=None,
            )
        ),
    )


def _load_llm_settings_json(path: Path) -> dict[str, Any]:
    """Load optional JSON settings for LLM configuration."""

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"LLM settings JSON must be an object: {path}")
    return payload


def _pick_setting(
    *,
    env_key: str,
    json_settings: dict[str, Any],
    json_key: str,
    default: Any,
) -> Any:
    """Read setting from env first, then JSON, then default."""

    env_val = os.getenv(env_key)
    if env_val is not None:
        return env_val
    if json_key in json_settings:
        return json_settings[json_key]
    return default


def _parse_optional_bool(raw_value: Any) -> bool | None:
    """Parse an optional boolean env value.

    Returns:
    - True / False when value is provided and parseable
    - None when value is unset or blank
    """

    if raw_value is None:
        return None

    if isinstance(raw_value, bool):
        return raw_value

    value = str(raw_value).strip().lower()
    if value == "":
        return None
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False

    raise ValueError(
        f"Invalid boolean env value: {raw_value!r}. "
        "Use one of: true/false, 1/0, yes/no, on/off."
    )


def build_thinking_extra_body(disable_thinking: bool | None) -> dict[str, Any] | None:
    """Build provider-specific `extra_body` for thinking mode.

    Rules:
    - True  => thinking disabled
    - False => thinking enabled
    - None  => omit extra_body
    """

    if disable_thinking is None:
        return None
    thinking_type = "disabled" if disable_thinking else "enabled"
    return {"thinking": {"type": thinking_type}}



def load_runtime_config(
    *,
    prompt_dir: str | Path = "prompts",
    registry_path: str | Path = "prompts/analyzers.json",
    strict_registry: bool = True,
    emit_long_csv: bool = True,
    emit_expanded_csv: bool = True,
    compatibility_layer: bool = True,
    summarizer_max_chars: int = 1200,
) -> RuntimeConfig:
    """Construct runtime configuration."""

    return RuntimeConfig(
        prompt_dir=Path(prompt_dir),
        registry_path=Path(registry_path),
        canonical_tags=DEFAULT_CANONICAL_TAGS,
        strict_registry=strict_registry,
        emit_long_csv=emit_long_csv,
        emit_expanded_csv=emit_expanded_csv,
        compatibility_layer=compatibility_layer,
        summarizer_max_chars=summarizer_max_chars,
    )
