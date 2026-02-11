from __future__ import annotations

from pathlib import Path

from lucid_dream.config import load_llm_config_from_env



def test_load_llm_config_from_json(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "llm.json"
    path.write_text(
        """
        {
          "api_key": "json-key",
          "base_url": "https://api.deepseek.com",
          "analyzer_model": "deepseek-reasoner",
          "summarizer_model": "deepseek-chat",
          "analyzer_temperature": 0.1,
          "summarizer_temperature": 0.2,
          "timeout_s": 70,
          "max_retries": 3,
          "backoff_base_s": 0.9,
          "backoff_factor": 2.1,
          "analyzer_disable_thinking": true,
          "summarizer_disable_thinking": false
        }
        """,
        encoding="utf-8",
    )

    monkeypatch.setenv("LUCID_LLM_SETTINGS_JSON", str(path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LUCID_ANALYZER_MODEL", raising=False)

    cfg = load_llm_config_from_env()
    assert cfg.api_key == "json-key"
    assert cfg.base_url == "https://api.deepseek.com"
    assert cfg.analyzer_model == "deepseek-reasoner"
    assert cfg.summarizer_model == "deepseek-chat"
    assert cfg.analyzer_temperature == 0.1
    assert cfg.summarizer_temperature == 0.2
    assert cfg.timeout_s == 70
    assert cfg.max_retries == 3
    assert cfg.backoff_base_s == 0.9
    assert cfg.backoff_factor == 2.1
    assert cfg.analyzer_disable_thinking is True
    assert cfg.summarizer_disable_thinking is False



def test_env_override_takes_priority(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "llm.json"
    path.write_text(
        '{"analyzer_model":"from-json","api_key":"json-key"}',
        encoding="utf-8",
    )
    monkeypatch.setenv("LUCID_LLM_SETTINGS_JSON", str(path))
    monkeypatch.setenv("LUCID_ANALYZER_MODEL", "from-env")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    cfg = load_llm_config_from_env()
    assert cfg.analyzer_model == "from-env"
    assert cfg.api_key == "env-key"
