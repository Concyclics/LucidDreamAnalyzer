# LLM settings (OpenAI-compatible)
export OPENAI_API_KEY="sk-e5b5507826934f9cada58fc49385050a"
#export OPENAI_API_KEY="sk-0b5f4e4e4f9c4b1a9d3f8e2c3a4b5c6d"
export OPENAI_BASE_URL="https://api.deepseek.com"   # optional if using OpenAI default
#export OPENAI_BASE_URL="https://api.moonshot.cn"   # optional if using OpenAI default
export LUCID_ANALYZER_MODEL="deepseek-reasoner"
export LUCID_SUMMARIZER_MODEL="deepseek-chat"

# Optional runtime tuning
export LUCID_ANALYZER_TEMPERATURE="0.0"
export LUCID_SUMMARIZER_TEMPERATURE="0.0"
export LUCID_TIMEOUT_S="60"
export LUCID_MAX_RETRIES="2"
export LUCID_BACKOFF_BASE_S="0.8"
export LUCID_BACKOFF_FACTOR="2.0"

# Run demo with template input
python -m lucid_dream.cli \
  --input prompts/template.txt \
  --registry prompts/analyzers.json \
  --outdir runs
