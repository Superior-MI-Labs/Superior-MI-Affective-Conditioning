#!/usr/bin/env bash
# Start an authenticated local-only GGUF model for the public research page.
set -euo pipefail
MODEL="${1:-$HOME/.cache/huggingface/hub/models--ggml-org--Qwen3-1.7B-GGUF/snapshots/daeb8e2d528a760970442092f6bf1e55c3b659eb/Qwen3-1.7B-Q4_K_M.gguf}"
if [[ ! -f "$MODEL" ]]; then
  printf 'Model file not found: %s\nSupply the GGUF path as the first argument.\n' "$MODEL" >&2
  exit 1
fi
# Some llama.cpp builds serialize --cors-origins verbatim into the response
# header. A browser requires a single matching Access-Control-Allow-Origin.
CORS_ORIGIN="${SMI_CORS_ORIGIN:-https://superior-mi-labs.github.io}"
case "$CORS_ORIGIN" in
  https://superior-mi-labs.github.io|http://127.0.0.1:8000|http://localhost:8000) ;;
  *) printf 'Unsupported SMI_CORS_ORIGIN: %s\nUse the public site origin or one of the documented loopback development origins.\n' "$CORS_ORIGIN" >&2; exit 1 ;;
esac
LLAMA_SERVER="${LLAMA_SERVER:-$(command -v llama-server || true)}"
if [[ ! -x "$LLAMA_SERVER" ]]; then
  printf 'llama-server was not found. Install/build llama.cpp and put llama-server on PATH, or set LLAMA_SERVER=/path/to/llama-server.\n' >&2
  exit 1
fi
KEY_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/smi-affective"
KEY_FILE="$KEY_DIR/llama-api-key"
umask 077
mkdir -p "$KEY_DIR"
if [[ ! -s "$KEY_FILE" ]]; then
  python3 - "$KEY_FILE" <<'PY'
import pathlib, secrets, sys
pathlib.Path(sys.argv[1]).write_text(secrets.token_urlsafe(32) + '\n', encoding='utf-8')
PY
fi
printf 'Local-only model service: http://127.0.0.1:8080\n'
printf 'Allowed web origin: %s\n' "$CORS_ORIGIN"
printf 'Paste this session API key into the GitHub Pages local-model field:\n'
cat "$KEY_FILE"
printf 'Starting %s\nPress Ctrl+C to stop the service.\n' "$MODEL"
exec "$LLAMA_SERVER" \
  --model "$MODEL" --host 127.0.0.1 --port 8080 --ctx-size 4096 \
  --n-gpu-layers "${SMI_GPU_LAYERS:-0}" \
  --cors-origins "$CORS_ORIGIN" \
  --api-key-file "$KEY_FILE"
