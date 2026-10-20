#!/bin/sh
# Shared Qwen3.6 API and web chat: 128K Q8, 2K prefill, 14 CPU threads, greedy MTP.
set -eu
STRATA_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$STRATA_ROOT"
exec "$STRATA_ROOT/.venv/bin/python" \
  "$STRATA_ROOT/serve/server.py" \
  --engine strata \
  --config "$STRATA_ROOT/strata-qwen36-ud-q4_k_xl.json" \
  --host 127.0.0.1 \
  --port 1234
