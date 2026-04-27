#!/usr/bin/env sh
set -eu

WORKERS="${UVICORN_WORKERS:-2}"

exec python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers "${WORKERS}"
