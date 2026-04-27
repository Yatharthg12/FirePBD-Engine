#!/usr/bin/env sh
set -eu

WORKERS="${UVICORN_WORKERS:-${WEB_CONCURRENCY:-1}}"
PORT="${PORT:-${UVICORN_PORT:-8000}}"

exec python -m uvicorn backend.main:app --host 0.0.0.0 --port "${PORT}" --workers "${WORKERS}"
