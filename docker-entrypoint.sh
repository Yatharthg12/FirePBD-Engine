#!/usr/bin/env sh
set -eu

WORKERS="${UVICORN_WORKERS:-}"
if [ -z "${WORKERS}" ]; then
  WORKERS="${WEB_CONCURRENCY:-1}"
fi

PORT="${PORT:-}"
if [ -z "${PORT}" ]; then
  PORT="${UVICORN_PORT:-8000}"
fi

exec python -m uvicorn backend.main:app --host 0.0.0.0 --port "${PORT}" --workers "${WORKERS}"
