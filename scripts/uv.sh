#!/usr/bin/env bash
# uv sync does not have --env-file; load the same local config for all uv commands.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_dir"
if [[ ! -f .env ]]; then
    printf '%s\n' 'Create .env from .env.example and set the storage paths first.' >&2
    exit 1
fi
set -a
source .env
set +a
: "${UV_CACHE_DIR:?Set UV_CACHE_DIR in .env}"
: "${UV_PROJECT_ENVIRONMENT:?Set UV_PROJECT_ENVIRONMENT in .env}"
exec uv "$@"
