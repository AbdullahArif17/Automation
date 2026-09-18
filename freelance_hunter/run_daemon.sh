#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs data

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

echo "=== Freelance Hunter Daemon started at $(date -u) ===" >> logs/freelance.log

exec "$SCRIPT_DIR/venv/bin/python" app/daemon.py >> logs/freelance.log 2>&1
