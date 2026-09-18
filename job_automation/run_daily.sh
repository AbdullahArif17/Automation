#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs data data/screenshots

# Load environment variables if present
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

echo "=== Job Automation Run started at $(date -u) ===" >> logs/job_automation.log

# Run the daily job applier (max 5 applications per day, min 75% match, jobs < 48h old)
"$SCRIPT_DIR/venv/bin/python" -m app.main \
    --max-apps 5 \
    --max-age-hours 48 \
    --min-match-score 75.0 \
    --profile config/profile.yaml \
    --resume assets/resume.pdf \
    --db-path data/jobs.db >> logs/job_automation.log 2>&1

echo "=== Job Automation Run finished at $(date -u) ===" >> logs/job_automation.log
