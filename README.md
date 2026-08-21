# YouTube Shorts Automation

Zero-cost, automated YouTube Shorts generation pipeline for an English-language
channel focused on AI, technology, programming, and emerging tech.

**Target cost: $0/month.** Local open-source software, free APIs/tiers, and
FFmpeg-based rendering. No paid VPS, paid AI-video, paid voice, or paid stock.

## Architecture

```
Schedule → Topic Discovery → Scoring → Research → Verify → Script → QC
        → Visual Plan → Free Media → Free TTS → FFmpeg → Captions
        → QC → Duplicate Check → Metadata → Upload → Analytics
```

Every external service (LLM, voice, media, YouTube) is behind a provider
abstraction so paid providers can be added later without rewrites.

## Requirements

- Python 3.12+
- FFmpeg (for later phases)
- SQLite (stdlib)
- Git

## Installation

```bash
git clone <repo> && cd youtube-shorts-automation
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env   # fill in secrets
```

## FFmpeg

Install FFmpeg and ensure `ffmpeg` is on PATH (needed from Phase 8):

```bash
# Windows (choco)
choco install ffmpeg
# macOS
brew install ffmpeg
# Debian/Ubuntu
sudo apt install ffmpeg
```

## Usage

```bash
python -m app.main --help          # show commands
python -m app.main --topic "AI topic"
python -m app.main --dry-run       # plan pipeline, no upload/spend
python -m app.main --generate
python -m app.main --upload
python -m app.main --analytics
python -m app.main --status
```

Publishing is disabled by default (`AUTO_UPLOAD=false`). One-time setup, then
schedule locally (cron / Task Scheduler) or via GitHub Actions.

## Configuration

See `.env.example`. Secrets are read from environment / `.env` at runtime and
never committed.

## Development Phases

1. Skeleton, config, logging, CLI (current)
2. SQLite + job states + file/hash/error tracking
3. LLM abstraction + script generation/evaluation
4. Topic discovery + research + scoring
5. Visual planning + free asset management
6. Local/free TTS
7. Caption generation
8. FFmpeg rendering
9. Metadata + duplicate detection + QC
10. YouTube OAuth + upload
11. Analytics + optimization
12. Scheduler + Docker
13. Free cloud deployment investigation

## Cost Control

`MAX_DAILY_GENERATIONS=2` caps work. Free-tier providers are verified before
use. If a service requires payment, the pipeline stops and reports — it never
silently switches to a paid service.

## Security

No secrets in source. Safe subprocess execution (no shell). File validation and
size limits. Least-privilege OAuth scopes.

## Compliance

Original content only. Licensed/permitted media. No reuploads or scraped clips.

## Project Structure

```
app/        # package (config, research, ai, content, media, video, youtube, storage, utils)
prompts/    # prompt templates
assets/     # music, fonts, templates
output/     # generated videos
data/       # sqlite db
tests/      # unit tests
```
