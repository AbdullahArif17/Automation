# YouTube Shorts Automation - System & Operations Guide

This document serves as the complete reference for the YouTube Shorts Clipper Automation bot, including server details, deployment instructions, scheduled tasks, and troubleshooting steps.

---

## 1. Cloud Server Details (Oracle Cloud)

* **Cloud Provider:** Oracle Cloud Infrastructure (OCI Always Free)
* **Instance Name:** `youtube-bot`
* **Shape:** `VM.Standard.A1.Flex` (2 OCPU ARM Ampere, 12 GB RAM - Guaranteed Always Free)
* **OS:** Ubuntu 24.04 LTS (aarch64)
* **Public IPv4:** `140.245.30.57`
* **Network:** VCN `new` with native IPv6 dual-stack (`/64` subnet enabled)
* **SSH Key Path (WSL):** `~/.ssh/ssh-key-2026-09-02.key`
* **SSH Key Path (Windows):** `D:\Code\Automation\ssh-key-2026-09-02.key`
* **Remote Project Directory:** `/home/ubuntu/Automation`

---

## 2. Quick Commands Cheatsheet

### SSH into Oracle Server
From your local WSL terminal:
```bash
ssh -i ~/.ssh/ssh-key-2026-09-02.key ubuntu@140.245.30.57
```

### Sync Local Changes to Cloud Server
Whenever you modify prompts, `.env` settings, or code on your PC, run this from WSL:
```bash
cd /mnt/d/Code/Automation
rsync -avz -e "ssh -i ~/.ssh/ssh-key-2026-09-02.key -o StrictHostKeyChecking=no" \
  --exclude 'input' \
  --exclude 'output' \
  --exclude 'miniconda*' \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '.gemini' \
  --exclude '*.MOV' \
  --exclude '*.mp4' \
  --exclude '*.mp3' \
  --exclude 'venv' \
  --exclude 'data' \
  ./ ubuntu@140.245.30.57:~/Automation/
```

### Check Live Server Logs
SSH into the server and run:
```bash
# View last 50 lines:
tail -n 50 ~/Automation/clipper.log

# Stream logs in real-time:
tail -f ~/Automation/clipper.log
```

### Trigger a Cloud Run Immediately from your PC
```bash
ssh -i ~/.ssh/ssh-key-2026-09-02.key ubuntu@140.245.30.57 "~/Automation/run_cron.sh"
```

### Run Locally on WSL (Optional)
```bash
source ~/miniconda/bin/activate
cd /mnt/d/Code/Automation
python -m app.clipper.__auto_main__
```

---

## 3. Automation Schedule (Cron Job)

The bot runs automatically on the cloud instance **3 times a day (every 8 hours)** via `cron`:

* **Crontab Rule:** `0 */8 * * * /home/ubuntu/Automation/run_cron.sh`
* **Execution Times (UTC):** `00:00`, `08:00`, `16:00`
* **Execution Times (PKT):** `05:00 AM`, `01:00 PM`, `09:00 PM`
* **Execution Times (IST):** `05:30 AM`, `01:30 PM`, `09:30 PM`
* **Log Location:** `/home/ubuntu/Automation/clipper.log`
* **Output per run:** 2 videos processed, 1 clip per video = 2 Shorts uploaded per run (~6 Shorts/day).

---

## 4. YouTube Anti-Bot Bypass Architecture

YouTube strictly detects and blocks datacenter IP ranges (AWS, Oracle, DigitalOcean) with `Sign in to confirm you're not a bot`. The following system architecture completely bypasses this:

1. **Dual-Stack IPv6 Routing:** Oracle VCN routes IPv6 through an Internet Gateway. `yt-dlp` uses `--force-ipv6` to circumvent IPv4 datacenter blacklists.
2. **Deno JavaScript Challenge Engine:** Deno is symlinked to `/usr/local/bin/deno`. `yt-dlp` uses `--js-runtimes deno` and `--remote-components ejs:github` to dynamically solve YouTube's n-challenge and player signatures.
3. **Authenticated Session Cookies:** YouTube requires an active signed-in session on datacenter connections. 
   - Cookie file location: `/home/ubuntu/Automation/cookies.txt`
   - Configured in `.env`: `YT_COOKIES_PATH=/home/ubuntu/Automation/cookies.txt`

---

## 5. How to Update Cookies (If Ever Expired)

YouTube login sessions typically last several months. If you ever see `Sign in to confirm you're not a bot` again in the logs:

1. Open your browser and log into YouTube with a **throwaway/burner Google account**.
2. Export the cookies using the *Get cookies.txt LOCALLY* browser extension.
3. Save the file to `D:\Code\Automation\cookies.txt` on your PC.
4. Upload it to the server from WSL:
   ```bash
   scp -i ~/.ssh/ssh-key-2026-09-02.key /mnt/d/Code/Automation/cookies.txt ubuntu@140.245.30.57:~/Automation/cookies.txt
   ```
The bot will immediately pick up the new session on the next run.

## 6. Video Quality Standards

The pipeline guarantees studio-grade vertical output:
* **Source Download:** `yt-dlp` requests `bestvideo+bestaudio/best` with `--merge-output-format mp4` to fetch the highest available resolution (1080p60, 1440p, or 4K).
* **AI Computer Vision:** OpenCV YuNet detects face locations at 100+ FPS, framing single speakers or generating a 2-speaker podcast split-screen.
* **Lanczos Resampling:** All scaling operations to 1080x1920 use `:flags=lanczos` to maintain maximum edge and facial sharpness.
* **Broadcast Encoding:**
  - Video codec: `libx264` with `-preset medium` and `-crf 17` (visually lossless master quality).
  - Frame rate: dynamically preserves up to `60 fps` for smooth motion.
  - Pixel format: `yuv420p` for universal mobile hardware playback.
  - Audio: AAC `192 kbps` at `48,000 Hz` stereo.

---

## 7. Project Structure Overview

```
Automation/
├── app/
│   ├── clipper/
│   │   ├── __auto_main__.py   # Entry point for the automated clipper
│   │   ├── storage_poller.py  # Searches YouTube/channel & handles downloads (yt-dlp)
│   │   ├── transcribe.py      # Audio extraction & faster-whisper transcription
│   │   ├── highlight.py       # Gemini AI viral segment detection
│   │   ├── captions.py        # Word-level SRT/ASS subtitle builder
│   │   ├── face_tracker.py    # AI face detection (YuNet) & smart 9:16 reframing
│   │   └── cut.py             # FFmpeg 9:16 vertical crop, cut, subtitle burn
│   ├── storage/
│   │   └── database.py        # SQLite database (videos, clips, run status)
│   └── youtube/
│       ├── auth.py            # OAuth 2.0 token refresh
│       └── uploader.py        # Uploads generated Short to your YouTube channel
├── data/
│   └── app.db                 # SQLite database tracking processed videos
├── .env                       # API keys, topics, limits, and configuration
├── cookies.txt                # Netscape-format YouTube session cookies
└── run_cron.sh                # Cron execution wrapper script (on server)
```
