"""FFmpeg video composition for YouTube Shorts (1080x1920, 9:16, 30fps).

Composes: image/video scenes with motions + voice audio + background music + captions.
All local, zero-cost. Uses ffmpeg binary (must be in PATH).
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import shutil

from app.content.visual_plan import VisualPlan, Scene
from app.media.asset_manager import AssetRecord
from app.media.captions import CaptionTrack, write_caption_files
from app.media.voice import VoiceResult
from app.utils.logging import get_logger
from app.utils.retry import retry

logger = get_logger(__name__)

# Output specs
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
VIDEO_FPS = 30
VIDEO_CODEC = "libx264"
VIDEO_PRESET = "fast"
VIDEO_CRF = 23
AUDIO_CODEC = "aac"
AUDIO_BITRATE = "128k"


@dataclass
class RenderResult:
    output_path: str
    duration: float
    width: int
    height: int


class VideoEditor:
    def __init__(self, ffmpeg_bin: str = "ffmpeg", ffprobe_bin: str = "ffprobe"):
        self.ffmpeg = ffmpeg_bin
        self.ffprobe = ffprobe_bin

    def check_available(self) -> bool:
        return (shutil.which(self.ffmpeg) is not None and
                shutil.which(self.ffprobe) is not None)

    def render(
        self,
        plan: VisualPlan,
        scene_assets: list[AssetRecord],  # parallel to plan.scenes
        voice: VoiceResult,
        captions: CaptionTrack,
        music_path: Optional[str] = None,
        music_volume: float = 0.1,
        output_path: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> RenderResult:
        """
        Build the final Short.

        Pipeline:
        1. Build filter_complex for each scene (scale, crop, motion)
        2. Concatenate scenes
        3. Mix voice + music
        4. Burn captions (ASS)
        5. Output MP4
        """
        if not self.check_available():
            raise RuntimeError("ffmpeg/ffprobe not found in PATH")

        if output_path is None:
            output_path = f"output/short_{job_id or 'out'}.mp4"

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Write caption file
        caption_files = write_caption_files(captions, output_path.replace(".mp4", ""), formats=["ass"])
        ass_path = caption_files["ass"]

        # Build filter graph
        filter_parts = []
        input_args = []
        # Scene inputs are appended one '-i' flag at a time, in order, so
        # scene i maps directly to ffmpeg input index i. Do NOT derive this
        # from len(input_args) — each scene adds 6 args, so that breaks.
        scene_inputs = self._scene_input_indices(len(plan.scenes))

        # Scene inputs
        for i, (scene, asset) in enumerate(zip(plan.scenes, scene_assets)):
            dur = scene.end - scene.start
            if asset.local_path and os.path.exists(asset.local_path):
                # Use -framerate 1 on input so looped image yields exactly 1 frame;
                # zoompan's own d/fps then controls output duration/frame count.
                # Without this, image2 defaults to ~25fps → zoompan's 'd' applies per
                # input frame, massively inflating duration.
                input_args.extend(["-loop", "1", "-framerate", "1", "-t", str(dur), "-i", asset.local_path])
            else:
                # Fallback: generate solid color using ffmpeg color source.
                # lavfi color source with explicit rate=VIDEO_FPS produces correct
                # frame count natively; no input framerate override needed.
                input_args.extend(["-f", "lavfi", "-t", str(dur), "-i",
                                 f"color=c=0x1a1a2e:size={VIDEO_WIDTH}x{VIDEO_HEIGHT}:rate={VIDEO_FPS}"])

        # Voice input
        voice_idx = len(scene_inputs)
        input_args.extend(["-i", voice.audio_path])

        # Music input (optional)
        music_idx = None
        if music_path and os.path.exists(music_path):
            music_idx = voice_idx + 1
            input_args.extend(["-stream_loop", "-1", "-i", music_path])

        # Build per-scene filters
        for i, (scene, asset) in enumerate(zip(plan.scenes, scene_assets)):
            inp = scene_inputs[i]
            dur = scene.end - scene.start
            vf = self._build_scene_filter(scene, asset, dur)
            filter_parts.append(f"[{inp}:v]{vf}[v{i}]")

        # Concatenate video streams
        concat_inputs = "".join(f"[v{i}]" for i in range(len(plan.scenes)))
        filter_parts.append(f"{concat_inputs}concat=n={len(plan.scenes)}:v=1:a=0[video]")

        # Audio mixing
        audio_filter = f"[{voice_idx}:a]volume=1.0[voice]"
        filter_parts.append(audio_filter)

        if music_idx is not None:
            filter_parts.append(f"[{music_idx}:a]volume={music_volume}[music]")
            filter_parts.append(f"[voice][music]amix=inputs=2:duration=first:dropout_transition=3[audio]")
        else:
            filter_parts.append("[voice]anull[audio]")

        # Caption burning (ASS)
        # ffmpeg's subtitles filter parses ':' as an option separator, so the
        # Windows drive-letter colon in the path must be escaped as '\:'.
        safe_ass = ass_path.replace("\\", "/").replace(":", "\\:")
        filter_parts.append(f"[video]subtitles='{safe_ass}'[vout]")

        filter_complex = ";".join(filter_parts)

        # Build full command
        cmd = [
            self.ffmpeg, "-y",
            *input_args,
            "-filter_complex", filter_complex,
            "-map", "[vout]", "-map", "[audio]",
            "-c:v", VIDEO_CODEC, "-preset", VIDEO_PRESET, "-crf", str(VIDEO_CRF),
            "-r", str(VIDEO_FPS),
            "-c:a", AUDIO_CODEC, "-b:a", AUDIO_BITRATE,
            "-shortest",
            "-movflags", "+faststart",
            output_path,
        ]

        logger.info(f"rendering video -> {output_path}",
                    extra={"job_id": job_id, "stage": "render", "status": "start"})

        def _run():
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr[-1000:]}")
            return result

        retry(_run, max_attempts=2, retry_on=(subprocess.TimeoutExpired,))

        # Verify output
        dur = self._probe_duration(output_path)
        w, h = self._probe_resolution(output_path)

        logger.info(f"rendered {dur:.1f}s {w}x{h} -> {output_path}",
                    extra={"job_id": job_id, "stage": "render", "status": "done"})

        return RenderResult(output_path, dur, w, h)

    def _scene_input_indices(self, n_scenes: int) -> list[int]:
        """Return the ffmpeg input index for each scene.

        Scenes are added one '-i' flag at a time, in order, so scene i is
        input index i. This is intentionally a direct counter — the previous
        implementation derived the index from len(input_args) // 2 - 1, which
        assumed 2 args per scene but each scene actually adds 6, producing
        [2, 5, 8, ...] and referencing non-existent streams.
        """
        return list(range(n_scenes))

    def _build_scene_filter(self, scene: Scene, asset: AssetRecord, dur: float) -> str:
        """Build filter for a single scene with motion."""
        # Base: scale to cover 1080x1920 (crop if needed)
        base = f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:force_original_aspect_ratio=increase,crop={VIDEO_WIDTH}:{VIDEO_HEIGHT}"

        if scene.visual_type == "text" or scene.visual_type == "graphic":
            return f"{base},setsar=1"

        motion = scene.motion
        if motion == "static":
            return f"{base},setsar=1"
        elif motion == "zoom_in":
            # Progressive zoom from 1.0 to 1.15 over the scene duration.
            # Requires input framerate=1 (set in render()) so zoompan's 'd'
            # applies once per scene, not per input frame.
            return (f"{base},zoompan=z='min(1.15,zoom+0.0015)':"
                    f"d={int(dur*VIDEO_FPS)}:s={VIDEO_WIDTH}x{VIDEO_HEIGHT}:fps={VIDEO_FPS},setsar=1")
        elif motion == "zoom_out":
            # Progressive zoom out from 1.15 to 1.0
            return (f"{base},zoompan=z='max(1.0,1.15-zoom*0.0015)':"
                    f"d={int(dur*VIDEO_FPS)}:s={VIDEO_WIDTH}x{VIDEO_HEIGHT}:fps={VIDEO_FPS},setsar=1")
        elif motion == "pan":
            # Pan horizontally (for wider source) - simple crop animation
            return f"{base},crop={VIDEO_WIDTH}:{VIDEO_HEIGHT}:{VIDEO_WIDTH}*(1-t/{dur}):0,setsar=1"
        elif motion == "ken_burns":
            # Slow zoom + pan (Ken Burns effect)
            return (f"{base},zoompan=z='min(1.1,zoom+0.001)':"
                    f"d={int(dur*VIDEO_FPS)}:s={VIDEO_WIDTH}x{VIDEO_HEIGHT}:fps={VIDEO_FPS}:"
                    f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)',setsar=1")
        else:
            return f"{base},setsar=1"

    def _probe_duration(self, path: str) -> float:
        cmd = [self.ffprobe, "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", path]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return float(result.stdout.strip())
        except Exception:
            return 0.0

    def _probe_resolution(self, path: str) -> tuple[int, int]:
        cmd = [self.ffprobe, "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=width,height", "-of", "csv=p=0", path]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            w, h = map(int, result.stdout.strip().split(","))
            return w, h
        except Exception:
            return VIDEO_WIDTH, VIDEO_HEIGHT