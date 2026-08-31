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


def _build_scene_input_args(asset: AssetRecord, dur: float) -> list[str]:
    """Build ffmpeg input arguments for a single scene's asset.

    Branches on asset.type to use correct input options:
    - image: -loop 1 -t <dur> -i (loops single frame for scene duration)
    - video: -t <dur> -i (trim if longer) or -stream_loop -1 -t <dur> -i (loop if shorter)
    - missing/fallback: lavfi color source
    """
    if asset.local_path and os.path.exists(asset.local_path):
        if asset.type == "video":
            # Real video file: do NOT use -loop 1 (image-specific).
            # If video is longer than needed, trim on input with -t.
            # If video is shorter, loop it with -stream_loop -1.
            asset_dur = max(0.0, asset.duration)
            if asset_dur >= dur:
                # Video covers the scene; trim to exact duration.
                return ["-t", str(dur), "-i", asset.local_path]
            else:
                # Video shorter than scene: loop it to fill duration.
                return ["-stream_loop", "-1", "-t", str(dur), "-i", asset.local_path]
        else:
            # Image (or unknown type treated as image): loop single frame.
            return ["-loop", "1", "-t", str(dur), "-i", asset.local_path]
    else:
        # Fallback: generate solid color using ffmpeg color source.
        return ["-f", "lavfi", "-t", str(dur), "-i",
                f"color=c=0x1a1a2e:size={1080}x{1920}:rate=30"]

# Output specs
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
VIDEO_FPS = 30
VIDEO_CODEC = "libx264"
# ultrafast: ~2-3x faster than "fast" on constrained CI hardware (2-core runners).
# Shorts are already heavily compressed for mobile; preset choice affects file
# size/encode-time tradeoff, not visual quality at a fixed CRF.
VIDEO_PRESET = "fast"
VIDEO_CRF = 18
AUDIO_CODEC = "aac"
AUDIO_BITRATE = "128k"

# Crossfade transition duration between scenes (seconds)
XFADE_DURATION = 0.4

# ffmpeg subprocess timeout (seconds). Baseline render of a 5-scene zoompan
# Short at CRF 23 on a 2-core GitHub runner: ~120-180s with -preset ultrafast.
# 900s = 15 min gives 3-5x headroom for slower runners / longer videos.
RENDER_TIMEOUT = 900


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
            if i < len(plan.scenes) - 1:
                dur += XFADE_DURATION
            input_args.extend(_build_scene_input_args(asset, dur))

        # Voice input
        voice_idx = len(scene_inputs)
        input_args.extend(["-i", voice.audio_path])

        # Music input (optional)
        music_idx = None
        if music_path and os.path.exists(music_path):
            music_idx = voice_idx + 1
            input_args.extend(["-stream_loop", "-1", "-i", music_path])

        # Build per-scene filters
        # Append fps=VIDEO_FPS to every scene's output so all scenes
        # share a consistent timebase before entering the xfade chain.
        # Without this, zoompan-based motions produce a different
        # internal timebase (e.g. 1/15360) than scale/crop (1/30),
        # causing xfade to fail with a timebase mismatch error.
        for i, (scene, asset) in enumerate(zip(plan.scenes, scene_assets)):
            inp = scene_inputs[i]
            dur = scene.end - scene.start
            if i < len(plan.scenes) - 1:
                dur += XFADE_DURATION
            vf = self._build_scene_filter(scene, asset, dur)
            filter_parts.append(f"[{inp}:v]{vf},fps={VIDEO_FPS}[v{i}]")

        # Crossfade transitions using xfade filter chain
        if len(plan.scenes) == 1:
            # Single scene: no transition needed
            filter_parts.append("[v0]copy[video]")
        else:
            # Build xfade chain
            prev_label = "v0"
            for i in range(1, len(plan.scenes)):
                out_label = f"vt{i-1}"
                # Since we padded previous scenes by XFADE_DURATION, the transition
                # starts exactly at the sum of original scene durations.
                offset = sum(s.end - s.start for s in plan.scenes[:i])
                filter_parts.append(
                    f"[{prev_label}][v{i}]xfade=transition=fade:duration={XFADE_DURATION}:offset={offset:.3f}[{out_label}]"
                )
                prev_label = out_label
            # Final output
            filter_parts.append(f"[{prev_label}]copy[video]")

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
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=RENDER_TIMEOUT)
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

        # For video assets, zoompan/ken_burns don't make sense — force safe motions.
        # Video already has motion; we just want to fit it to 9:16.
        motion = scene.motion
        if asset.type == "video" and motion in ("zoom_in", "zoom_out", "ken_burns"):
            # Force pan for video (safe, just crops over time) or static
            motion = "pan"
            logger.debug(f"Overriding motion '{scene.motion}' to 'pan' for video asset",
                         extra={"asset_type": asset.type, "original_motion": scene.motion})

        # zoompan's 'd' (output frame count) applies PER INPUT FRAME, not once
        # per stream. Both looped images (-loop 1 -t dur) and lavfi color
        # sources produce multiple input frames. Force exactly ONE frame into
        # the motion chain with trim, so zoompan's d/fps controls total
        # output frames/duration correctly.
        needs_trim = motion in ("zoom_in", "zoom_out", "ken_burns")
        trim_prefix = "trim=end_frame=1,setpts=PTS-STARTPTS," if needs_trim else ""

        if scene.visual_type == "text" or scene.visual_type == "graphic":
            return f"{trim_prefix}{base},setsar=1"

        if motion == "static":
            return f"{trim_prefix}{base},setsar=1"
        elif motion == "zoom_in":
            # Progressive zoom from 1.0 to 1.15 over the scene duration.
            return (f"{trim_prefix}{base},zoompan=z='min(1.15,zoom+0.0015)':"
                    f"d={int(dur*VIDEO_FPS)}:s={VIDEO_WIDTH}x{VIDEO_HEIGHT}:fps={VIDEO_FPS},setsar=1")
        elif motion == "zoom_out":
            # Progressive zoom out from 1.15 to 1.0
            return (f"{trim_prefix}{base},zoompan=z='max(1.0,1.15-zoom*0.0015)':"
                    f"d={int(dur*VIDEO_FPS)}:s={VIDEO_WIDTH}x{VIDEO_HEIGHT}:fps={VIDEO_FPS},setsar=1")
        elif motion == "pan":
            # Pan horizontally (for wider source) - simple crop animation over time.
            # Does NOT use zoompan; trim=1 would break the time-based crop.
            return f"{trim_prefix}{base},crop={VIDEO_WIDTH}:{VIDEO_HEIGHT}:{VIDEO_WIDTH}*(1-t/{dur}):0,setsar=1"
        elif motion == "ken_burns":
            # Slow zoom + pan (Ken Burns effect)
            return (f"{trim_prefix}{base},zoompan=z='min(1.1,zoom+0.001)':"
                    f"d={int(dur*VIDEO_FPS)}:s={VIDEO_WIDTH}x{VIDEO_HEIGHT}:fps={VIDEO_FPS}:"
                    f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)',setsar=1")
        else:
            return f"{trim_prefix}{base},setsar=1"

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