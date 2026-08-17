"""Video slicing + randomized-prefix helpers for the video load test.

This module precomputes a pool of "video specs" so each locust request can send
a different video slice (base64 mode) or a different frame-sampling of the same
remote clip (remote mode). Varying the bytes and/or the sampling knobs defeats
the server-side video chunk cache (keyed on video_bytes + frame_indices +
overrides) so every request forces a real decode -- which is what we need to
sniff out CPU blocking and resource contention in the text-completion container.

It also generates randomized text prefixes that precede the video block so the
prompt prefix never matches across requests, defeating prefix caching and
forcing real tokenization/prefill every time.
"""

import base64
import logging
import os
import random
import shutil
import subprocess
import threading
import urllib.request
import uuid
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

# Common-word corpus for randomized prefixes. Content is irrelevant to the
# model; only token count and uniqueness matter for cache defeat.
_WORD_CORPUS = (
    "the of and to a in is it you that he was for on are with as his they at "
    "be this have from or one had by word but not what all were we when your "
    "can said there use an each which she do how their if will up other about "
    "out many then them these so some her would make like him into time has "
    "look two more write go see number no way could people my than first water "
    "been call who oil its now find long down day did get come made may part "
    "ember falcon granite harbor ivy juniper kaleidoscope lighthouse meadow "
    "nimble orchard parchment quartz ripple sequoia thicket umbrella velvet "
    "willow xenon yonder zephyr avalanche birch canyon driftwood evergreen "
    "fjord geyser hickory indigo jasmine kudzu laurel marsh nautical opal "
    "quicksilver raven sycamore tundra upland vortex whisper yarrow zigzag"
).split()


@dataclass
class VideoSpec:
    # A remote https URL or a base64 data-URI ("data:video/mp4;base64,...").
    url: str
    # Per-request sampling knobs that flow through to the K3 frame sampler.
    # Varying these changes frame_indices and thus the chunk cache hash.
    max_frames: Optional[int] = None
    sample_fps: Optional[float] = None
    spatial_limit: Optional[int] = None

    def to_wire(self) -> dict:
        # Emit the OpenAI-style video_url content block payload.
        block: dict = {"url": self.url}
        if self.max_frames is not None:
            block["max_frames"] = self.max_frames
        if self.sample_fps is not None:
            block["sample_fps"] = self.sample_fps
        if self.spatial_limit is not None:
            block["spatial_limit"] = self.spatial_limit
        return {"type": "video_url", "video_url": block}


@dataclass
class _Range:
    low: float
    high: float

    def sample(self) -> float:
        if self.low == self.high:
            return self.low
        return random.uniform(self.low, self.high)


def _parse_range(s, cast=float):
    # Accept "N" (fixed) or "N:M" (inclusive range).
    if not s:
        return None
    if ":" in s:
        a, b = s.split(":", 1)
        return _Range(cast(a), cast(b))
    v = cast(s)
    return _Range(v, v)


def random_prompt_prefix(target_tokens: int, tokenizer=None, jitter: float = 0.2) -> str:
    # Unique nonce guarantees the first tokens differ every request -> no prefix
    # cache hit. Random words then pad to roughly target_tokens.
    nonce = f"[req-{uuid.uuid4().hex}]"
    words = [nonce]
    if tokenizer is not None:
        text = nonce
        target = max(1, int(target_tokens))
        # Greedily add words until we cross (1 - jitter) * target tokens.
        floor = int(target * (1 - jitter))
        while len(tokenizer.encode(text)) < floor:
            chunk = " ".join(random.choices(_WORD_CORPUS, k=64))
            text = text + " " + chunk
        # Trim back to ~target tokens.
        toks = tokenizer.encode(text)
        if len(toks) > target:
            text = tokenizer.decode(toks[:target])
        return text
    # No tokenizer: approximate 1.3 tokens per word.
    approx_words = max(1, int(target_tokens / 1.3))
    words += random.choices(_WORD_CORPUS, k=approx_words)
    return " ".join(words)


def _ffprobe_duration(path: str) -> float:
    probe = shutil.which("ffprobe")
    if not probe:
        return 10.0
    try:
        out = subprocess.check_output(
            [probe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            stderr=subprocess.STDOUT, text=True, timeout=30,
        ).strip()
        return float(out) if out else 10.0
    except Exception as e:
        logger.warning(f"ffprobe duration failed ({e!r}); assuming 10s")
        return 10.0


def _ffmpeg_slice(src: str, start: float, dur: float, out: str) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    cmd = [ffmpeg, "-y", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
           "-i", src, "-an", "-c:v", "copy", out]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=60)
        return os.path.getsize(out) > 0
    except Exception:
        # Fall back to a re-encode if stream copy produces nothing useful.
        cmd2 = [ffmpeg, "-y", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
                "-i", src, "-an", "-c:v", "libx264", "-preset", "ultrafast",
                "-crf", "28", out]
        try:
            subprocess.run(cmd2, check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=120)
            return os.path.getsize(out) > 0
        except Exception as e:
            logger.warning(f"ffmpeg slice failed ({e!r})")
            return False


def _b64_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        return "data:video/mp4;base64," + base64.b64encode(f.read()).decode("ascii")


def _download(url: str, dest: str, timeout: int = 60) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "fw-loadtest/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
        return os.path.getsize(dest) > 0
    except Exception as e:
        logger.warning(f"video download failed for {url} ({e!r})")
        return False


class VideoSlicer:
    """Precomputes a pool of VideoSpecs to draw from per request."""

    def __init__(self, transport: str, remote_url: Optional[str], fixture_path: Optional[str],
                 workdir: str, slice_count: int, slice_min_secs: float, slice_max_secs: float,
                 max_frames_range, sample_fps_range, spatial_limit_range):
        self.transport = transport
        self.specs: List[VideoSpec] = []
        self._lock = threading.Lock()
        os.makedirs(workdir, exist_ok=True)
        self.workdir = workdir

        if transport == "remote":
            self.specs = self._build_remote(remote_url, slice_count,
                                            max_frames_range, sample_fps_range,
                                            spatial_limit_range)
        else:  # base64
            self.specs = self._build_base64(remote_url, fixture_path, slice_count,
                                            slice_min_secs, slice_max_secs,
                                            max_frames_range, sample_fps_range,
                                            spatial_limit_range)
        if not self.specs:
            raise RuntimeError("VideoSlicer produced no specs; check ffmpeg/fixture/url")
        logger.info(f"VideoSlicer: {len(self.specs)} specs (transport={transport})")

    def _knobs(self, max_frames_range, sample_fps_range, spatial_limit_range):
        mf = int(max_frames_range.sample()) if max_frames_range else None
        sf = round(sample_fps_range.sample(), 2) if sample_fps_range else None
        sl = int(spatial_limit_range.sample()) if spatial_limit_range else None
        return mf, sf, sl

    def _build_remote(self, url, slice_count, mfr, sfr, slr):
        if not url:
            raise ValueError("remote transport requires --prompt-video-url")
        specs = []
        for _ in range(max(1, slice_count)):
            mf, sf, sl = self._knobs(mfr, sfr, slr)
            specs.append(VideoSpec(url=url, max_frames=mf, sample_fps=sf, spatial_limit=sl))
        return specs

    def _source_path(self, remote_url, fixture_path):
        # Prefer an explicit local fixture; otherwise download the remote URL.
        if fixture_path and os.path.isfile(fixture_path):
            return fixture_path
        if remote_url:
            dest = os.path.join(self.workdir, "source.mp4")
            if _download(remote_url, dest):
                return dest
        if fixture_path:
            raise FileNotFoundError(f"video fixture not found: {fixture_path}")
        raise ValueError("base64 transport needs --video-fixture-path or --prompt-video-url")

    def _build_base64(self, remote_url, fixture_path, slice_count,
                      smin, smax, mfr, sfr, slr):
        src = self._source_path(remote_url, fixture_path)
        total = _ffprobe_duration(src)
        specs: List[VideoSpec] = []
        ffmpeg = shutil.which("ffmpeg")
        for i in range(max(1, slice_count)):
            dur = random.uniform(smin, smax) if smax > smin else smax
            dur = min(dur, total)
            start = random.uniform(0, max(0.0, total - dur))
            mf, sf, sl = self._knobs(mfr, sfr, slr)
            out = os.path.join(self.workdir, f"slice_{i}.mp4")
            ok = False
            if ffmpeg:
                ok = _ffmpeg_slice(src, start, dur, out)
            if not ok:
                # Whole-file fallback: use the source bytes as a single spec.
                url = _b64_data_uri(src)
                specs.append(VideoSpec(url=url, max_frames=mf, sample_fps=sf, spatial_limit=sl))
                # Keep at least one spec then stop slicing.
                if not specs[:-1]:
                    continue
            else:
                specs.append(VideoSpec(url=_b64_data_uri(out),
                                       max_frames=mf, sample_fps=sf, spatial_limit=sl))
        if not specs:
            # Last-resort: base64 the whole source.
            url = _b64_data_uri(src)
            mf, sf, sl = self._knobs(mfr, sfr, slr)
            specs.append(VideoSpec(url=url, max_frames=mf, sample_fps=sf, spatial_limit=sl))
        return specs

    def sample(self) -> VideoSpec:
        with self._lock:
            return random.choice(self.specs)


class VideoSlicerHolder:
    # Singleton builder so the (expensive) slicing happens once per run, not per user.
    _instance: Optional[VideoSlicer] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, options) -> VideoSlicer:
        if cls._instance is not None:
            return cls._instance
        with cls._lock:
            if cls._instance is None:
                transport = getattr(options, "video_transport", "remote")
                remote_url = getattr(options, "prompt_video_url", None)
                fixture = getattr(options, "video_fixture_path", None)
                workdir = getattr(options, "video_workdir", None) or "/tmp/fw_video_loadtest"
                slice_count = int(getattr(options, "video_slice_count", 8))
                smin = float(getattr(options, "video_slice_min_secs", 2.0))
                smax = float(getattr(options, "video_slice_max_secs", 5.0))
                mfr = _parse_range(getattr(options, "video_max_frames", None), int)
                sfr = _parse_range(getattr(options, "video_sample_fps", None), float)
                slr = _parse_range(getattr(options, "video_spatial_limit", None), int)
                cls._instance = VideoSlicer(
                    transport, remote_url, fixture, workdir,
                    slice_count, smin, smax, mfr, sfr, slr,
                )
            return cls._instance

    @classmethod
    def reset(cls):
        with cls._lock:
            cls._instance = None
