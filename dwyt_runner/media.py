"""Video metadata and audio: yt-dlp for the URL, ffmpeg for a small mono FLAC."""
from __future__ import annotations

import glob
import hashlib
import os
import re
import subprocess
from typing import Any, Callable

from . import tools
from .config import Config
from .errors import WorkError, from_ytdlp
from .i18n import gettext as _


def video_meta(info: dict[str, Any]) -> dict[str, Any]:
    published = info.get("upload_date")
    if published and len(published) == 8:
        published = f"{published[:4]}-{published[4:6]}-{published[6:]}"
    from .chapters import from_ytdlp
    meta = {"title": info.get("title"), "channel": info.get("uploader") or info.get("channel"),
            "duration_s": int(info.get("duration") or 0) or None, "published": published}
    chapters = from_ytdlp(info)  # the uploader's own chapters (decision 19.21)
    if chapters:
        meta["chapters"] = chapters
    return meta


def probe(url: str, cfg: Config) -> dict[str, Any]:
    """Metadata without downloading; refuses playlists, live streams and too long videos."""
    import yt_dlp
    try:
        with yt_dlp.YoutubeDL(tools.ytdlp_options(cfg.cookies_file)) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:  # yt-dlp raises many types
        raise from_ytdlp(exc, _("beolvasni a videót"), url) from exc
    check(info, cfg)
    return info


def check(info: dict[str, Any], cfg: Config) -> None:
    if info.get("_type") == "playlist":
        raise WorkError(_("Ez egy lejátszási lista; egyetlen videó URL-je kell."))
    if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
        raise WorkError(_("Élő adás vagy premier; a vége után dolgozható fel."), "later")
    duration = info.get("duration") or 0
    if duration and duration > cfg.max_duration_min * 60:
        raise WorkError(_("A videó {int} perces, hosszabb a beállított {max_duration_min} percnél "
                           "(max_duration_min).").format(
                          int=int(duration) // 60, max_duration_min=cfg.max_duration_min))


def download_audio(url: str, cfg: Config, outdir: str) -> str:
    import yt_dlp
    try:
        with yt_dlp.YoutubeDL(tools.ytdlp_options(cfg.cookies_file, outdir)) as ydl:
            ydl.extract_info(url, download=True)
    except Exception as exc:
        raise from_ytdlp(exc, _("letölteni a hangot"), url) from exc
    files = [f for f in glob.glob(os.path.join(outdir, "audio.*")) if not f.endswith(".part")]
    if not files:
        raise WorkError(_("A yt-dlp nem hozott létre hangfájlt."))
    return files[0]


def download_video(url: str, cfg: Config, outdir: str) -> str:
    """The picture for the slides (plan 19c round C): video only, at most 720p (the text stays readable)."""
    import yt_dlp
    opts = {**tools.ytdlp_options(cfg.cookies_file), "format": "bv*[height<=720]/b[height<=720]/bv*/b",
            "outtmpl": os.path.join(outdir, "video.%(ext)s")}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.extract_info(url, download=True)
    except Exception as exc:
        raise from_ytdlp(exc, _("letölteni a videót"), url) from exc
    files = [f for f in glob.glob(os.path.join(outdir, "video.*")) if not f.endswith(".part")]
    if not files:
        raise WorkError(_("A yt-dlp nem hozott létre videófájlt."))
    return files[0]


def to_flac(src: str, workdir: str) -> str:
    """Mono 16 kHz FLAC: lossless for speech models and a fraction of a video's size."""
    dst = os.path.join(workdir, "audio.flac")
    try:
        subprocess.run([tools.ffmpeg(), "-nostdin", "-loglevel", "error", "-y", "-i", src, "-vn",
                        "-ac", "1", "-ar", "16000", "-c:a", "flac", dst],
                       check=True, capture_output=True, timeout=3600, **tools.no_window())
    except subprocess.CalledProcessError as exc:
        raise WorkError(_("Az ffmpeg nem tudta kinyerni a hangot: ")
                        + exc.stderr.decode("utf-8", "replace")[-300:]) from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WorkError(_("Az ffmpeg nem indult el vagy nem végzett: {exc}").format(exc=exc)) from exc
    return dst


# ---- where the media comes from -------------------------------------------------------------

class Fetch:
    """A media source for the transcription modes: info() (title, duration ...), get(workdir) -> path."""

    def info(self) -> dict[str, Any]:
        raise NotImplementedError

    def get(self, workdir: str) -> str:
        raise NotImplementedError


class from_url(Fetch):  # noqa: N801 - reads like a function at the call site
    def __init__(self, url: str, cfg: Config) -> None:
        self.url, self.cfg, self._info = url, cfg, None

    def info(self) -> dict[str, Any]:
        if self._info is None:
            self._info = probe(self.url, self.cfg)
        return self._info

    def get(self, workdir: str) -> str:
        return download_audio(self.url, self.cfg, workdir)


class from_file(Fetch):  # noqa: N801
    """An own recording: obtain(workdir) returns its path (downloaded from the central service,
    or the file in the watched folder); its SHA-256 must match what the central service named."""

    def __init__(self, name: str, sha256: str, obtain: Callable[[str], str],
                 options: dict[str, Any] | None = None) -> None:
        self.name, self.sha256, self.obtain = name, sha256, obtain
        self.options = options or {}
        self.path: str | None = None
        self.tracks = None  # tracks.Tracks, read by get()
        self.audio_used: int | None = None
        self.subtitle_used: int | None = None

    def info(self) -> dict[str, Any]:
        return {"title": self.name, "duration": duration(self.path) if self.path else None}

    def verified(self, workdir: str) -> str:
        """The file itself (checked), with its tracks read; no audio track chosen yet."""
        from . import tracks
        path = self.obtain(workdir)
        if file_sha256(path) != self.sha256:
            raise WorkError(_("A fájl tartalma nem egyezik a beküldöttel (megváltozott vagy sérült)."))
        self.path = path
        self.tracks = tracks.read(path)
        return path

    def get(self, workdir: str) -> str:
        """The audio to transcribe: the file, or its chosen audio track alone when it has several (decision 19.12)."""
        from . import tracks
        path = self.verified(workdir)
        self.audio_used = tracks.choose_audio(self.tracks, self.options)
        if self.audio_used is not None and len(self.tracks.audio) > 1:
            return tracks.audio_only(path, self.audio_used, workdir)
        return path

    def report(self) -> dict[str, Any] | None:
        if self.tracks is None:
            return None
        return self.tracks.report(self.audio_used, self.subtitle_used)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_DURATION = re.compile(r"Duration:\s*(\d+):(\d{2}):(\d{2}(?:\.\d+)?)")


def duration(path: str) -> float | None:
    """Seconds of a media file, read from ffmpeg's banner (no ffprobe in imageio-ffmpeg)."""
    try:
        proc = subprocess.run([tools.ffmpeg(), "-nostdin", "-hide_banner", "-i", path], capture_output=True,
                              timeout=60, check=False, **tools.no_window())
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = _DURATION.search(proc.stderr.decode("utf-8", "replace"))
    if not m:
        return None
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
