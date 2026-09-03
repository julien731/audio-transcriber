"""Filesystem locations and bundle-awareness for the local service.

This module is the single place that knows *where things live* and whether the
process is running as a bundled, self-contained service (PyInstaller) or from a
developer checkout. It must never import ``config`` — ``config`` depends on this
module, so a reverse dependency would create an import cycle.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Tests set this to redirect the Application Support location to a tmp dir,
# mirroring the existing pattern of patching ``config.DATA_DIR``. Ambient
# environment variables are intentionally NOT consulted (BR-21).
_APP_SUPPORT_OVERRIDE: Path | None = None

APP_NAME = "MeetingTranscriber"


def is_bundled() -> bool:
    """True when running as a PyInstaller-frozen, self-contained service."""
    return bool(getattr(sys, "frozen", False))


def bundle_dir() -> Path | None:
    """Root of the unpacked bundle (PyInstaller ``_MEIPASS``), or None in dev."""
    meipass = getattr(sys, "_MEIPASS", None)
    return Path(meipass) if meipass else None


def bundled_bin_dir() -> Path | None:
    """Directory holding vendored native binaries (ffmpeg) inside the bundle."""
    root = bundle_dir()
    return root / "bin" if root else None


def app_support_dir() -> Path:
    """User-writable base directory for all service data (BR-12).

    Defaults to ``~/Library/Application Support/MeetingTranscriber`` on macOS and
    an XDG-style location elsewhere (so tests/CI on Linux still work). The
    directory is created on access.
    """
    if _APP_SUPPORT_OVERRIDE is not None:
        base = _APP_SUPPORT_OVERRIDE
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / APP_NAME
    else:
        base = Path.home() / ".local" / "share" / APP_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base


def logs_dir() -> Path:
    """User-writable directory for service + app log files (story #137).

    Lives under the Application Support base so it is redirected by the same
    ``_APP_SUPPORT_OVERRIDE`` test hook (and the ``_isolate_app_support``
    fixture), keeping tests off the real home directory. Created on access.
    """
    directory = app_support_dir() / "Logs"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ffmpeg_path() -> str | None:
    """Resolve the ffmpeg executable, preferring the bundled binary (BR-3, BR-20).

    Bundled: the vendored, arm64-native static binary shipped in the bundle.
    Dev: whatever ffmpeg is on PATH. Returns None if none is found.
    """
    bin_dir = bundled_bin_dir()
    if bin_dir is not None:
        candidate = bin_dir / "ffmpeg"
        if candidate.exists():
            return str(candidate)
    return shutil.which("ffmpeg")
