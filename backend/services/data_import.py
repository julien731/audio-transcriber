"""First-run import of a prior Terminal-installation's meetings (BR-13, BR-14).

Import is non-destructive (copy, never move) and idempotent — existing meetings
in the destination are never overwritten. Runs only in bundled mode; a developer
checkout keeps using its own ``./data`` directory unchanged (BR-18).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from backend.services import app_paths, service_config

logger = logging.getLogger(__name__)


def legacy_candidates() -> list[Path]:
    """Candidate locations of a prior Terminal-installation data directory (OQ-4).

    The old install stored meetings under ``<checkout>/data``. Since a bundled
    service has no checkout, probe a few conventional locations under $HOME. This
    list is intentionally small and easy to extend.
    """
    home = Path.home()
    return [
        home / "audio-transcriber" / "data",
        home / "meeting-transcriber" / "data",
        home / "Developer" / "audio-transcriber" / "data",
        home / "Projects" / "audio-transcriber" / "data",
    ]


def _has_meetings(data_dir: Path) -> bool:
    meetings = data_dir / "meetings"
    return meetings.is_dir() and any(meetings.glob("*/metadata.json"))


def find_legacy_data_dir(candidates: list[Path] | None = None, *, exclude: Path | None = None) -> Path | None:
    """Return the first candidate data dir that holds meetings, or None (BR-15)."""
    for candidate in candidates if candidates is not None else legacy_candidates():
        if exclude is not None and candidate.resolve() == exclude.resolve():
            continue
        if _has_meetings(candidate):
            return candidate
    return None


def import_meetings(src_data_dir: Path, dest_data_dir: Path) -> int:
    """Copy meetings from ``src_data_dir`` into ``dest_data_dir`` non-destructively.

    Existing meeting directories in the destination are left untouched. Returns
    the number of meetings copied.
    """
    src_meetings = src_data_dir / "meetings"
    dest_meetings = dest_data_dir / "meetings"
    dest_meetings.mkdir(parents=True, exist_ok=True)

    copied = 0
    for meeting_dir in sorted(src_meetings.glob("*/")):
        if not (meeting_dir / "metadata.json").exists():
            continue
        target = dest_meetings / meeting_dir.name
        if target.exists():
            continue  # idempotent: never overwrite existing history
        shutil.copytree(meeting_dir, target)
        copied += 1
    return copied


def run_first_run_import() -> None:
    """Detect and import a prior installation's meetings once (bundled-only).

    A missing prior directory is not an error — first run simply proceeds with an
    empty library (BR-15, EC-7).
    """
    if not app_paths.is_bundled():
        return

    cfg = service_config.load()
    if cfg.imported_from:
        return  # already imported on a previous run

    dest = Path(cfg.data_dir)
    legacy = find_legacy_data_dir(exclude=dest)
    if legacy is None:
        logger.info("First run: no prior data directory found; starting with an empty library")
        return

    count = import_meetings(legacy, dest)
    cfg.imported_from = str(legacy)
    service_config.save(cfg)
    logger.info("First run: imported %d meeting(s) from %s (original left intact)", count, legacy)
