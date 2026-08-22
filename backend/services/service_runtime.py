"""Process-level bootstrap for the bundled service.

All effects here are scoped to the service's own process and only run in bundled
mode — they never touch the user's global PATH, shell environment, or an existing
Terminal installation (BR-19, BR-20, BR-22).
"""

from __future__ import annotations

import logging
import os

from backend.services import app_paths, service_config

logger = logging.getLogger(__name__)


def bootstrap() -> None:
    """Redirect the model cache and ffmpeg lookup into the bundle (bundled-only).

    - Sets ``HF_HOME`` to the user-writable models dir so HuggingFace/WhisperX
      caches download there in isolation (BR-12, BR-19, OQ-8).
    - Prepends the bundled ``bin`` dir to this process's PATH so WhisperX's own
      ffmpeg subprocess resolves the vendored binary and it takes precedence
      over any system/Homebrew ffmpeg (BR-20, EC-10).

    No-op in a developer checkout, so existing behavior is unchanged (BR-18).
    """
    if not app_paths.is_bundled():
        return

    cfg = service_config.load()
    os.environ["HF_HOME"] = cfg.models_dir
    logger.info("Bundled service: HF_HOME set to %s", cfg.models_dir)

    bin_dir = app_paths.bundled_bin_dir()
    if bin_dir and bin_dir.exists():
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
        logger.info("Bundled service: prepended %s to process PATH", bin_dir)
