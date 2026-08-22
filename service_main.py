"""Bundled-service entrypoint (BR-4, BR-5, BR-6, EC-1, EC-8).

Binds an ephemeral localhost port, announces it to the client via both a written
``service.json`` and a stdout JSON handshake (OQ-2), then runs the FastAPI app
with auto-reload disabled. The dev entrypoint (``run.py``) is unchanged.

The client learns *where* to reach the service from the handshake/port file, and
*when* it is ready by polling ``GET /api/health`` (BR-5).
"""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path

from backend.services import app_paths

HOST = "127.0.0.1"
SERVICE_FILE = "service.json"
# Launch-coordination nonce set by the spawning app (native macOS app, plan
# Artifact A). It lets the parent reject a stale/foreign service.json after PID
# reuse. This is coordination, not configuration, so it is exempt in spirit from
# the BR-21 "ignore ambient env" rule; when unset the service starts normally and
# omits the nonce (dev run.py path, any non-app client).
NONCE_ENV = "MT_SERVICE_NONCE"


def service_nonce() -> str:
    return os.environ.get(NONCE_ENV, "")


def bind_free_socket(host: str = HOST, retries: int = 5) -> socket.socket:
    """Bind and return a socket on an OS-assigned free port.

    Holding the bound socket (rather than probing then re-binding) avoids a
    TOCTOU race. Retries defensively; raises OSError if it cannot bind (EC-1).
    """
    last_error: OSError | None = None
    for _ in range(retries):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, 0))
            return sock
        except OSError as exc:
            last_error = exc
            sock.close()
    raise OSError(f"Could not bind an ephemeral port: {last_error}")


def service_file_path() -> Path:
    return app_paths.app_support_dir() / SERVICE_FILE


def write_service_file(port: int, pid: int | None = None, nonce: str = "") -> Path:
    """Persist the chosen port and pid for clients that discover via file.

    The launch nonce is included only when set, so an empty nonce keeps the file
    byte-identical to the pre-nonce format (backward compatible).
    """
    path = service_file_path()
    payload: dict[str, object] = {"port": port, "pid": os.getpid() if pid is None else pid}
    if nonce:
        payload["nonce"] = nonce
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def handshake_line(port: int, nonce: str = "") -> str:
    """Machine-readable stdout line announcing the service's port (and nonce)."""
    payload: dict[str, object] = {"event": "ready", "port": port}
    if nonce:
        payload["nonce"] = nonce
    return json.dumps(payload)


def _emit_error(message: str) -> None:
    print(json.dumps({"event": "error", "message": message}), file=sys.stderr, flush=True)


def main() -> int:
    try:
        sock = bind_free_socket()
    except OSError as exc:
        _emit_error(str(exc))  # machine-readable error the client can surface (EC-1)
        return 1

    port = sock.getsockname()[1]
    nonce = service_nonce()
    write_service_file(port, nonce=nonce)
    print(handshake_line(port, nonce=nonce), flush=True)

    import uvicorn

    config = uvicorn.Config("backend.main:app", reload=False, log_level="info")
    server = uvicorn.Server(config)
    server.run(sockets=[sock])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
