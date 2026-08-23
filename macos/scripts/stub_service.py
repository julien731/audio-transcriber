#!/usr/bin/env python3
"""Stub transcription service for the D1 packaging tracer bullet + manual testing.

This is NOT the real bundled service — it exists only to exercise the app's
child-process plumbing and UI flows without the multi-GB ML stack. It:
  - binds an ephemeral localhost port, prints the stdout `ready` handshake
    (echoing MT_SERVICE_NONCE), and writes service.json;
  - answers GET /api/health;
  - simulates first-run provisioning (token + a fast fake model download) so the
    native setup wizard can be walked end to end.
The real self-contained PyInstaller service is embedded/smoke-tested at Milestone D0.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

NONCE = os.environ.get("MT_SERVICE_NONCE", "")

# In-memory provisioning state (mirrors ProvisioningStatus).
_state = {
    "provisioning_completed": False,
    "models_present": False,
    "whisper_model": "large-v3",
    "diarization_available": False,
    "download_state": "idle",
    "download_progress": 0,
    "download_error": None,
}
_lock = threading.Lock()


def _simulate_download() -> None:
    with _lock:
        _state.update(download_state="downloading", download_progress=0, download_error=None)
    for pct in range(0, 101, 20):
        time.sleep(0.5)
        with _lock:
            _state["download_progress"] = pct
    with _lock:
        _state.update(download_state="completed", provisioning_completed=True, models_present=True)


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/health":
            with _lock:
                self._send(
                    200,
                    {
                        "status": "ready",
                        "provisioning_completed": _state["provisioning_completed"],
                        "diarization_available": _state["diarization_available"],
                    },
                )
        elif self.path == "/api/provisioning":
            with _lock:
                self._send(200, dict(_state))
        else:
            self._send(404, {"detail": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        if self.path == "/api/provisioning/token":
            try:
                token = json.loads(raw).get("hf_token", "")
            except json.JSONDecodeError:
                token = ""
            with _lock:
                _state["diarization_available"] = bool(token)
                self._send(200, dict(_state))
        elif self.path == "/api/provisioning/models":
            threading.Thread(target=_simulate_download, daemon=True).start()
            time.sleep(0.05)
            with _lock:
                self._send(200, dict(_state))
        else:
            self._send(404, {"detail": "Not found"})

    def log_message(self, *args) -> None:
        return


def app_support_dir() -> str:
    base = os.path.expanduser("~/Library/Application Support/MeetingTranscriber")
    os.makedirs(base, exist_ok=True)
    return base


def main() -> None:
    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]

    print(json.dumps({"event": "ready", "port": port, "nonce": NONCE}), flush=True)
    with open(os.path.join(app_support_dir(), "service.json"), "w", encoding="utf-8") as handle:
        json.dump({"port": port, "pid": os.getpid(), "nonce": NONCE}, handle)

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.server_close()


if __name__ == "__main__":
    main()
