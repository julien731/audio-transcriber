#!/usr/bin/env python3
"""Stub transcription service for the D1 packaging tracer bullet.

This is NOT the real bundled service — it exists only to exercise the app's
child-process plumbing (plan Milestone D1): bind an ephemeral localhost port,
print the stdout `ready` handshake (echoing MT_SERVICE_NONCE), write the same
into service.json, and answer GET /api/health. The real self-contained
PyInstaller service is embedded and smoke-tested in Milestone D0.
"""

from __future__ import annotations

import json
import os
import signal
from http.server import BaseHTTPRequestHandler, HTTPServer

NONCE = os.environ.get("MT_SERVICE_NONCE", "")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        if self.path == "/api/health":
            body = json.dumps(
                {
                    "status": "ready",
                    "provisioning_completed": False,
                    "diarization_available": bool(NONCE),
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args) -> None:  # silence request logging
        return


def app_support_dir() -> str:
    base = os.path.expanduser("~/Library/Application Support/MeetingTranscriber")
    os.makedirs(base, exist_ok=True)
    return base


def main() -> None:
    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]

    # Announce the port two ways, matching the real service contract (OQ-2).
    print(json.dumps({"event": "ready", "port": port, "nonce": NONCE}), flush=True)
    with open(os.path.join(app_support_dir(), "service.json"), "w", encoding="utf-8") as handle:
        json.dump({"port": port, "pid": os.getpid(), "nonce": NONCE}, handle)

    # Clean SIGTERM shutdown so the supervisor's graceful path is exercised.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.server_close()


if __name__ == "__main__":
    main()
