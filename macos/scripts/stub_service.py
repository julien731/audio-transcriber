#!/usr/bin/env python3
"""Stub transcription service for the D1 packaging tracer bullet + manual testing.

This is NOT the real bundled service — it exists only to exercise the app's
child-process plumbing and UI flows without the multi-GB ML stack. It:
  - binds an ephemeral localhost port, prints the stdout `ready` handshake
    (echoing MT_SERVICE_NONCE), and writes service.json;
  - answers GET /api/health;
  - simulates first-run provisioning (token + a fast fake model download) so the
    native setup wizard can be walked end to end;
  - serves one canned READY meeting (list + detail + transcript + job + silent
    audio) so the meeting UI can be worked on in dev. Mutating endpoints
    (create/patch/rename/delete/retry/cancel) are stateless no-ops over that
    single fixture — enough to click through the detail view without 500s.
The real self-contained PyInstaller service is embedded/smoke-tested at Milestone D0.
"""

from __future__ import annotations

import io
import json
import os
import signal
import socketserver
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlsplit

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

# --- Canned meeting fixture ------------------------------------------------
# One READY meeting is enough to render the list, detail tabs, transcript
# viewer, speaker editor, and analysis views. All ids reference DEMO_ID so a
# meeting created via POST /api/meetings navigates straight to this fixture,
# and the /{id} routes ignore the path id and always serve it. Shapes are the
# wire (snake_case) form the Swift decoder expects — see
# Sources/MeetingTranscriberKitTests/ModelDecodingTests.swift.
DEMO_ID = "demo-1"
DEMO_DURATION = 95.0

_SEGMENTS = [
    (0.0, 8.0, "SPEAKER_00", "Thanks everyone for joining. Let's start with the roadmap."),
    (8.0, 19.0, "SPEAKER_01", "Sure. We shipped the upload redesign last week and it's testing well."),
    (19.0, 34.0, "SPEAKER_00", "Great. What's the status on the transcription pipeline?"),
    (34.0, 52.0, "SPEAKER_01", "Still integrating diarization. I expect it ready by the end of the sprint."),
    (52.0, 73.0, "SPEAKER_00", "Let's make sure we have fixtures so the team can work on the UI in parallel."),
    (73.0, 95.0, "SPEAKER_01", "Agreed. I'll set up a stub backend with canned meetings today."),
]

DETAIL = {
    "metadata": {
        "id": DEMO_ID,
        "title": "Product Strategy Sync",
        "type": "other",
        "created_at": "2026-08-22T10:31:59",
        "duration_seconds": DEMO_DURATION,
        "audio_filename": "demo.wav",
        "status": "ready",
        "language": "en",
        "expected_languages": ["en"],
        "num_speakers": 2,
        "preprocess_audio": True,
        "audio_analysis_enabled": False,
        "audio_analysis_status": None,
        "job_id": DEMO_ID,
        "speakers": {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"},
        "error": None,
        "context": "Quarterly planning",
    },
    "transcript": {
        "segments": [
            {
                "id": f"s{i + 1}",
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": text,
                "language": "en",
            }
            for i, (start, end, speaker, text) in enumerate(_SEGMENTS)
        ],
        "language": "en",
    },
    "audio_analysis": None,
}

SUMMARY = [
    {
        "id": DEMO_ID,
        "title": DETAIL["metadata"]["title"],
        "type": DETAIL["metadata"]["type"],
        "created_at": DETAIL["metadata"]["created_at"],
        "duration_seconds": DEMO_DURATION,
        "status": "ready",
    }
]

JOB = {
    "id": DEMO_ID,
    "meeting_id": DEMO_ID,
    "status": "completed",
    "progress": 100,
    "stage": "completed",
    "error": None,
    "created_at": "2026-08-22T10:31:59",
    "updated_at": "2026-08-22T10:33:34",
}

START_RESPONSE = {"meeting_id": DEMO_ID, "job_id": DEMO_ID}

ANALYSIS_PROMPT = {
    "prompt": (
        "You are analyzing a meeting transcript. Summarize the key decisions, "
        "action items, and open questions from the conversation below.\n\n"
        + "\n".join(f"{speaker}: {text}" for _, _, speaker, text in _SEGMENTS)
    )
}


def _silent_wav(duration_seconds: float) -> bytes:
    """A mono 16-bit 8 kHz silent WAV so the detail view's AVPlayer has a real,
    seekable asset whose length matches the transcript timeline."""
    frame_rate = 8000
    frame_count = int(duration_seconds * frame_rate)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(frame_rate)
        handle.writeframes(b"\x00\x00" * frame_count)
    return buffer.getvalue()


DEMO_AUDIO = _silent_wav(DEMO_DURATION)


def _simulate_download() -> None:
    with _lock:
        _state.update(download_state="downloading", download_progress=0, download_error=None)
    for pct in range(0, 101, 20):
        time.sleep(0.5)
        with _lock:
            _state["download_progress"] = pct
    with _lock:
        _state.update(download_state="completed", provisioning_completed=True, models_present=True)


def _parse_json(raw: bytes) -> dict:
    try:
        parsed = json.loads(raw or b"{}")
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _apply_meeting_update(raw: bytes) -> None:
    """Persist a PATCH /api/meetings/{id} into the in-memory fixture so speaker
    renames (all-segments scope) and title/type/context edits survive the GET
    that follows. Mirrors backend.routers.meetings.update_meeting. Caller holds
    _lock."""
    body = _parse_json(raw)
    metadata = DETAIL["metadata"]
    for field in ("title", "type", "context"):
        if body.get(field) is not None:
            metadata[field] = body[field]
    if body.get("speakers") is not None:
        metadata["speakers"] = body["speakers"]


def _apply_segment_speaker(raw: bytes) -> None:
    """Persist a PATCH /api/meetings/{id}/segments/speaker (single-segment scope).
    Mirrors backend.routers.meetings.update_segment_speaker: reuse an existing
    speaker id when the name already maps, else mint a per-segment id. Caller
    holds _lock."""
    body = _parse_json(raw)
    segment_id = body.get("segment_id")
    new_name = (body.get("speaker_name") or "").strip()
    if not segment_id or not new_name:
        return
    segment = next((s for s in DETAIL["transcript"]["segments"] if s["id"] == segment_id), None)
    if segment is None:
        return
    speakers = DETAIL["metadata"]["speakers"]
    normalized = new_name.casefold()
    existing_id = next(
        (sid for sid, name in speakers.items() if name.strip().casefold() == normalized),
        None,
    )
    if existing_id is not None:
        segment["speaker"] = existing_id
        return
    new_speaker_id = f"{segment['speaker']}_seg_{segment_id}"
    segment["speaker"] = new_speaker_id
    speakers[new_speaker_id] = new_name


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: dict | list) -> None:
        """JSON responder."""
        body = json.dumps(payload).encode("utf-8")
        self._send_bytes(status, body, "application/json")

    def _send_bytes(self, status: int, body: bytes, content_type: str, extra_headers: dict | None = None) -> None:
        """Raw/binary responder (audio, empty 200s) — keeps _send JSON-only."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _drain_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    def _send_audio(self) -> None:
        """Serve DEMO_AUDIO with minimal single-range support (AVPlayer sends
        Range and won't seek reliably against a plain 200)."""
        total = len(DEMO_AUDIO)
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            spec = range_header[len("bytes=") :].split(",", 1)[0]
            start_str, _, end_str = spec.partition("-")
            try:
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else total - 1
            except ValueError:
                start, end = 0, total - 1
            start = max(0, start)
            end = min(end, total - 1)
            if start > end:
                start, end = 0, total - 1
            chunk = DEMO_AUDIO[start : end + 1]
            self._send_bytes(
                206,
                chunk,
                "audio/wav",
                {
                    "Accept-Ranges": "bytes",
                    "Content-Range": f"bytes {start}-{end}/{total}",
                },
            )
        else:
            self._send_bytes(200, DEMO_AUDIO, "audio/wav", {"Accept-Ranges": "bytes"})

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/api/health":
            with _lock:
                self._send(
                    200,
                    {
                        "status": "ready",
                        "provisioning_completed": _state["provisioning_completed"],
                        "diarization_available": _state["diarization_available"],
                    },
                )
        elif path == "/api/provisioning":
            with _lock:
                self._send(200, dict(_state))
        elif path == "/api/meetings":
            self._send(200, SUMMARY)
        elif path.startswith("/api/meetings/") and path.endswith("/audio"):
            self._send_audio()
        elif path.startswith("/api/meetings/") and path.endswith("/analysis-prompt"):
            self._send(200, ANALYSIS_PROMPT)
        elif path.startswith("/api/jobs/"):
            self._send(200, JOB)
        elif path.startswith("/api/meetings/"):
            self._send(200, DETAIL)
        else:
            self._send(404, {"detail": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        raw = self._drain_body()
        path = urlsplit(self.path).path
        if path == "/api/provisioning/token":
            try:
                token = json.loads(raw or b"{}").get("hf_token", "")
            except json.JSONDecodeError:
                token = ""
            with _lock:
                _state["diarization_available"] = bool(token)
                self._send(200, dict(_state))
        elif path == "/api/provisioning/models":
            threading.Thread(target=_simulate_download, daemon=True).start()
            time.sleep(0.05)
            with _lock:
                self._send(200, dict(_state))
        elif path == "/api/meetings":
            self._send(200, START_RESPONSE)
        elif path.startswith("/api/meetings/") and path.endswith("/retry"):
            self._send(200, START_RESPONSE)
        elif path.startswith("/api/meetings/") and path.endswith("/cancel"):
            self._send(200, {})
        else:
            self._send(404, {"detail": "Not found"})

    def do_PATCH(self) -> None:  # noqa: N802
        raw = self._drain_body()
        path = urlsplit(self.path).path
        if path.startswith("/api/meetings/") and path.endswith("/segments/speaker"):
            with _lock:
                _apply_segment_speaker(raw)
                self._send(200, {})
        elif path.startswith("/api/meetings/"):
            with _lock:
                _apply_meeting_update(raw)
                self._send(200, DETAIL["metadata"])
        else:
            self._send(404, {"detail": "Not found"})

    def do_DELETE(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path.startswith("/api/meetings/"):
            self._send(200, {})
        else:
            self._send(404, {"detail": "Not found"})

    def log_message(self, *args) -> None:
        return


class _StubServer(HTTPServer):
    """HTTPServer that skips the reverse-DNS lookup in the default server_bind.

    http.server.HTTPServer.server_bind() calls socket.getfqdn(host), which does
    a reverse-DNS resolution. On CI runners with slow or absent reverse DNS this
    blocks for ~20s during construction — delaying the stdout readiness handshake
    past the supervisor's timeout and failing the integration suite. We don't use
    server_name, so bind the socket and set it to the bound host directly.
    """

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        self.server_name = self.server_address[0]
        self.server_port = self.server_address[1]


def app_support_dir() -> str:
    base = os.path.expanduser("~/Library/Application Support/MeetingTranscriber")
    os.makedirs(base, exist_ok=True)
    return base


def main() -> None:
    httpd = _StubServer(("127.0.0.1", 0), Handler)
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
