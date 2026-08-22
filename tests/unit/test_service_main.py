from __future__ import annotations

import json
import socket

import pytest

import service_main


class TestBindFreeSocket:
    def test_returns_bound_socket_on_free_port(self):
        sock = service_main.bind_free_socket()
        try:
            host, port = sock.getsockname()
            assert host == service_main.HOST
            assert port > 0
        finally:
            sock.close()

    def test_raises_after_exhausting_retries(self, monkeypatch):
        class _FakeSock:
            def setsockopt(self, *a):
                pass

            def bind(self, *a):
                raise OSError("address in use")

            def close(self):
                pass

        monkeypatch.setattr(service_main.socket, "socket", lambda *a, **k: _FakeSock())
        with pytest.raises(OSError, match="Could not bind"):
            service_main.bind_free_socket(retries=3)


class TestServiceFileAndHandshake:
    def test_write_service_file(self):
        path = service_main.write_service_file(54321, pid=999)
        data = json.loads(path.read_text())
        assert data == {"port": 54321, "pid": 999}

    def test_handshake_line_is_machine_readable(self):
        line = service_main.handshake_line(12345)
        assert json.loads(line) == {"event": "ready", "port": 12345}


class TestMainErrorPath:
    def test_returns_1_and_emits_error_when_bind_fails(self, monkeypatch, capsys):
        def _fail(*a, **k):
            raise OSError("no ports")

        monkeypatch.setattr(service_main, "bind_free_socket", _fail)
        rc = service_main.main()
        assert rc == 1
        err = json.loads(capsys.readouterr().err.strip())
        assert err["event"] == "error"

    def test_main_runs_server_with_bound_socket(self, monkeypatch, capsys):
        # Use a real bound socket so a port can be read back, but stub uvicorn.
        real = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        real.bind((service_main.HOST, 0))
        monkeypatch.setattr(service_main, "bind_free_socket", lambda *a, **k: real)

        captured = {}

        class _FakeServer:
            def __init__(self, config):
                captured["config"] = config

            def run(self, sockets=None):
                captured["sockets"] = sockets

        fake_uvicorn = type("U", (), {"Config": lambda *a, **k: dict(kwargs=k), "Server": _FakeServer})
        monkeypatch.setitem(__import__("sys").modules, "uvicorn", fake_uvicorn)

        try:
            rc = service_main.main()
        finally:
            real.close()

        assert rc == 0
        assert captured["sockets"] == [real]
        # Handshake announced a real port.
        out = json.loads(capsys.readouterr().out.strip())
        assert out["event"] == "ready"
        assert out["port"] > 0
        assert captured["config"]["kwargs"]["reload"] is False
