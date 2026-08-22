from __future__ import annotations

import json
import shutil
import struct
import wave
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from backend.services.job_queue import job_queue

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _create_silence_wav(path: Path) -> None:
    """Create a minimal valid WAV file (~76 bytes, 1ms silence)."""
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(struct.pack("<" + "h" * 16, *([0] * 16)))


@pytest.fixture(autouse=True)
def _clean_job_queue():
    """Clear the job_queue singleton after each test to prevent state leakage."""
    yield
    job_queue.clear()


@pytest.fixture(autouse=True)
def _isolate_app_support(tmp_path_factory, monkeypatch):
    """Redirect the Application Support location to a tmp dir for every test, so
    service config / provisioning never read or write the real home directory."""
    from backend.services import app_paths

    base = tmp_path_factory.mktemp("app_support")
    monkeypatch.setattr(app_paths, "_APP_SUPPORT_OVERRIDE", base)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Isolated data directory for a single test."""
    meetings_dir = tmp_path / "meetings"
    meetings_dir.mkdir()
    return tmp_path


@pytest.fixture
def meetings_dir(data_dir: Path) -> Path:
    return data_dir / "meetings"


@pytest.fixture
async def client(data_dir: Path) -> AsyncClient:
    """Async test client with DATA_DIR patched to tmp_path."""
    import config as config_module
    from backend.main import app

    original_data_dir = config_module.DATA_DIR
    original_meetings_dir = config_module.MEETINGS_DIR

    config_module.DATA_DIR = data_dir
    config_module.MEETINGS_DIR = data_dir / "meetings"

    # Also patch the routers module that may have imported these
    import backend.routers.analysis as analysis_module
    import backend.routers.meetings as meetings_module

    original_router_meetings_dir = getattr(meetings_module, "MEETINGS_DIR", None)
    meetings_module.MEETINGS_DIR = data_dir / "meetings"

    original_analysis_meetings_dir = getattr(analysis_module, "MEETINGS_DIR", None)
    analysis_module.MEETINGS_DIR = data_dir / "meetings"

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    config_module.DATA_DIR = original_data_dir
    config_module.MEETINGS_DIR = original_meetings_dir
    if original_router_meetings_dir is not None:
        meetings_module.MEETINGS_DIR = original_router_meetings_dir
    if original_analysis_meetings_dir is not None:
        analysis_module.MEETINGS_DIR = original_analysis_meetings_dir


@pytest.fixture
def sample_audio(tmp_path: Path) -> Path:
    """Generate a minimal WAV file in tmp_path."""
    wav_path = tmp_path / "sample.wav"
    _create_silence_wav(wav_path)
    return wav_path


@pytest.fixture
def sample_metadata() -> dict:
    """READY-state metadata as a dict."""
    return json.loads((FIXTURES_DIR / "metadata.json").read_text())


@pytest.fixture
def sample_metadata_processing() -> dict:
    """PROCESSING-state metadata as a dict."""
    return json.loads((FIXTURES_DIR / "metadata_processing.json").read_text())


@pytest.fixture
def sample_metadata_error() -> dict:
    """ERROR-state metadata as a dict."""
    return json.loads((FIXTURES_DIR / "metadata_error.json").read_text())


@pytest.fixture
def sample_transcript() -> dict:
    """Transcript with 4 segments and 2 speakers."""
    return json.loads((FIXTURES_DIR / "transcript.json").read_text())


@pytest.fixture
def populated_meeting(meetings_dir: Path, sample_metadata: dict, sample_transcript: dict, sample_audio: Path) -> str:
    """Create a fully populated meeting on disk. Returns the meeting ID."""
    meeting_id = sample_metadata["id"]
    meeting_dir = meetings_dir / meeting_id
    meeting_dir.mkdir()

    (meeting_dir / "metadata.json").write_text(json.dumps(sample_metadata))
    (meeting_dir / "transcript.json").write_text(json.dumps(sample_transcript))
    shutil.copy(sample_audio, meeting_dir / sample_metadata["audio_filename"])

    return meeting_id
