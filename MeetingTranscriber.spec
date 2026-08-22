# PyInstaller spec for the bundled local transcription service (BR-1, BR-2, BR-3).
#
# Produces a self-contained, Apple-Silicon service that embeds the Python
# interpreter, all libraries, and the vendored ffmpeg/ffprobe binaries. Build on
# an arm64 Mac after running scripts/vendor_ffmpeg.sh:
#
#     ./scripts/vendor_ffmpeg.sh
#     pyinstaller MeetingTranscriber.spec
#
# See docs/packaging.md. This spec is scaffolding: the multi-GB torch/whisperx/
# pyannote artifact is produced and verified on the build machine, not in CI.

from PyInstaller.utils.hooks import collect_all, collect_data_files

block_cipher = None

# ML dependency trees ship data files, dynamic libs, and lazy submodules that
# PyInstaller cannot infer statically — collect them wholesale.
_datas, _binaries, _hiddenimports = [], [], []
for pkg in ("whisperx", "torch", "torchaudio", "faster_whisper", "ctranslate2", "pyannote", "lightning_fabric"):
    try:
        d, b, h = collect_all(pkg)
        _datas += d
        _binaries += b
        _hiddenimports += h
    except Exception:
        # A package absent at build time is surfaced by the build log; keep going
        # so the spec documents intent even on an incomplete dev machine.
        pass

# App resources served/read at runtime.
_datas += [
    ("frontend", "frontend"),
    ("templates", "templates"),
]

# Vendored native binaries (see scripts/vendor_ffmpeg.sh) → bundle bin/.
_binaries += [
    ("vendor/bin/ffmpeg", "bin"),
    ("vendor/bin/ffprobe", "bin"),
]

a = Analysis(
    ["service_main.py"],
    pathex=["."],
    binaries=_binaries,
    datas=_datas,
    hiddenimports=_hiddenimports + ["huggingface_hub", "uvicorn.logging", "uvicorn.protocols"],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MeetingTranscriber",
    console=True,  # stdout handshake is part of the client contract (OQ-2)
    target_arch="arm64",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="MeetingTranscriber",
)
