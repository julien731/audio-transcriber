#!/usr/bin/env bash
# Vendor a pinned, static, arm64-native ffmpeg + ffprobe into vendor/bin for the
# bundled service (OQ-6). The PyInstaller build (MeetingTranscriber.spec) copies
# vendor/bin into the bundle, and app_paths.ffmpeg_path() resolves it there.
#
# Static binaries are used so nothing dynamically links to dylibs that may be
# absent on a clean machine (why the Homebrew binary is unsuitable — see the
# spec's OQ-6). Run once on an Apple Silicon Mac before building.
set -euo pipefail

# Pin a specific evermeet.cx build for reproducibility. evermeet.cx publishes
# static macOS arm64 builds; bump these deliberately, don't float them.
FFMPEG_VERSION="${FFMPEG_VERSION:-7.1}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/vendor/bin"

mkdir -p "$DEST"

fetch() {
  local tool="$1"
  local url="https://evermeet.cx/ffmpeg/${tool}-${FFMPEG_VERSION}.zip"
  echo "Downloading ${tool} ${FFMPEG_VERSION} …"
  local tmp
  tmp="$(mktemp -d)"
  curl -fsSL "$url" -o "${tmp}/${tool}.zip"
  unzip -o -q "${tmp}/${tool}.zip" -d "$DEST"
  chmod +x "${DEST}/${tool}"
  rm -rf "$tmp"
}

fetch ffmpeg
fetch ffprobe

echo "Verifying architecture (expect arm64) …"
file "${DEST}/ffmpeg" "${DEST}/ffprobe"

echo "Vendored ffmpeg/ffprobe into ${DEST}"
