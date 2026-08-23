#!/usr/bin/env bash
# Package the bundled service into a pinned, verifiable release asset (Decision B,
# plan finding #3). Run on the arm64 build machine after `pyinstaller
# MeetingTranscriber.spec` (see docs/packaging.md). Produces:
#
#   macos/build/MeetingTranscriberService-<version>.zip   — upload as a release asset
#   macos/service-manifest.json                           — commit (the pinned expectation)
#
# The zip embeds its own service-manifest.json (identity fields, no outer sha256);
# the committed manifest adds the outer sha256. release-macos.yml's fetch_service.sh
# downloads the asset, checks the sha256, then checks the embedded identity fields.
#
# Env:
#   VERSION        service release tag (e.g. service-v1.0.0). Required.
#   SOURCE_COMMIT  backend commit the service was built from. Default: git HEAD.
#   DIST           PyInstaller onedir output. Default: dist/MeetingTranscriber.
set -euo pipefail

: "${VERSION:?VERSION (service release tag, e.g. service-v1.0.0) required}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DIST="${DIST:-${ROOT}/dist/MeetingTranscriber}"
SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "${ROOT}" rev-parse HEAD)}"
OUT_DIR="${ROOT}/macos/build"
mkdir -p "${OUT_DIR}"

[ -x "${DIST}/MeetingTranscriber" ] || { echo "::error::service executable not found at ${DIST}/MeetingTranscriber (run pyinstaller first)"; exit 1; }
file "${DIST}/MeetingTranscriber" | grep -q arm64 || { echo "::error::service executable is not arm64"; exit 1; }

INSTALLED_KB=$(du -sk "${DIST}" | awk '{print $1}')
PYINSTALLER_VER=$("${ROOT}/.venv/bin/pyinstaller" --version 2>/dev/null || echo "unknown")
PYTHON_VER=$("${ROOT}/.venv/bin/python" -c "import platform;print(platform.python_version())" 2>/dev/null || echo "unknown")

# 1) Embedded manifest (identity + provenance; NO outer sha256 — that would be circular).
python3 - "${DIST}/service-manifest.json" "${VERSION}" "${SOURCE_COMMIT}" \
  "${INSTALLED_KB}" "${PYINSTALLER_VER}" "${PYTHON_VER}" <<'PY'
import json, sys
path, version, commit, installed_kb, pyi, py = sys.argv[1:]
json.dump({
    "service_version": version,
    "source_commit": commit,
    "architecture": "arm64",
    "installed_size_bytes": int(installed_kb) * 1024,
    "pyinstaller_version": pyi,
    "python_version": py,
}, open(path, "w"), indent=2)
print("embedded manifest:", path)
PY

# 2) Zip the onedir CONTENTS (no keepParent → extraction yields MeetingTranscriber,
#    _internal, service-manifest.json at the top, matching fetch_service.sh).
ZIP="${OUT_DIR}/MeetingTranscriberService-${VERSION}.zip"
rm -f "${ZIP}"
( cd "${DIST}" && ditto -c -k . "${ZIP}" )

SHA=$(shasum -a 256 "${ZIP}" | awk '{print $1}')
COMPRESSED=$(stat -f%z "${ZIP}")

# 3) Pinned manifest (committed in the app repo): adds the outer sha256.
python3 - "${ROOT}/macos/service-manifest.json" "${VERSION}" "${SOURCE_COMMIT}" \
  "${SHA}" "${COMPRESSED}" "${INSTALLED_KB}" <<'PY'
import json, sys
path, version, commit, sha, compressed, installed_kb = sys.argv[1:]
json.dump({
    "service_version": version,
    "source_commit": commit,
    "architecture": "arm64",
    "sha256": sha,
    "compressed_size_bytes": int(compressed),
    "installed_size_bytes": int(installed_kb) * 1024,
}, open(path, "w"), indent=2)
print("pinned manifest:", path)
PY

echo ""
echo "==> Packaged ${ZIP}"
echo "    sha256=${SHA}  size=$(( COMPRESSED / 1024 / 1024 )) MB"
echo ""
echo "Next:"
echo "  1. Publish the asset to a GitHub Release on tag ${VERSION}:"
echo "       gh release create ${VERSION} \"${ZIP}\" --repo <owner>/<repo> \\"
echo "         --title \"Bundled service ${VERSION}\" --notes \"Self-contained arm64 service for the native macOS app.\""
echo "     (or: gh release upload ${VERSION} \"${ZIP}\" --clobber  if the release exists)"
echo "  2. Commit macos/service-manifest.json (pins the app release to this artifact)."
