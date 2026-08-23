#!/usr/bin/env bash
# Assemble the MeetingTranscriber .app bundle (plan Milestone D1 + Artifact C).
#
# D1 embeds the stub service (scripts/stub_service.py); Milestone D0 swaps in the
# real self-contained PyInstaller service under Contents/Resources/service/.
# Sparkle.framework is embedded and signed inner→outer; the app is ad-hoc signed
# (self-signing, BR-23). On a signed build the EdDSA appcast signature is the
# trust anchor, so ad-hoc signing is acceptable (with disable-library-validation).
set -euo pipefail

CONFIG="${1:-debug}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> swift build -c ${CONFIG}"
swift build -c "${CONFIG}" --product MeetingTranscriberApp
BIN_PATH="$(swift build -c "${CONFIG}" --show-bin-path)"

APP="${ROOT}/build/Blah.app"
CONTENTS="${APP}/Contents"
FRAMEWORKS="${CONTENTS}/Frameworks"
echo "==> assembling ${APP}"
rm -rf "${APP}"
mkdir -p "${CONTENTS}/MacOS" "${CONTENTS}/Resources/service" "${FRAMEWORKS}"

# The SwiftPM product is still MeetingTranscriberApp; copy it to the CFBundleExecutable name (Blah).
cp "${BIN_PATH}/MeetingTranscriberApp" "${CONTENTS}/MacOS/Blah"
cp "${ROOT}/Resources/Info.plist" "${CONTENTS}/Info.plist"

# STRICT mode (release builds): missing real service and any signing failure are
# fatal, and the finished bundle is verified. Set STRICT=1 in release-macos.yml.
STRICT="${STRICT:-0}"

# Embed the service: the real self-contained PyInstaller bundle if available
# (D0/release), else the dev stub (D1). SERVICE_DIST defaults to
# ../dist/MeetingTranscriber (local pyinstaller output); release-macos.yml sets it
# to the fetched+verified artifact.
SERVICE_DIST="${SERVICE_DIST:-${ROOT}/../dist/MeetingTranscriber}"
if [ -d "${SERVICE_DIST}" ]; then
  echo "==> embedding real service from ${SERVICE_DIST}"
  cp -R "${SERVICE_DIST}" "${CONTENTS}/Resources/service/MeetingTranscriber"
elif [ "${STRICT}" = "1" ]; then
  echo "::error::real service bundle not found at ${SERVICE_DIST}; refusing to ship the dev stub in a release build." >&2
  exit 1
else
  echo "==> embedding dev stub service (no real bundle at ${SERVICE_DIST})"
  cp "${ROOT}/scripts/stub_service.py" "${CONTENTS}/Resources/service/stub_service.py"
fi

# Embed Sparkle.framework preserving symlinks + executable permissions.
if [ -d "${BIN_PATH}/Sparkle.framework" ]; then
  cp -RP "${BIN_PATH}/Sparkle.framework" "${FRAMEWORKS}/"
fi

SIGN_IDENTITY="${CODESIGN_IDENTITY:--}"   # ad-hoc by default; override for a real identity
ENTITLEMENTS="${ROOT}/Resources/MeetingTranscriber.entitlements"

sign() { codesign --force --options runtime --timestamp=none --sign "${SIGN_IDENTITY}" "$@"; }

# In STRICT mode xargs propagates codesign failures (exit non-zero on any error);
# in dev mode failures are tolerated (some envs lack a usable signing context).
if [ "${STRICT}" = "1" ]; then
  SIGN_MANY() { xargs -0 -P8 -n1 codesign --force --sign "${SIGN_IDENTITY}"; }
else
  SIGN_MANY() { xargs -0 -P8 -n1 codesign --force --sign "${SIGN_IDENTITY}" 2>/dev/null || true; }
fi

# Sign every mach-o in the embedded service. Unsigned dylibs make macOS validate
# the whole ML tree on first load, which can take minutes on a cold cache and made
# the app time out waiting for readiness. Signing makes validation fast and
# cacheable. Parallelized — the service ships hundreds of dylibs.
SERVICE_EMBED="${CONTENTS}/Resources/service"
if [ -d "${SERVICE_EMBED}" ]; then
  echo "==> signing embedded service mach-o"
  find "${SERVICE_EMBED}" -type f \( -name "*.so" -o -name "*.dylib" \) -print0 | SIGN_MANY
  find "${SERVICE_EMBED}" -type f -perm +111 ! -name "*.so" ! -name "*.dylib" -print0 2>/dev/null | SIGN_MANY
fi

# Inner→outer signing order for Sparkle (XPC services → helpers → framework → app).
FW="${FRAMEWORKS}/Sparkle.framework"
if [ -d "${FW}" ]; then
  for xpc in "${FW}/Versions/B/XPCServices/"*.xpc; do [ -e "${xpc}" ] && sign "${xpc}"; done
  [ -e "${FW}/Versions/B/Autoupdate" ] && sign "${FW}/Versions/B/Autoupdate"
  [ -d "${FW}/Versions/B/Updater.app" ] && sign "${FW}/Versions/B/Updater.app"
  sign "${FW}"
fi

echo "==> codesign app (${SIGN_IDENTITY})"
if [ "${STRICT}" = "1" ]; then
  # Release: signing must succeed, and the finished bundle must verify.
  codesign --force --sign "${SIGN_IDENTITY}" --entitlements "${ENTITLEMENTS}" "${APP}"
  echo "==> codesign --verify --deep --strict"
  codesign --verify --deep --strict --verbose=2 "${APP}"
else
  codesign --force --sign "${SIGN_IDENTITY}" --entitlements "${ENTITLEMENTS}" "${APP}" \
    || echo "warning: codesign failed (expected without a signing identity in some envs)"
fi

echo "==> built ${APP}"
