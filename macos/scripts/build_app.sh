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

APP="${ROOT}/build/MeetingTranscriber.app"
CONTENTS="${APP}/Contents"
FRAMEWORKS="${CONTENTS}/Frameworks"
echo "==> assembling ${APP}"
rm -rf "${APP}"
mkdir -p "${CONTENTS}/MacOS" "${CONTENTS}/Resources/service" "${FRAMEWORKS}"

cp "${BIN_PATH}/MeetingTranscriberApp" "${CONTENTS}/MacOS/MeetingTranscriber"
cp "${ROOT}/Resources/Info.plist" "${CONTENTS}/Info.plist"

# Embed the service: the real self-contained PyInstaller bundle if available
# (D0), else the dev stub (D1). SERVICE_DIST defaults to ../dist/MeetingTranscriber
# (pyinstaller output). The app's AppState prefers the real executable.
SERVICE_DIST="${SERVICE_DIST:-${ROOT}/../dist/MeetingTranscriber}"
if [ -d "${SERVICE_DIST}" ]; then
  echo "==> embedding real service from ${SERVICE_DIST}"
  cp -R "${SERVICE_DIST}" "${CONTENTS}/Resources/service/MeetingTranscriber"
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

# Inner→outer signing order for Sparkle (XPC services → helpers → framework → app).
FW="${FRAMEWORKS}/Sparkle.framework"
if [ -d "${FW}" ]; then
  for xpc in "${FW}/Versions/B/XPCServices/"*.xpc; do [ -e "${xpc}" ] && sign "${xpc}"; done
  [ -e "${FW}/Versions/B/Autoupdate" ] && sign "${FW}/Versions/B/Autoupdate"
  [ -d "${FW}/Versions/B/Updater.app" ] && sign "${FW}/Versions/B/Updater.app"
  sign "${FW}"
fi

echo "==> codesign app (${SIGN_IDENTITY})"
codesign --force --sign "${SIGN_IDENTITY}" --entitlements "${ENTITLEMENTS}" "${APP}" \
  || echo "warning: codesign failed (expected without a signing identity in some envs)"

echo "==> built ${APP}"
