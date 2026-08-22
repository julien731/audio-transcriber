#!/usr/bin/env bash
# Assemble the MeetingTranscriber .app bundle (plan Milestone D1 / Artifact D).
#
# D1 embeds the stub service (scripts/stub_service.py). Milestone D0 swaps in the
# real self-contained PyInstaller service under Contents/Resources/service/ and
# the Sparkle framework under Contents/Frameworks/ (slice 10) with inner→outer
# signing. This script ad-hoc signs the bundle (self-signing; plan BR-23).
set -euo pipefail

CONFIG="${1:-debug}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> swift build -c ${CONFIG}"
swift build -c "${CONFIG}" --product MeetingTranscriberApp
BIN_PATH="$(swift build -c "${CONFIG}" --show-bin-path)"

APP="${ROOT}/build/MeetingTranscriber.app"
CONTENTS="${APP}/Contents"
echo "==> assembling ${APP}"
rm -rf "${APP}"
mkdir -p "${CONTENTS}/MacOS" "${CONTENTS}/Resources/service" "${CONTENTS}/Frameworks"

cp "${BIN_PATH}/MeetingTranscriberApp" "${CONTENTS}/MacOS/MeetingTranscriber"
cp "${ROOT}/Resources/Info.plist" "${CONTENTS}/Info.plist"
cp "${ROOT}/scripts/stub_service.py" "${CONTENTS}/Resources/service/stub_service.py"

# Ad-hoc sign (self-signed). Real builds sign inner→outer once Sparkle is bundled.
echo "==> codesign (ad-hoc)"
codesign --force --deep --sign - \
	--entitlements "${ROOT}/Resources/MeetingTranscriber.entitlements" \
	"${APP}" || echo "warning: codesign failed (expected without a signing identity in some envs)"

echo "==> built ${APP}"
