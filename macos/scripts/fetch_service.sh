#!/usr/bin/env bash
# Fetch + verify the pinned self-contained service artifact (plan Decision B,
# finding #3/#4). The service bundle is built ONCE per service version on an
# arm64 machine and published as a checksummed release asset; this downloads it
# and verifies its manifest against the app-repo's pinned expectation before
# embedding into macos/build/.../Contents/Resources/service/.
#
# Finalized at Milestone D0 (needs the real service artifact + hosting decided).
# Fails loudly rather than silently embedding an unverified/mismatched payload.
set -euo pipefail

MANIFEST="${MANIFEST:-macos/service-manifest.json}"
if [ ! -f "${MANIFEST}" ]; then
  echo "::error::${MANIFEST} not found. Milestone D0 must produce the pinned service manifest before releasing."
  exit 1
fi

# Expected values pinned in the app repo.
EXPECT_SHA=$(python3 -c "import json;print(json.load(open('${MANIFEST}'))['sha256'])")
EXPECT_TAG=$(python3 -c "import json;print(json.load(open('${MANIFEST}'))['service_version'])")
EXPECT_COMMIT=$(python3 -c "import json;print(json.load(open('${MANIFEST}'))['source_commit'])")

DEST="macos/build/service"
mkdir -p "${DEST}"
gh release download "${EXPECT_TAG}" --repo "${GITHUB_REPOSITORY:-nimblehq/audio-transcriber}" \
  --pattern 'MeetingTranscriberService-*.zip' --dir "${DEST}" --clobber

ARCHIVE=$(ls "${DEST}"/MeetingTranscriberService-*.zip | head -1)
ACTUAL_SHA=$(shasum -a 256 "${ARCHIVE}" | awk '{print $1}')
[ "${ACTUAL_SHA}" = "${EXPECT_SHA}" ] || { echo "::error::service sha256 mismatch (${ACTUAL_SHA} != ${EXPECT_SHA})"; exit 1; }

ditto -x -k "${ARCHIVE}" "${DEST}/unpacked"
# Verify embedded manifest matches, and arch of the executable + ffmpeg.
EMBEDDED="${DEST}/unpacked/service-manifest.json"
python3 - "$EMBEDDED" "$EXPECT_COMMIT" "$EXPECT_TAG" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
assert m["source_commit"]==sys.argv[2], f"commit mismatch: {m['source_commit']} != {sys.argv[2]}"
assert m["service_version"]==sys.argv[3], f"version mismatch: {m['service_version']} != {sys.argv[3]}"
assert m["architecture"]=="arm64", f"arch not arm64: {m['architecture']}"
print("service manifest verified")
PY
for bin in MeetingTranscriber bin/ffmpeg bin/ffprobe; do
  f="${DEST}/unpacked/${bin}"
  [ -e "${f}" ] && file "${f}" | grep -q arm64 || { echo "::error::${bin} is not arm64"; exit 1; }
done
echo "service artifact fetched + verified"
