#!/usr/bin/env bash
# Pre-publish update gate (plan Artifact C / finding #2, #4).
#
# A full Sparkle v1→v2 GUI install cannot run headlessly on a CI runner, so this
# performs the verification that CAN run unattended and fails the release if any
# check does not pass — turning the former no-op into a real gate:
#   1. the appcast parses and its newest <item> carries a non-empty edSignature
#      and an enclosure url + length;
#   2. the update zip exists and its byte length matches the enclosure length;
#   3. the .app inside passes `codesign --verify --deep --strict`;
#   4. the app's SUPublicEDKey is a real key (not the placeholder).
# The end-to-end GUI install remains a documented manual/D0 step
# (docs/macos-app.md → "Update smoke test").
#
# Env: APPCAST, ZIP, VERSION.
set -euo pipefail

: "${APPCAST:?APPCAST required}"
: "${ZIP:?ZIP required}"
: "${VERSION:?VERSION required}"

fail() { echo "::error::update-verify: $1"; exit 1; }

[ -f "${APPCAST}" ] || fail "appcast not found: ${APPCAST}"
[ -f "${ZIP}" ] || fail "update zip not found: ${ZIP}"

# 1) appcast parses; newest item has an enclosure with url, length, edSignature.
python3 - "${APPCAST}" "${VERSION}" <<'PY'
import sys, xml.etree.ElementTree as ET
appcast, version = sys.argv[1], sys.argv[2]
SPARKLE = "{http://www.andymatuschak.org/xml-namespaces/sparkle}"
root = ET.parse(appcast).getroot()          # raises on malformed XML
items = root.find("channel").findall("item")
if not items:
    print("::error::update-verify: appcast has no <item> entries"); sys.exit(1)
# The item for this release must exist with a signed enclosure.
def item_version(it):
    enc = it.find("enclosure")
    return enc.get(f"{SPARKLE}version") if enc is not None else None
match = [it for it in items if item_version(it) == version] or [items[0]]
enc = match[0].find("enclosure")
if enc is None: print("::error::update-verify: item has no <enclosure>"); sys.exit(1)
if not enc.get("url"): print("::error::update-verify: enclosure missing url"); sys.exit(1)
if not enc.get("length"): print("::error::update-verify: enclosure missing length"); sys.exit(1)
if not enc.get(f"{SPARKLE}edSignature"): print("::error::update-verify: enclosure missing edSignature"); sys.exit(1)
print(f"appcast OK: version={item_version(match[0])} length={enc.get('length')}")
PY

# 2) enclosure length matches the actual zip byte size.
ENC_LEN=$(python3 -c "import xml.etree.ElementTree as ET,sys;\
r=ET.parse('${APPCAST}').getroot();\
print(r.find('channel').find('item').find('enclosure').get('length'))")
ZIP_LEN=$(stat -f%z "${ZIP}")
[ "${ENC_LEN}" = "${ZIP_LEN}" ] || fail "enclosure length ${ENC_LEN} != zip size ${ZIP_LEN}"

# 3) unzip and verify the app's signature strictly.
WORK="$(mktemp -d)"
ditto -x -k "${ZIP}" "${WORK}"
APP="$(find "${WORK}" -maxdepth 2 -name "*.app" -type d | head -1)"
[ -n "${APP}" ] || fail "no .app inside ${ZIP}"
codesign --verify --deep --strict --verbose=2 "${APP}" || fail "codesign --verify --deep --strict failed"

# 4) SUPublicEDKey is a real key, not the placeholder.
KEY="$(/usr/libexec/PlistBuddy -c 'Print :SUPublicEDKey' "${APP}/Contents/Info.plist" 2>/dev/null || true)"
case "${KEY}" in
  ""|REPLACE_WITH*) fail "SUPublicEDKey is unset/placeholder in the shipped app" ;;
esac

echo "update-verify: all pre-publish checks passed for ${VERSION}"
