#!/usr/bin/env bash
# Produce/extend the Sparkle appcast for a new release (plan Artifact C, finding #4).
#
# generate_appcast operates over an archive DIRECTORY, so we generate the NEW
# item from a directory containing only the new zip (correct namespace, enclosure
# length, version fields, MIME type, edSignature, minimumSystemVersion), then
# splice retained prior <item> entries verbatim. Deltas are disabled (historical
# archives are not retained). Prior enclosure URLs point at still-live releases.
#
# Inputs (env):
#   VERSION                 new short version string (release tag)
#   ZIP_PATH                path to the new signed app zip
#   ENCLOSURE_URL           public URL the new zip will live at
#   SPARKLE_BIN             dir containing generate_appcast + sign_update
#   PREVIOUS_APPCAST        path to the current appcast.xml (may be absent)
#   OUT_APPCAST             output path
#   RETAIN                  number of prior <item>s to keep (default 10)
#   MIN_SYSTEM_VERSION      e.g. 13.0
set -euo pipefail

RETAIN="${RETAIN:-10}"
WORKDIR="$(mktemp -d)"
mkdir -p "${WORKDIR}/archives"
cp "${ZIP_PATH}" "${WORKDIR}/archives/"

# Generate an appcast for JUST the new archive (Sparkle fills all required fields
# and signs the enclosure with the EdDSA key it finds via SPARKLE_ED_PRIVATE_KEY).
"${SPARKLE_BIN}/generate_appcast" \
  --no-delta \
  --download-url-prefix "$(dirname "${ENCLOSURE_URL}")/" \
  --minimum-system-version "${MIN_SYSTEM_VERSION:-13.0}" \
  "${WORKDIR}/archives"

NEW_APPCAST="${WORKDIR}/archives/appcast.xml"

# Merge: keep the new item, then splice up to RETAIN prior <item>s verbatim.
python3 - "$NEW_APPCAST" "${PREVIOUS_APPCAST:-}" "$OUT_APPCAST" "$RETAIN" <<'PY'
import sys, xml.etree.ElementTree as ET

new_path, prev_path, out_path, retain = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
ET.register_namespace('sparkle', 'http://www.andymatuschak.org/xml-namespaces/sparkle')

new_tree = ET.parse(new_path)
channel = new_tree.getroot().find('channel')
new_items = channel.findall('item')

# Validate the new item has an enclosure with a signature before publishing.
for item in new_items:
    enc = item.find('enclosure')
    assert enc is not None, "new item missing enclosure"
    assert enc.get('{http://www.andymatuschak.org/xml-namespaces/sparkle}edSignature'), "new item missing edSignature"

if prev_path:
    try:
        prev_items = ET.parse(prev_path).getroot().find('channel').findall('item')
        for item in prev_items[:retain]:
            channel.append(item)
    except (FileNotFoundError, ET.ParseError):
        pass

new_tree.write(out_path, encoding='utf-8', xml_declaration=True)
# Fail if the merged feed does not parse (validated before publish).
ET.parse(out_path)
print(f"appcast written: {out_path}")
PY

echo "==> appcast at ${OUT_APPCAST}"
