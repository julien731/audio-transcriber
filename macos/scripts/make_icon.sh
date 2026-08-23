#!/usr/bin/env bash
# Turn a 1024x1024 PNG into macos/Resources/AppIcon.icns (the app icon).
# build_app.sh embeds AppIcon.icns into the bundle when present.
#
# Usage: bash macos/scripts/make_icon.sh path/to/icon-1024.png
set -euo pipefail

SRC="${1:?usage: make_icon.sh <1024x1024 png>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/Resources/AppIcon.icns"
SET="$(mktemp -d)/AppIcon.iconset"
mkdir -p "${SET}"

# Standard macOS iconset sizes (1x + 2x).
for size in 16 32 128 256 512; do
  sips -z "${size}" "${size}"       "${SRC}" --out "${SET}/icon_${size}x${size}.png"      >/dev/null
  sips -z $((size*2)) $((size*2))   "${SRC}" --out "${SET}/icon_${size}x${size}@2x.png"   >/dev/null
done

iconutil -c icns "${SET}" -o "${OUT}"
echo "==> wrote ${OUT}"
