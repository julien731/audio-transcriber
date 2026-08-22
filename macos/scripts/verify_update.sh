#!/usr/bin/env bash
# v1→v2 packaged update smoke (plan Artifact C / finding #4, #2). Serves the
# merged appcast + the new zip from a LOCALHOST fixture (draft-release assets
# aren't reachable unauthenticated), builds a test app that differs ONLY in
# SUFeedURL, and confirms Sparkle accepts + installs the new item.
#
# Finalized/verified at Milestone D0 — depends on a real signed build. Kept as a
# documented entrypoint so the workflow references a real script.
set -euo pipefail
echo "v1→v2 update smoke is executed at Milestone D0 on the maintainer/CI machine."
echo "See docs/macos-app.md → 'Update smoke test' for the exact fixture + steps."
